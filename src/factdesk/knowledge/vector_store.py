"""Keep chunks in a Chroma collection and search them by meaning.

One class for both stores: embedded (a folder) and the Chroma server that
Langflow's Chroma DB component reads, so the code and the flow share one
collection. Each chunk is stored with its text as the document and chunk_id,
source_id, title and its own metadata as Chroma metadata (what a Langflow
Parser pattern such as "[{chunk_id}] {text}" reads).
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    collections.abc                 import  Callable
from    dataclasses                     import  dataclass
from    functools                       import  partial
from    typing                          import  Any

from    langchain_core.callbacks        import  CallbackManagerForRetrieverRun
from    langchain_core.documents        import  Document
from    langchain_core.embeddings       import  Embeddings
from    langchain_core.retrievers       import  BaseRetriever

from    factdesk.ai_logger                       import  aiLogger
from    factdesk.config                          import  Config, KnowledgeConfig
from    factdesk.integrations.chroma_server      import  open_chroma_client
from    factdesk.integrations.server_embeddings  import  ServerEmbeddings
from    factdesk.knowledge.chunking              import  Chunk
from    factdesk.knowledge.hash_embeddings       import  HashEmbeddings


#==========================================================================
# One search result
#==========================================================================
@dataclass(frozen=True)
class Hit:
    """
    Purpose: one chunk found by a search, with its cosine similarity
    (1 - cosine distance: 1 = same direction, 0 = unrelated).
    """

    chunk_id: str
    source_id: str
    title: str
    text: str
    score: float
    metadata: dict[str, Any]


#==========================================================================
# What one indexing run changed
#==========================================================================
@dataclass(frozen=True)
class IndexReport:
    """
    Purpose: the counts of one index() call; a second run on the same chunks
    adds, changes and removes nothing.
    """

    added: int
    changed: int
    removed: int
    unchanged: int
    total: int


#==========================================================================
# Chroma collection of chunks
#==========================================================================
class VectorStore:
    """
    Purpose: open the collection on first use (cosine space), upsert chunks
    by id, drop ids that left the chunk set, and search with an optional
    metadata filter. Refuses a collection built with another embedding model.
    """

    #----------------------------------------------------------------------
    # Keep the settings; nothing is opened here
    #----------------------------------------------------------------------
    def __init__(
        self,
        knowledge: KnowledgeConfig,
        *,
        open_client: Callable[[], Any],
        open_embeddings: Callable[[], Embeddings],
        embedding_name: str,
        batch_size: int = 256,
    ) -> None:
        """
        Remember how to open the client and the embedding model.

        Input:
            . knowledge     : collection name, top_k, threshold
            . open_client   : returns a chromadb client (embedded or HTTP)
            . open_embeddings: returns the LangChain embeddings
            . embedding_name: recorded on the collection and checked on open
            . batch_size    : chunks embedded and upserted per request
        """
        self._knowledge       = knowledge
        self._open_client     = open_client
        self._open_embeddings = open_embeddings
        self._embedding_name  = embedding_name
        self._batch_size      = batch_size
        self._collection: Any = None
        self._embeddings: Embeddings | None = None

    #----------------------------------------------------------------------
    # Build or refresh the collection
    #----------------------------------------------------------------------
    def index(self, chunks: list[Chunk]) -> IndexReport:
        """
        Make the collection hold exactly these chunks: new and changed ones
        are embedded and upserted, unchanged ones are left alone, ids that are
        no longer produced are deleted.

        Input:
            . chunks        : the full chunk set (from chunk_documents)
        Output:
            . report        : added / changed / removed / unchanged / total
        """

        #------------------------------------------------------------------
        # Compare the chunk set with what the collection holds
        #------------------------------------------------------------------
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicated chunk ids: {len(ids) - len(set(ids))} repeats")
        collection = self._open()
        stored     = collection.get(include=["documents", "metadatas"])
        old        = {
            chunk_id: (text, metadata)
            for chunk_id, text, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"])
        }
        todo  = [chunk for chunk in chunks if old.get(chunk.chunk_id) != (chunk.text, _metadata(chunk))]
        stale = sorted(set(old) - set(ids))

        #------------------------------------------------------------------
        # Upsert in batches, then delete the stale ids
        #------------------------------------------------------------------
        for start in range(0, len(todo), self._batch_size):
            batch = todo[start : start + self._batch_size]
            collection.upsert(
                ids         = [chunk.chunk_id for chunk in batch],
                documents   = [chunk.text for chunk in batch],
                metadatas   = [_metadata(chunk) for chunk in batch],
                embeddings  = self._embed().embed_documents([chunk.text for chunk in batch]),
            )
        if stale:
            collection.delete(ids=stale)
        added = sum(chunk.chunk_id not in old for chunk in todo)
        aiLogger.info("Collection %s: %d chunks", self._knowledge.collection, collection.count())
        return IndexReport(
            added       = added,
            changed     = len(todo) - added,
            removed     = len(stale),
            unchanged   = len(chunks) - len(todo),
            total       = collection.count(),
        )

    #----------------------------------------------------------------------
    # Search by meaning
    #----------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """
        Return the k nearest chunks with their scores, best first; no
        threshold here (the retriever applies it).

        Input:
            . query         : the question
            . k             : number of chunks; None = knowledge.top_k
            . where         : Chroma metadata filter, e.g. {"lang": "vi"}, or None
        Output:
            . hits          : up to k hits, highest score first
        """

        #------------------------------------------------------------------
        # Query the nearest chunks
        #------------------------------------------------------------------
        collection = self._open()
        found      = collection.query(
            query_embeddings    = [self._embed().embed_query(query)],
            n_results           = k or self._knowledge.top_k,
            where               = where or None,
            include             = ["documents", "metadatas", "distances"],
        )
        #------------------------------------------------------------------
        # Turn distances into scored hits
        #------------------------------------------------------------------
        return [
            Hit(
                chunk_id    = chunk_id,
                source_id   = metadata["source_id"],
                title       = metadata["title"],
                text        = text,
                score       = round(1.0 - distance, 4),
                metadata    = {
                    key: value for key, value in metadata.items()
                    if key not in ("chunk_id", "source_id", "title")
                },
            )
            for chunk_id, text, metadata, distance in zip(
                found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0]
            )
        ]

    #----------------------------------------------------------------------
    # LangChain retriever with the score threshold
    #----------------------------------------------------------------------
    def retriever(
        self,
        *,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> BaseRetriever:
        """
        Wrap search() as a LangChain retriever that drops chunks below
        knowledge.threshold; it shows up as a retriever run in traces.

        Input:
            . k             : number of chunks searched; None = knowledge.top_k
            . where         : Chroma metadata filter, or None
        Output:
            . retriever     : Documents with chunk_id, source_id, title, score metadata
        """
        return ThresholdRetriever(
            store       = self,
            k           = k or self._knowledge.top_k,
            threshold   = self._knowledge.threshold,
            where       = where,
        )

    #----------------------------------------------------------------------
    # Number of chunks
    #----------------------------------------------------------------------
    def count(self) -> int:
        """
        Count the chunks in the collection.

        Output:
            . count         : number of stored chunks
        """
        return self._open().count()

    #----------------------------------------------------------------------
    # Open the collection once
    #----------------------------------------------------------------------
    def _open(self) -> Any:
        """
        Get or create the collection in cosine space and check that it was
        built with the same embedding model.

        Output:
            . collection    : the chromadb collection
        """

        #------------------------------------------------------------------
        # Get or create the collection
        #------------------------------------------------------------------
        if self._collection is not None:
            return self._collection
        collection = self._open_client().get_or_create_collection(
            name                = self._knowledge.collection,
            configuration       = {"hnsw": {"space": "cosine"}},
            metadata            = {"embedding_model": self._embedding_name},
            embedding_function  = None,
        )
        #------------------------------------------------------------------
        # Refuse vectors of another embedding model
        #------------------------------------------------------------------
        built_with = (collection.metadata or {}).get("embedding_model")
        if built_with not in (None, self._embedding_name):
            raise ValueError(
                f"collection {self._knowledge.collection!r} was built with {built_with!r}, "
                f"not {self._embedding_name!r}: delete it (or use another collection name) and index again"
            )
        self._collection = collection
        return collection

    #----------------------------------------------------------------------
    # Load the embedding model once
    #----------------------------------------------------------------------
    def _embed(self) -> Embeddings:
        """
        Create the embeddings the first time they are needed.

        Output:
            . embeddings    : the LangChain embeddings
        """
        if self._embeddings is None:
            self._embeddings = self._open_embeddings()
        return self._embeddings


#==========================================================================
# Retriever with a score threshold
#==========================================================================
class ThresholdRetriever(BaseRetriever):
    """
    Purpose: a LangChain retriever over VectorStore.search that keeps hits at
    or above the threshold; an empty list means "nothing relevant".
    """

    store: Any
    k: int
    threshold: float
    where: dict[str, Any] | None        = None

    #----------------------------------------------------------------------
    # Search and filter
    #----------------------------------------------------------------------
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """
        Return the hits above the threshold as Documents.

        Input:
            . query         : the question
            . run_manager   : LangChain callback manager
        Output:
            . documents     : best first; metadata has chunk_id, source_id, title, score
        """
        return [
            Document(
                page_content    = hit.text,
                id              = hit.chunk_id,
                metadata        = {
                    **hit.metadata,
                    "chunk_id": hit.chunk_id,
                    "source_id": hit.source_id,
                    "title": hit.title,
                    "score": hit.score,
                },
            )
            for hit in self.store.search(query, k=self.k, where=self.where)
            if hit.score >= self.threshold
        ]


#==========================================================================
# Wire the store for the configured backend
#==========================================================================
def open_vector_store(config: Config) -> VectorStore:
    """
    Create the store with the embeddings of the backend: hashing for fake,
    the model server for server, sentence-transformers here for local. The
    server and local paths compute the same vectors, under the same name.

    Input:
        . config        : the whole configuration
    Output:
        . store         : a VectorStore; nothing is opened until first use
    """
    model = config.model
    name  = model.embedding_model
    make: Callable[[], Embeddings] = partial(_local_embeddings, name)
    if model.backend == "fake":
        make = HashEmbeddings
        name = HashEmbeddings().name
    elif model.backend == "server":
        make = partial(ServerEmbeddings, config.servers.model_url, name, timeout=config.servers.request_timeout)
    client = partial(open_chroma_client, config.knowledge, config.servers, kb_dir=config.paths.kb_dir)
    return VectorStore(config.knowledge, open_client=client, open_embeddings=make, embedding_name=name)


#==========================================================================
# Chunk metadata as Chroma stores it
#==========================================================================
def _metadata(chunk: Chunk) -> dict[str, Any]:
    """
    Merge the chunk's own metadata with its id fields.

    Input:
        . chunk         : one chunk
    Output:
        . metadata      : flat str/int/float/bool values for Chroma
    """
    return {**chunk.metadata, "chunk_id": chunk.chunk_id, "source_id": chunk.source_id, "title": chunk.title}


#==========================================================================
# Local sentence-transformers embeddings
#==========================================================================
def _local_embeddings(model_name: str) -> Embeddings:
    """
    Load the embedding model from the Hugging Face cache; imported here so
    the fake and server paths never load torch.

    Input:
        . model_name    : Hugging Face id
    Output:
        . embeddings    : HuggingFaceEmbeddings
    """
    from    langchain_huggingface           import  HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)
