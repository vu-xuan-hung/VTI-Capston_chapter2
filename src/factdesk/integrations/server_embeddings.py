"""Ask the model server for embeddings, as Langflow's OpenAI Embeddings does."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    langchain_core.embeddings       import  Embeddings

from    factdesk.integrations.http_json  import  request_json


#==========================================================================
# Embeddings over HTTP
#==========================================================================
class ServerEmbeddings(Embeddings):
    """
    Purpose: the same vectors as the local HuggingFaceEmbeddings, computed by
    the model server; Langflow calls the same address.
    """

    #----------------------------------------------------------------------
    # Keep the address and the model name
    #----------------------------------------------------------------------
    def __init__(self, base_url: str, model_name: str, *, timeout: float = 300) -> None:
        """
        Remember where the server is.

        Input:
            . base_url      : e.g. http://127.0.0.1:8000/v1
            . model_name    : the embedding model name sent with each request
            . timeout       : seconds per request
        """
        self._base_url   = base_url
        self._model_name = model_name
        self._timeout    = timeout

    #----------------------------------------------------------------------
    # Embed many texts in one request
    #----------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        POST /embeddings with all texts at once.

        Input:
            . texts         : the chunk texts
        Output:
            . vectors       : one vector per text, in order
        """
        body = {"model": self._model_name, "input": list(texts)}
        out  = request_json(f"{self._base_url}/embeddings", body, timeout=self._timeout)
        return [item["embedding"] for item in sorted(out["data"], key=lambda item: item["index"])]

    #----------------------------------------------------------------------
    # Embed one question
    #----------------------------------------------------------------------
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a search query the same way as a document.

        Input:
            . text          : the query
        Output:
            . vector        : its embedding
        """
        return self.embed_documents([text])[0]
