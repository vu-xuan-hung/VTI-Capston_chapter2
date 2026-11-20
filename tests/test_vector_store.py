"""VectorStore: re-indexing makes no copies, stale ids go, filters and threshold work."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    pathlib                         import  Path

import  pytest
from    helpers                         import  hash_store

from    factdesk.knowledge.chunking     import  SourceDoc, chunk_documents


#==========================================================================
# Test documents
#==========================================================================
def _docs() -> list[SourceDoc]:
    """
    Three short documents in two languages.

    Output:
        . docs          : refund (en), shipping (en), doi tra (vi)
    """
    return [
        SourceDoc("refund", "Refunds", "refund money back within fourteen days of delivery", {"lang": "en"}),
        SourceDoc("shipping", "Shipping", "shipping takes three days by courier to the city", {"lang": "en"}),
        SourceDoc("doi-tra", "Doi tra", "doi tra hang trong muoi bon ngay refund money", {"lang": "vi"}),
    ]


#==========================================================================
# Re-indexing makes no duplicates
#==========================================================================
def test_index_twice_adds_nothing_the_second_time(tmp_path: Path) -> None:
    """
    Pin upsert by stable id: the second run changes nothing, the count stays.

    Input:
        . tmp_path      : temporary Chroma folder
    """
    chunks = chunk_documents(_docs(), size=5, overlap=1)
    store  = hash_store(tmp_path)
    first  = store.index(chunks)
    second = hash_store(tmp_path).index(chunks)
    assert (first.added, first.total) == (len(chunks), len(chunks))
    assert (second.added, second.changed, second.removed, second.unchanged) == (0, 0, 0, len(chunks))
    assert second.total == len(chunks)


#==========================================================================
# A changed chunk set removes stale ids
#==========================================================================
def test_shrinking_the_chunk_set_removes_stale_ids(tmp_path: Path) -> None:
    """
    Pin that ids no longer produced are deleted and changed text is updated.

    Input:
        . tmp_path      : temporary Chroma folder
    """
    store = hash_store(tmp_path)
    store.index(chunk_documents(_docs(), size=5, overlap=1))
    edited = [SourceDoc("refund", "Refunds", "refund within thirty days", {"lang": "en"})]
    report = store.index(chunk_documents(edited, size=5, overlap=1))
    assert (report.changed, report.total) == (1, 1)
    assert report.removed > 0
    assert store.search("refund", k=5)[0].text == "refund within thirty days"


#==========================================================================
# Metadata filter
#==========================================================================
def test_search_with_a_metadata_filter_keeps_only_matching_chunks(tmp_path: Path) -> None:
    """
    Pin Chroma `where` filtering and the scored hit records.

    Input:
        . tmp_path      : temporary Chroma folder
    """
    store = hash_store(tmp_path)
    store.index(chunk_documents(_docs(), size=50, overlap=5))
    hits  = store.search("refund money", k=3)
    only  = store.search("refund money", k=3, where={"lang": "vi"})
    assert [hit.source_id for hit in only] == ["doi-tra"]
    assert only[0].metadata == {"lang": "vi"}
    assert hits[0].score >= hits[-1].score
    assert {hit.source_id for hit in hits} == {"refund", "shipping", "doi-tra"}


#==========================================================================
# Retriever threshold
#==========================================================================
def test_retriever_drops_chunks_below_the_threshold(tmp_path: Path) -> None:
    """
    Pin that the LangChain retriever applies the threshold and keeps the score.

    Input:
        . tmp_path      : temporary Chroma folder
    """
    hash_store(tmp_path).index(chunk_documents(_docs(), size=50, overlap=5))
    strict = hash_store(tmp_path, threshold=0.99).retriever(k=3).invoke("courier city")
    loose  = hash_store(tmp_path, threshold=0.2).retriever(k=3).invoke("courier city")
    assert strict == []
    assert loose[0].metadata["chunk_id"] == "shipping#001"
    assert loose[0].metadata["score"] >= 0.2


#==========================================================================
# Another embedding model is refused
#==========================================================================
def test_a_collection_built_with_other_embeddings_is_refused(tmp_path: Path) -> None:
    """
    Pin the embedding-model guard recorded on the collection.

    Input:
        . tmp_path      : temporary Chroma folder
    """
    store = hash_store(tmp_path)
    store.index(chunk_documents(_docs(), size=50, overlap=5))
    other = hash_store(tmp_path, embedding_name="sentence-transformers/other")
    with pytest.raises(ValueError, match="was built with"):
        other.count()
