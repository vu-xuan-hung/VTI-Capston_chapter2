"""Chunking: windows overlap, ids are stable, documents load from a folder."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
from    pathlib                         import  Path

import  pytest

from    factdesk.knowledge.chunking     import  SourceDoc, chunk_documents, chunk_words, load_documents


#==========================================================================
# Windows and overlap
#==========================================================================
def test_windows_repeat_the_overlap_and_end_cleanly() -> None:
    """
    Pin size/overlap arithmetic: 10 words, size 4, overlap 1 -> 3 windows.
    """
    words   = " ".join(f"w{i}" for i in range(10))
    windows = chunk_words(words, size=4, overlap=1)
    assert windows == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert chunk_words("", size=4, overlap=1) == [""]
    with pytest.raises(ValueError, match="overlap"):
        chunk_words(words, size=4, overlap=4)


#==========================================================================
# Stable ids
#==========================================================================
def test_chunk_ids_are_stable_and_carry_metadata() -> None:
    """
    Pin "<source_id>#NNN" ids: chunking twice gives the same ids and texts.
    """
    docs   = [SourceDoc("guide", "Guide", "one two three four five six", {"lang": "en"})]
    first  = chunk_documents(docs, size=4, overlap=1)
    second = chunk_documents(docs, size=4, overlap=1)
    assert [chunk.chunk_id for chunk in first] == ["guide#001", "guide#002"]
    assert first == second
    assert first[0].metadata == {"lang": "en"}


#==========================================================================
# Load a folder
#==========================================================================
def test_load_documents_reads_markdown_and_json(tmp_path: Path) -> None:
    """
    Pin the two input formats and the duplicate-id check.

    Input:
        . tmp_path      : temporary documents folder
    """
    (tmp_path / "intro.md").write_text("# Intro\nHello there.\n", encoding="utf-8")
    rows = [{"source_id": "faq", "title": "FAQ", "text": "Q and A", "lang": "vi"}]
    (tmp_path / "more.json").write_text(json.dumps(rows), encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [(doc.source_id, doc.title) for doc in docs] == [("intro", "Intro"), ("faq", "FAQ")]
    assert docs[1].metadata == {"lang": "vi"}
    (tmp_path / "faq.md").write_text("again", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated source_id"):
        load_documents(tmp_path)
