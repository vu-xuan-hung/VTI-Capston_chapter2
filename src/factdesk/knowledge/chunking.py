"""Load documents and cut them into word windows with stable ids.

A chunk id is "<source_id>#<number>": the same documents and settings give
the same ids, so indexing again updates chunks instead of copying them.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
from    dataclasses                     import  dataclass, field
from    pathlib                         import  Path
from    typing                          import  Any


#==========================================================================
# One source document
#==========================================================================
@dataclass(frozen=True)
class SourceDoc:
    """
    Purpose: one document before chunking; metadata travels to every chunk
    and can be used in a search filter (Chroma `where`).
    """

    source_id: str
    title: str
    text: str
    metadata: dict[str, Any]            = field(default_factory=dict)


#==========================================================================
# One chunk
#==========================================================================
@dataclass(frozen=True)
class Chunk:
    """
    Purpose: one searchable piece of a document, with its stable id.
    """

    chunk_id: str
    source_id: str
    title: str
    text: str
    metadata: dict[str, Any]            = field(default_factory=dict)


#==========================================================================
# Load documents from a folder
#==========================================================================
def load_documents(folder: Path) -> list[SourceDoc]:
    """
    Read every .md / .txt file (source_id = file name without suffix, title =
    first "# " line) and every .json file (a list of {"source_id", "title",
    "text", other keys = metadata}), sorted by file name.

    Input:
        . folder        : the documents folder, e.g. data/docs
    Output:
        . docs          : documents with unique source ids
    """

    #----------------------------------------------------------------------
    # Read the files
    #----------------------------------------------------------------------
    docs: list[SourceDoc] = []
    for path in sorted(folder.iterdir()):
        if path.suffix in (".md", ".txt"):
            text  = path.read_text(encoding="utf-8")
            title = next(
                (line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem
            )
            docs.append(SourceDoc(path.stem, title, text, {"file": path.name}))
        elif path.suffix == ".json":
            for row in json.loads(path.read_text(encoding="utf-8")):
                extra = {key: value for key, value in row.items() if key not in ("source_id", "title", "text")}
                docs.append(SourceDoc(row["source_id"], row["title"], row["text"], extra))

    #----------------------------------------------------------------------
    # Check the ids and the metadata Chroma can store
    #----------------------------------------------------------------------
    ids = [doc.source_id for doc in docs]
    duplicated = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
    if duplicated:
        raise ValueError(f"{folder}: duplicated source_id {duplicated}")
    if not docs:
        raise ValueError(f"{folder}: no .md, .txt or .json document")
    for doc in docs:
        bad = [key for key, value in doc.metadata.items() if not isinstance(value, str | int | float | bool)]
        if bad:
            raise ValueError(f"{doc.source_id}: metadata {bad} must be str, int, float or bool")
    return docs


#==========================================================================
# Cut documents into chunks
#==========================================================================
def chunk_documents(docs: list[SourceDoc], *, size: int, overlap: int) -> list[Chunk]:
    """
    Cut every document into word windows; ids number the windows per source.

    Input:
        . docs          : documents from load_documents
        . size          : words per chunk
        . overlap       : words each chunk repeats from the previous one
    Output:
        . chunks        : all chunks, document order kept
    """
    return [
        Chunk(f"{doc.source_id}#{number:03d}", doc.source_id, doc.title, text, dict(doc.metadata))
        for doc in docs
        for number, text in enumerate(chunk_words(doc.text, size=size, overlap=overlap), 1)
    ]


#==========================================================================
# Word windows
#==========================================================================
def chunk_words(text: str, *, size: int, overlap: int) -> list[str]:
    """
    Cut text into windows of `size` words; each repeats `overlap` words of
    the previous one, and no window holds only repeated words.

    Input:
        . text          : any text; whitespace is normalised
        . size          : words per window
        . overlap       : words shared with the previous window, < size
    Output:
        . windows       : at least one window (empty text gives [""])
    """
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size): {overlap} vs {size}")
    words = text.split()
    step  = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, max(1, len(words) - overlap), step)]
