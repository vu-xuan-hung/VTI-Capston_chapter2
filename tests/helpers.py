"""Shared test helpers: a fake config in a temporary folder, a hashing store."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
import  re
from    dataclasses                     import  replace
from    functools                       import  partial
from    pathlib                         import  Path
from    typing                          import  Any

from    factdesk.config                      import  Config, KnowledgeConfig, PathConfig, ServerConfig
from    factdesk.config                      import  load_config, with_backend
from    factdesk.integrations.chroma_server  import  open_chroma_client
from    factdesk.knowledge.hash_embeddings   import  HashEmbeddings
from    factdesk.knowledge.vector_store      import  VectorStore
from    factdesk.llm.chat_source             import  ChatSource
from    factdesk.llm.fake_chat_model         import  FakeRules

from    main                            import  main


ROOT     = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


#==========================================================================
# Fake config in a temporary folder
#==========================================================================
def fake_config(tmp_path: Path) -> Config:
    """
    Build the fake-backend config with every path inside tmp_path.

    Input:
        . tmp_path      : pytest's temporary folder
    Output:
        . config        : backend fake, embedded store, temporary paths
    """
    paths = PathConfig(
        root        = tmp_path,
        data_dir    = tmp_path / "data",
        docs_dir    = tmp_path / "docs",
        runtime_dir = tmp_path / "runtime",
        results_dir = tmp_path / "results",
        kb_dir      = tmp_path / "kb_store",
    )
    return with_backend(replace(Config(), paths=paths), "fake")


#==========================================================================
# Embedded store with hashing vectors
#==========================================================================
def hash_store(
    tmp_path: Path,
    *,
    collection: str = "test-kb",
    threshold: float = 0.1,
    embedding_name: str = "hash-64",
) -> VectorStore:
    """
    Open a VectorStore in tmp_path with HashEmbeddings: no model download.

    Input:
        . tmp_path      : pytest's temporary folder
        . collection    : collection name (3+ characters, Chroma's rule)
        . threshold     : retriever score threshold
        . embedding_name: the name recorded on the collection
    Output:
        . store         : an unopened VectorStore
    """
    knowledge = KnowledgeConfig(collection=collection, threshold=threshold)
    client    = partial(open_chroma_client, knowledge, ServerConfig(), kb_dir=tmp_path / "kb_store")
    return VectorStore(
        knowledge,
        open_client     = client,
        open_embeddings = HashEmbeddings,
        embedding_name  = embedding_name,
    )


#==========================================================================
# The project's own config, with runtime files in a temporary folder
#==========================================================================
def project_toml(tmp_path: Path) -> Path:
    """
    Copy configs/app.toml with the runtime, results and Chroma folders moved
    into tmp_path; data and docs stay where they are (read only).

    Input:
        . tmp_path      : pytest's temporary folder
    Output:
        . path          : the TOML file to pass with --config
    """
    text = (ROOT / "configs" / "app.toml").read_text(encoding="utf-8")
    for key in ("runtime_dir", "results_dir", "kb_dir"):
        folder = tmp_path / key.removesuffix("_dir")
        value  = json.dumps(folder.as_posix(), ensure_ascii=False)
        text   = re.sub(rf'^{key} = ".*?"', lambda _: f'{key} = {value}', text, flags=re.MULTILINE)
    root = json.dumps(ROOT.as_posix(), ensure_ascii=False)
    text = re.sub(r'^root = "\."', lambda _: f'root = {root}', text, flags=re.MULTILINE)
    path = tmp_path / "app.toml"
    path.write_text(text, encoding="utf-8")
    return path


#==========================================================================
# Project config
#==========================================================================
def project_config(tmp_path: Path) -> Config:
    """
    Load the project's config on the fake backend with temporary runtime folders.

    Input:
        . tmp_path      : pytest's temporary folder
    Output:
        . config        : the whole configuration
    """
    return with_backend(load_config(project_toml(tmp_path), root=ROOT), "fake")


#==========================================================================
# A desk with an indexed fake collection
#==========================================================================
def indexed_desk(tmp_path: Path, *, rules: FakeRules | None = None, checkpointer: Any = None) -> Any:
    """
    Index the knowledge base with hashing vectors and return a Desk that uses
    the given fake rules (None = the project's rules).

    Input:
        . tmp_path      : pytest's temporary folder
        . rules         : fake-model rules for this desk
        . checkpointer  : a LangGraph checkpointer, or None for in memory
    Output:
        . desk          : a Desk on the fake backend
    """
    from    factdesk.runtime.desk           import  Desk

    assert main(["--config", str(project_toml(tmp_path)), "--fake", "index"]) == 0
    config = project_config(tmp_path)
    return Desk(config, chat=ChatSource(config.model, config.servers, rules=rules), checkpointer=checkpointer)
