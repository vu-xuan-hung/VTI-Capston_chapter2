"""Open Chroma embedded (a folder) or as a client of `chroma run`.

Both read the same on-disk format: `chroma run --path kb_store` serves the
folder the embedded mode writes. Never open the folder embedded while the
server runs on it; use knowledge.store = "server" then.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    pathlib                         import  Path
from    typing                          import  Any

from    factdesk.config                 import  KnowledgeConfig, ServerConfig


#==========================================================================
# Open the configured Chroma client
#==========================================================================
def open_chroma_client(
    knowledge: KnowledgeConfig,
    servers: ServerConfig,
    *,
    kb_dir: Path,
) -> Any:
    """
    Create the Chroma client of the configured store kind; telemetry off.

    Input:
        . knowledge     : store kind, "embedded" or "server"
        . servers       : Chroma server host and port
        . kb_dir        : the embedded folder
    Output:
        . client        : chromadb PersistentClient or HttpClient
    """
    import  chromadb
    from    chromadb.config                 import  Settings

    settings = Settings(anonymized_telemetry=False)
    if knowledge.store == "server":
        return chromadb.HttpClient(
            host        = servers.chroma_host,
            port        = servers.chroma_port,
            settings    = settings,
        )
    return chromadb.PersistentClient(path=str(kb_dir), settings=settings)


#==========================================================================
# The command that starts the Chroma server
#==========================================================================
def chroma_run_command(servers: ServerConfig, *, kb_dir: Path) -> str:
    """
    Write the `chroma run` line for this project, for messages and docs.

    Input:
        . servers       : Chroma server host and port
        . kb_dir        : the folder the server serves
    Output:
        . command       : one shell line
    """
    return f"chroma run --host {servers.chroma_host} --port {servers.chroma_port} --path {kb_dir.name}"
