"""Say which local servers answer, and how to start the others."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    dataclasses                     import  dataclass
from    pathlib                         import  Path

from    factdesk.config                      import  ServerConfig
from    factdesk.integrations.chroma_server  import  chroma_run_command
from    factdesk.integrations.http_json      import  request_json


#==========================================================================
# One server's state
#==========================================================================
@dataclass(frozen=True)
class ServerStatus:
    """
    Purpose: one server: its name, the address tried, whether it answered and
    the command that starts it.
    """

    name: str
    url: str
    alive: bool
    start: str


#==========================================================================
# Try each server
#==========================================================================
def check_servers(servers: ServerConfig, *, kb_dir: Path) -> list[ServerStatus]:
    """
    Call the model server, Chroma and Langflow with a short timeout.

    Input:
        . servers       : the addresses
        . kb_dir        : the Chroma folder, for the start command
    Output:
        . statuses      : model server, Chroma server, Langflow, in that order
    """
    chroma = f"http://{servers.chroma_host}:{servers.chroma_port}/api/v2/heartbeat"
    tests  = [
        ("model server", f"{servers.model_url}/models", "python main.py serve-model"),
        ("Chroma server", chroma, chroma_run_command(servers, kb_dir=kb_dir)),
        (
            "Langflow",
            f"{servers.langflow_url}/health_check",
            "langflow run --env-file .env.langflow   (in its own env)",
        ),
    ]
    return [ServerStatus(name, url, _answers(url), start) for name, url, start in tests]


#==========================================================================
# Does one address answer?
#==========================================================================
def _answers(url: str) -> bool:
    """
    Treat a refusal, a timeout or an error status as "not running".

    Input:
        . url           : address to GET
    Output:
        . alive         : True when it answered with JSON
    """
    try:
        request_json(url, timeout=5)
    except (OSError, ValueError):
        return False
    return True
