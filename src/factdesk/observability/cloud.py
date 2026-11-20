"""Switch LangSmith on with environment variables; a key never enters a file.

Without a key, everything still works locally (LocalTracer, run_experiment);
only the cloud path asks for one, in one sentence.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  os
from    pathlib                         import  Path


#==========================================================================
# Read .env
#==========================================================================
def load_env_file(path: Path) -> None:
    """
    Read KEY=VALUE lines from a .env file; a value already in the
    environment wins, so a shell export overrides the file.

    Input:
        . path          : the .env file; a missing file is fine
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


#==========================================================================
# Is a LangSmith key available?
#==========================================================================
def cloud_available() -> bool:
    """
    Tell whether a LangSmith key is in the environment.

    Output:
        . available     : True when LANGSMITH_API_KEY is set and not empty
    """
    return bool(os.environ.get("LANGSMITH_API_KEY"))


#==========================================================================
# Stop with one sentence when there is no key
#==========================================================================
def require_cloud() -> None:
    """
    Raise one clear error before any LangSmith call when no key is set.
    """
    if not cloud_available():
        raise ValueError(
            "LANGSMITH_API_KEY is not set: create a key at smith.langchain.com "
            "(Settings -> API Keys) and put it in .env next to main.py; "
            "the local tracer and the local experiment need no key."
        )


#==========================================================================
# Send every LangChain / LangGraph run of this process to LangSmith
#==========================================================================
def enable_tracing(project: str) -> str:
    """
    Turn LangSmith tracing on for this process. Agent-based workspace:
    LANGSMITH_AGENT_ID is set and a project must NOT be (it is removed).
    Project-based workspace (older, self-hosted): one project name.

    Input:
        . project       : project name for project-based workspaces
    Output:
        . target        : where the traces go, for one log line
    """
    require_cloud()
    os.environ["LANGSMITH_TRACING"] = "true"
    agent = os.environ.get("LANGSMITH_AGENT_ID")
    if agent:
        os.environ.pop("LANGSMITH_PROJECT", None)
        os.environ.setdefault("LANGSMITH_AGENT_ENVIRONMENT", "local")
        return f"agent {agent}, environment {os.environ['LANGSMITH_AGENT_ENVIRONMENT']}"
    os.environ["LANGSMITH_PROJECT"] = project
    return f"project {project}"
