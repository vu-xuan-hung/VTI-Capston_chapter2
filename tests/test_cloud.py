"""LangSmith switch: no key = one clear sentence; agent vs project workspace."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  os
from    pathlib                         import  Path

import  pytest

from    factdesk.evaluation.cloud_experiment  import  run_cloud_experiment
from    factdesk.evaluation.dataset           import  Example
from    factdesk.observability.cloud          import  cloud_available, enable_tracing, load_env_file


#==========================================================================
# Clean LangSmith environment
#==========================================================================
@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """
    Remove every LangSmith variable; monkeypatch restores them afterwards,
    so LANGSMITH_TRACING never leaks into other tests.

    Input:
        . monkeypatch   : pytest's environment patcher
    Output:
        . monkeypatch   : the same patcher, for more changes
    """
    for name in ("LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT",
                 "LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return monkeypatch


#==========================================================================
# No key: the cloud experiment stops at once
#==========================================================================
def test_cloud_experiment_without_key_is_one_sentence(clean_env: pytest.MonkeyPatch) -> None:
    """
    Pin that the cloud path imports and fails before any network call.

    Input:
        . clean_env     : no LangSmith variable set
    """
    assert not cloud_available()
    with pytest.raises(ValueError, match="^LANGSMITH_API_KEY is not set: create a key"):
        run_cloud_experiment("v1", dict, [Example("E1", {}, {})], [], dataset_name="factdesk-dataset")
    with pytest.raises(ValueError, match="LANGSMITH_API_KEY is not set"):
        enable_tracing("factdesk")
    assert "LANGSMITH_TRACING" not in os.environ


#==========================================================================
# Agent-based and project-based workspaces
#==========================================================================
def test_enable_tracing_handles_both_workspace_kinds(clean_env: pytest.MonkeyPatch) -> None:
    """
    Pin both workspace kinds: with LANGSMITH_AGENT_ID the project variable is removed; without
    it the project name is set.

    Input:
        . clean_env     : no LangSmith variable set
    """
    clean_env.setenv("LANGSMITH_API_KEY", "test-value")
    clean_env.setenv("LANGSMITH_PROJECT", "old")
    clean_env.setenv("LANGSMITH_AGENT_ID", "agent-1")
    assert enable_tracing("factdesk") == "agent agent-1, environment local"
    assert "LANGSMITH_PROJECT" not in os.environ
    clean_env.delenv("LANGSMITH_AGENT_ID")
    assert enable_tracing("factdesk") == "project factdesk"
    assert os.environ["LANGSMITH_PROJECT"] == "factdesk"
    assert os.environ["LANGSMITH_TRACING"] == "true"


#==========================================================================
# .env reading
#==========================================================================
def test_env_file_never_overrides_the_shell(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Pin that .env fills missing variables only and skips empty values.

    Input:
        . clean_env     : no LangSmith variable set
        . tmp_path      : folder of the .env file
    """
    env = tmp_path / ".env"
    env.write_text("# comment\nLANGSMITH_PROJECT=from-file\nLANGSMITH_API_KEY=\n", encoding="utf-8")
    clean_env.setenv("LANGSMITH_AGENT_ENVIRONMENT", "shell")
    load_env_file(env)
    load_env_file(tmp_path / "missing.env")
    assert os.environ["LANGSMITH_PROJECT"] == "from-file"
    assert "LANGSMITH_API_KEY" not in os.environ
