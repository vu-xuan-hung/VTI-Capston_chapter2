"""T07: the labels of one evaluation case (run: pytest -k t07). The notes in my_work/notes.toml are checked by submit-check."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    factdesk.config                      import  TracingConfig
from    factdesk.observability.local_tracer  import  LocalTracer
from    factdesk.observability.trace_config  import  case_config


#==========================================================================
# Tags and metadata
#==========================================================================
def test_t07_case_config_names_the_run_and_labels_it_with_version_and_case() -> None:
    """
    Pin T07: run name "<version>-<case id>", the project tags plus the version as a tag, the
    metadata version, case_id and backend, and the tracer as a callback.
    """
    tracer = LocalTracer()
    config = case_config(TracingConfig(project="p", tags=("factdesk",)), tracer=tracer, version="v2", case_id="D06", backend="local")
    assert config["run_name"] == "v2-D06"
    assert config["tags"] == ["factdesk", "v2"]
    assert config["metadata"] == {"version": "v2", "case_id": "D06", "backend": "local"}
    assert config["callbacks"] == [tracer]


#==========================================================================
# Without a tracer there is no callback
#==========================================================================
def test_t07_without_a_tracer_there_is_no_callback() -> None:
    """
    Pin T07: tracer=None (LangSmith alone) adds no callbacks key.
    """
    config = case_config(TracingConfig(), tracer=None, version="v1", case_id="D01", backend="fake")
    assert "callbacks" not in config and config["tags"][-1] == "v1"
