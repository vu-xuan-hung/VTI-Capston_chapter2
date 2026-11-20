"""Build the run config that connects a call to a tracer, with labels."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                          import  annotations

from    typing                              import  Any

from    langchain_core.callbacks            import  BaseCallbackHandler
from    langchain_core.runnables            import  RunnableConfig

from    factdesk.config                     import  TracingConfig


#==========================================================================
# Run config
#==========================================================================
def run_config(
    tracing: TracingConfig,
    *,
    tracer: BaseCallbackHandler | None = None,
    run_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    extra_tags: tuple[str, ...] = (),
) -> RunnableConfig:
    """
    Pass the tracer as a callback; tags and metadata make the run findable
    later, locally and on LangSmith (where they become filters).

    Input:
        . tracing       : the project's tags
        . tracer        : a LocalTracer, or None (LangSmith alone needs none)
        . run_name      : name of the root run, or None
        . metadata      : e.g. {"version": "v2", "case_id": "E03"}
        . extra_tags    : tags for this call only
    Output:
        . config        : the config argument of invoke / stream
    """
    config: RunnableConfig = {
        "tags": [*tracing.tags, *extra_tags],
        "metadata": dict(metadata or {}),
    }
    if tracer is not None:
        config["callbacks"] = [tracer]
    if run_name:
        config["run_name"] = run_name
    return config


#==========================================================================
# The labels of one evaluation case
#==========================================================================
def case_config(
    tracing: TracingConfig,
    *,
    tracer: BaseCallbackHandler | None,
    version: str,
    case_id: str,
    backend: str,
) -> RunnableConfig:
    """
    The run config of one case in an experiment: the root run is named
    "<version>-<case id>", it carries the project tags plus the version as a
    tag, and the metadata version, case_id and backend. On LangSmith these are
    the filters that find "all v1 runs" or "case D06 of both versions".

    Input:
        . tracing       : the project's tags
        . tracer        : a LocalTracer, or None
        . version       : "v1" (agent) or "v2" (graph)
        . case_id       : the case id, e.g. "D06"
        . backend       : "local", "server" or "fake"
    Output:
        . config        : the config argument of invoke
    """
    return run_config(
        tracing,
        tracer=tracer,
        run_name=f"{version}-{case_id}",
        metadata={"version": version, "case_id": case_id, "backend": backend},
        extra_tags=(version,),
    )
    # TODO T07 -- return run_config(tracing, tracer=tracer, run_name=f"{version}-{case_id}",
    # metadata={"version", "case_id", "backend"}, extra_tags=(version,)). Then run both versions and fill my_work/notes.toml.
    # CHECK: pytest -k t07 -- run name, tags, metadata and the tracer callback; the notes are checked by python main.py submit-check.


# raise NotImplementedError("T07: case_config")
