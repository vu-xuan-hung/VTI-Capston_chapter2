"""The two apps under test as targets: inputs dict -> contract dict. Each case leaves a local trace."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    collections.abc                 import  Callable
from    typing                          import  Any

from    factdesk.observability.local_tracer  import  LocalTracer
from    factdesk.observability.trace_config  import  case_config
from    factdesk.observability.trace_view    import  save_trace, trace_path
from    factdesk.runtime.desk                import  Desk


#==========================================================================
# Make a target
#==========================================================================
def make_target(version: str, desk: Desk, *, save_traces: bool = True) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Build the function an experiment calls for each case. v1 asks the free
    agent; v2 asks the graph and, when it stops to ask which country is meant,
    answers from the case's "choice" (the scripted user).

    Input:
        . version       : "v1" or "v2"
        . desk          : the wired desk
        . save_traces   : write runtime/traces/<time>-<version>-<case id>.json per case
    Output:
        . target        : inputs {"question", "case_id", "choice"?} -> the contract dict
    """
    if version not in ("v1", "v2"):
        raise ValueError(f"version must be v1 or v2, got {version!r}")
    config = desk.config

    #----------------------------------------------------------------------
    # One case
    #----------------------------------------------------------------------
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Answer one case with the chosen version and save its trace.

        Input:
            . inputs        : question, case_id, choice (optional scripted answer to a clarification)
        Output:
            . contract      : the answer as a dict
        """
        #------------------------------------------------------------------
        # Label the run
        #------------------------------------------------------------------
        tracer = LocalTracer()
        run    = case_config(
            config.tracing,
            tracer      = tracer,
            version     = version,
            case_id     = inputs["case_id"],
            backend     = config.model.backend,
        )
        #------------------------------------------------------------------
        # Ask the chosen version
        #------------------------------------------------------------------
        thread = f"{version}-{inputs['case_id']}"
        if version == "v1":
            answer = desk.ask_agent(inputs["question"], run_config=run)
        else:
            answer = desk.ask(inputs["question"], thread_id=thread, run_config=run)
            if answer.status == "needs_clarification" and inputs.get("choice"):
                answer = desk.resume(thread_id=thread, choice=inputs["choice"], run_config=run)
        #------------------------------------------------------------------
        # Keep the trace
        #------------------------------------------------------------------
        if save_traces:
            save_trace(tracer, trace_path(config.paths.runtime_dir, f"{version}-{inputs['case_id']}"))
        return answer.to_contract()

    return target
