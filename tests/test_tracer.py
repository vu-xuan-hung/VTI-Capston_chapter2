"""LocalTracer: an agent run becomes a saved run tree that prints as a tree."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    pathlib                         import  Path

from    langchain.agents                import  create_agent
from    langchain.tools                 import  tool
from    langchain_core.messages         import  HumanMessage

from    factdesk.config                      import  TracingConfig
from    factdesk.llm.fake_chat_model         import  FakeChatModel
from    factdesk.observability.local_tracer  import  LocalTracer
from    factdesk.observability.trace_config  import  run_config
from    factdesk.observability.trace_view    import  load_trace, save_trace, trace_path, tree_lines


#==========================================================================
# Test tool
#==========================================================================
@tool
def clock(city: str) -> str:
    """
    Tell the local time of a city.

    Input:
        . city          : a city name
    Output:
        . time          : one sentence
    """
    return f"It is 09:00 in {city}."


#==========================================================================
# Run config parts
#==========================================================================
def test_run_config_has_callbacks_tags_metadata_and_name() -> None:
    """
    Pin the shape of run_config: tracer, project tags + extra tags, metadata.
    """
    tracer = LocalTracer()
    config = run_config(
        TracingConfig(tags=("factdesk",)),
        tracer      = tracer,
        run_name    = "ask",
        metadata    = {"version": "v1", "case_id": "E01"},
        extra_tags  = ("dev",),
    )
    assert config == {
        "tags": ["factdesk", "dev"],
        "metadata": {"version": "v1", "case_id": "E01"},
        "callbacks": [tracer],
        "run_name": "ask",
    }


#==========================================================================
# An agent run is recorded as a tree
#==========================================================================
def test_agent_run_is_saved_and_printed_as_a_tree(tmp_path: Path) -> None:
    """
    Pin that model and tool runs are recorded under one root, saved to JSON,
    and printed back from the file.

    Input:
        . tmp_path      : temporary runtime folder
    """
    tracer = LocalTracer()
    agent  = create_agent(FakeChatModel(), tools=[clock])
    config = run_config(TracingConfig(), tracer=tracer, run_name="agent", metadata={"case_id": "E01"})
    agent.invoke({"messages": [HumanMessage("Time in Hanoi?")]}, config=config)
    assert len(tracer.find(kind="llm")) == 2
    assert [run.name for run in tracer.find(kind="tool")] == ["clock"]
    assert [root.name for root in tracer.roots()] == ["agent"]
    assert tracer.roots()[0].metadata["case_id"] == "E01"
    path  = save_trace(tracer, trace_path(tmp_path / "runtime", "agent"))
    lines = tree_lines(load_trace(path))
    assert path.parent == tmp_path / "runtime" / "traces"
    assert lines[0].startswith("chain     agent")
    assert any(line.strip().startswith("tool      clock") for line in lines)
    assert any(" tok " in line for line in lines if line.strip().startswith("llm"))
