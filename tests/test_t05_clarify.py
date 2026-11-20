"""T05: the clarify node: interrupt with the candidates, apply the choice (run: pytest -k t05)."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    functools                       import  partial
from    pathlib                         import  Path
from    typing                          import  Any

import  pytest
from    langgraph.checkpoint.memory     import  InMemorySaver
from    langgraph.graph                 import  END, START, StateGraph
from    langgraph.types                 import  Command

from    factdesk.graph.nodes.clarify    import  clarify_node
from    factdesk.graph.state            import  FactState


#==========================================================================
# A graph of just the clarify node
#==========================================================================
def one_node_graph(outbox: Path) -> Any:
    """
    Compile START -> clarify -> END with an in-memory checkpointer.

    Input:
        . outbox        : the outbox file of the node
    Output:
        . app           : the compiled graph
    """
    graph = StateGraph(FactState)
    graph.add_node("clarify", partial(clarify_node, outbox=outbox))
    graph.add_edge(START, "clarify")
    graph.add_edge("clarify", END)
    return graph.compile(checkpointer=InMemorySaver())


START_STATE = {
    "question": "What was the population of Korea in 2015?",
    "thread_id": "t1",
    "resolved": {"Congo": "COD"},
    "pending": [{"text": "Korea", "candidates": [
        {"code": "KOR", "name": "Korea, Rep.", "score": 1.0}, {"code": "PRK", "name": "Korea, Dem. People's Rep.", "score": 1.0},
    ]}],
}


#==========================================================================
# Interrupt, resume, apply
#==========================================================================
def test_t05_clarify_interrupts_with_the_candidates_and_applies_the_choice(tmp_path: Path) -> None:
    """
    Pin T05: the graph stops with the candidates in the interrupt value; the resume value
    (a code) is stored in "resolved" next to the earlier choices and "pending" is cleared.

    Input:
        . tmp_path      : temporary folder
    """
    app    = one_node_graph(tmp_path / "outbox.jsonl")
    config = {"configurable": {"thread_id": "t1"}}
    out    = app.invoke(START_STATE, config)
    ask    = out["__interrupt__"][0].value
    assert ask["text"] == "Korea" and [item["code"] for item in ask["options"]] == ["KOR", "PRK"]
    done   = app.invoke(Command(resume="KOR"), config)
    assert done["resolved"] == {"Congo": "COD", "Korea": "KOR"} and done["pending"] == []


#==========================================================================
# A choice outside the candidates is refused
#==========================================================================
def test_t05_a_choice_outside_the_candidates_is_refused(tmp_path: Path) -> None:
    """
    Pin T05: a code that is not one of the options raises ValueError naming the options.

    Input:
        . tmp_path      : temporary folder
    """
    app    = one_node_graph(tmp_path / "outbox.jsonl")
    config = {"configurable": {"thread_id": "t1"}}
    app.invoke(START_STATE, config)
    with pytest.raises(ValueError, match="not one of"):
        app.invoke(Command(resume="JPN"), config)


#==========================================================================
# The effect before the interrupt happens once
#==========================================================================
def test_t05_the_outbox_line_is_written_once_although_the_node_runs_twice(tmp_path: Path) -> None:
    """
    Pin T05: after resume the node runs again from its first line; the outside effect
    (one outbox line per thread and name) must not be repeated.

    Input:
        . tmp_path      : temporary folder
    """
    outbox = tmp_path / "outbox.jsonl"
    app    = one_node_graph(outbox)
    config = {"configurable": {"thread_id": "t1"}}
    app.invoke(START_STATE, config)
    assert len(outbox.read_text(encoding="utf-8").splitlines()) == 1
    app.invoke(Command(resume="PRK"), config)
    lines = outbox.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and '"key": "t1:Korea"' in lines[0]
