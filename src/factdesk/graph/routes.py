"""The conditional edges of the graph: small pure functions of the state.

A router reads the state and returns the NAME of the next node. No model
call, no side effect: easy to test with a plain dict.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                          import  annotations

from    langgraph.graph                     import  END

from    factdesk.graph.state                import  FactState


#==========================================================================
# After link
#==========================================================================
def after_link(state: FactState) -> str:
    """
    Decide what follows linking: a declined question (state["status"] is set by
    link) goes to "decline", a state with ambiguous names in state["pending"]
    goes to "clarify", anything else goes to "compute".

    Input:
        . state         : after the link node
    Output:
        . node          : "decline", "clarify" or "compute"
    """
    if state.get("status"):
        return "decline"

    if state.get("pending"):
        return "clarify"
    return "compute"
    # TODO T04 -- return "decline" when state["status"]
    # is set (link declined), "clarify" when state["pending"] is not empty,
    # else "compute".
    # CHECK: pytest -k t04 -- after_link returns the right node for the four states of the test.
    # raise NotImplementedError("T04: after_link")


#==========================================================================
# After compute
#==========================================================================
def after_compute(state: FactState) -> str:
    """
    Decide what follows compute: status "answered" goes to "explain", any other
    status (no_data, out_of_scope) goes to "decline".

    Input:
        . state         : after the compute node
    Output:
        . node          : "explain" or "decline"
    """
    # TODO T04 -- return "explain" when
    # state["status"] == "answered", else "decline".
    # CHECK: pytest -k t04 -- after_compute declines no_data and out_of_scope.
    status = state.get("status")
    if status == "answered":
        return "explain"
    return "decline"

    # raise NotImplementedError("T04: after_compute")


#==========================================================================
# After verify
#==========================================================================
def after_verify(state: FactState) -> str:
    """
    Decide what follows verify: a verified answer ends the graph (END); a wrong
    one is written again while state["tries"] < state["max_tries"] ("explain");
    after the last try the code writes the answer ("template").

    Input:
        . state         : after the verify node; tries counts the explain calls so far
    Output:
        . node          : END, "explain" or "template"
    """
    # TODO T04 -- return END when state["verified"]; else "explain" while state["tries"] < state["max_tries"]; else "template".
    # CHECK: pytest -k t04 -- tries 1 of 2 retries, tries 2 of 2 goes to the template answer.
    if state.get("verified"):
        return END
    if state.get("tries", 0) < state.get("max_tries", 0):
        return "explain"
    return "template"
    # raise NotImplementedError("T04: after_verify")
