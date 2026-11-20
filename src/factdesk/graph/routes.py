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
    status = state.get("status")
    if status == "answered":
        return "explain"
    return "decline"


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
    if state.get("verified"):
        return END
    if state.get("tries", 0) < state.get("max_tries", 0):
        return "explain"
    return "template"
