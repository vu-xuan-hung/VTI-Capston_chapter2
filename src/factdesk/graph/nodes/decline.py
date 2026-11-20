"""Node decline: an honest hand-over. A fixed sentence from code; the reason says why."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    typing                          import  Any

from    factdesk.compute.table          import  FactsTable
from    factdesk.graph.state            import  FactState


#==========================================================================
# Decline
#==========================================================================
def decline_node(state: FactState, *, table: FactsTable) -> dict[str, Any]:
    """
    Write the hand-over sentence for status no_data or out_of_scope; the reason
    of an out_of_scope run is part of the sentence. It holds no data number, so
    there is nothing to verify (verified is True).

    Input:
        . state         : status, reason, missing (a no_data cell) or unknown country texts
        . table         : names and titles
    Output:
        . update        : answer, verified, log
    """
    if state["status"] == "no_data" and state.get("missing"):
        cell   = state["missing"]
        answer = (
            f"The table has no {table.titles[cell['indicator']]} for {table.names[cell['code']]} "
            f"in {cell['year']}, so I cannot answer."
        )
    elif state["status"] == "no_data":
        answer = f"I could not find {', '.join(state['unknown'])} in the table, so I cannot answer."
    else:
        answer = (
            f"I cannot answer this question: {state['reason']}. I answer questions that ask for the "
            "population or GDP of countries: a value, a comparison, a growth or a ranking."
        )
    return {"answer": answer, "verified": True, "log": [f"decline: {state['status']}"]}
