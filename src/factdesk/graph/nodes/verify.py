"""Node verify (CODE) and the template answer (CODE): the numbers in the final text always come from the table."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    typing                          import  Any

from    factdesk.compute.formatting     import  template_answer
from    factdesk.compute.table          import  FactsTable
from    factdesk.config                 import  FormatConfig
from    factdesk.graph.state            import  FactState
from    factdesk.verify.numbers         import  find_unverified


#==========================================================================
# Verify
#==========================================================================
def verify_node(state: FactState, *, numbers: FormatConfig) -> dict[str, Any]:
    """
    Check that every number in the answer is a fact, the result, a year of the
    question or a counting word. Sets verified, and on success the final status.

    Input:
        . state         : answer, facts, result, years, tries
        . numbers       : the verification rules
    Output:
        . update        : verified, bad_numbers, and when verified: status "answered" and reason
    """
    bad = find_unverified(
        state["answer"], facts=state["facts"], result=state["result"], years=state["years"], numbers=numbers
    )
    update: dict[str, Any] = {"verified": not bad, "bad_numbers": bad, "log": [f"verify: bad numbers {bad}"]}
    if not bad:
        update["status"] = "answered"
        update["reason"] = "model answer verified" + (" after a retry" if state["tries"] > 1 else "")
    return update


#==========================================================================
# Template answer
#==========================================================================
def template_node(state: FactState, *, table: FactsTable, numbers: FormatConfig) -> dict[str, Any]:
    """
    Write the answer by code after the model failed to copy the numbers.

    Input:
        . state         : facts, result, tries
        . table         : names and titles
        . numbers       : number formatting
    Output:
        . update        : answer, verified True, status "answered", reason "template: ..."
    """
    text = template_answer(state["facts"], state["result"], names=table.names, titles=table.titles, numbers=numbers)
    return {
        "answer": text,
        "verified": True,
        "status": "answered",
        "reason": f"template: the model's numbers did not verify after {state['tries']} tries",
        "log": ["template: answer written by code"],
    }
