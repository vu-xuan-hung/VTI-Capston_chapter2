"""Node compute: CODE reads the facts table. No model is involved."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    typing                          import  Any

from    factdesk.compute.operations     import  compute
from    factdesk.compute.table          import  FactsTable
from    factdesk.config                 import  FormatConfig
from    factdesk.graph.state            import  FactState


#==========================================================================
# Compute
#==========================================================================
def compute_node(state: FactState, *, table: FactsTable, numbers: FormatConfig) -> dict[str, Any]:
    """
    Run the operation of the question on the linked countries and years.

    Input:
        . state         : operation, indicator, countries, years
        . table         : the facts table
        . numbers       : rounding of results
    Output:
        . update        : status (answered, no_data, out_of_scope), reason, facts, result, missing, log
    """
    #----------------------------------------------------------------------
    # Run the operation
    #----------------------------------------------------------------------
    computed = compute(
        table,
        operation   = state["operation"],
        indicator   = state["indicator"],
        countries   = state["countries"],
        years       = state["years"],
        numbers     = numbers,
    )
    #----------------------------------------------------------------------
    # Hand the outcome to the state
    #----------------------------------------------------------------------
    return {
        "status": computed.status,
        "reason": computed.reason,
        "facts": computed.facts,
        "result": computed.result,
        "missing": computed.missing,
        "log": [f"compute: {computed.status} {computed.reason}".rstrip()],
    }
