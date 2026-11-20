"""The state that flows through the FactDesk graph."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  operator
from    typing                          import  Annotated, Any, TypedDict


#==========================================================================
# State
#==========================================================================
class FactState(TypedDict, total=False):
    """
    Purpose: everything the nodes pass along. Only `log` has a reducer (its
    lists are added); every other key is replaced by the node that writes it.
    """

    question: str
    thread_id: str
    max_tries: int
    query: dict[str, Any]               # the parsed question (Query as a dict)
    operation: str
    indicator: str | None
    years: list[int]
    links: list[dict[str, Any]]         # one entry per country text: status, code, method, candidates
    resolved: dict[str, str]            # country text -> code, chosen by the user in clarify
    pending: list[dict[str, Any]]       # ambiguous country texts with their candidates
    unknown: list[str]                  # country texts nothing matched
    countries: list[str]
    facts: list[dict[str, Any]]
    result: dict[str, Any] | None
    missing: dict[str, Any] | None
    answer: str
    bad_numbers: list[str]
    verified: bool
    tries: int
    status: str
    reason: str
    log: Annotated[list[str], operator.add]
