"""Compare what the code chain and the Langflow flow answered to the same question."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                          import  annotations

from    dataclasses                         import  dataclass
from    typing                              import  Any

from    factdesk.config                     import  FormatConfig
from    factdesk.verify.numbers             import  find_unverified


#==========================================================================
# The comparison of two answers
#==========================================================================
@dataclass(frozen=True)
class AnswerComparison:
    """
    Purpose: what each answer is missing and which of its numbers are not in
    the facts. Empty lists mean "nothing wrong found by the rules".
    """

    code_missing_terms: list[str]
    flow_missing_terms: list[str]
    code_unverified: list[str]
    flow_unverified: list[str]


#==========================================================================
# Compare two answers
#==========================================================================
def compare_answers(
    code_answer: str,
    flow_answer: str,
    *,
    expected_terms: list[str],
    facts: list[dict[str, Any]] | None,
    result: dict[str, Any] | None,
    years: list[int],
    numbers: FormatConfig,
) -> AnswerComparison:
    """
    Compare the code's answer and the flow's answer by rules, not by reading.
    expected_terms: words a good answer contains (matched ignoring case).
    facts: for a NUMERIC question, the facts the code computed - each answer's
    numbers are then checked against them; None means "a definition question,
    do not check numbers".

    Input:
        . code_answer   : the answer of the code (graph or RAG chain)
        . flow_answer   : the answer of the Langflow flow
        . expected_terms: terms a correct answer contains
        . facts         : the code's facts for a numeric question, else None
        . result        : the code's result for a numeric question, else None
        . years         : the years of the question
        . numbers       : the verification rules
    Output:
        . comparison    : missing terms and unverified numbers of both answers
    """
    #----------------------------------------------------------------------
    # Check terms and numbers against the same evidence
    #----------------------------------------------------------------------
    return AnswerComparison(
        code_missing_terms=[
            term for term in expected_terms if term.lower() not in code_answer.lower()
        ],
        flow_missing_terms=[
            term for term in expected_terms if term.lower() not in flow_answer.lower()
        ],
        code_unverified=[] if facts is None else find_unverified(
            answer=code_answer,
            facts=facts,
            result=result,
            years=years,
            numbers=numbers,
        ),
        flow_unverified=[] if facts is None else find_unverified(
            answer=flow_answer,
            facts=facts,
            result=result,
            years=years,
            numbers=numbers,
        ),
    )
