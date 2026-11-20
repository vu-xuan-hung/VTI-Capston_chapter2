"""The JSON contract of every run: one object, the same keys for v1, v2 and every status."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    typing                          import  Any, Literal

from    pydantic                        import  BaseModel, Field


#==========================================================================
# One fact and one result
#==========================================================================
class Fact(BaseModel):
    """
    Purpose: one number read from the facts table.
    """

    code: str
    year: int
    indicator: str
    value: int | float


#==========================================================================
# One result
#==========================================================================
class Result(BaseModel):
    """
    Purpose: the number the question asked for: a value, a ratio, a percent change or the highest.
    """

    label: str
    value: int | float


#==========================================================================
# The answer
#==========================================================================
class Answer(BaseModel):
    """
    Purpose: what `python main.py ask` prints and what the evaluators read.
    Every number in `facts` and `result` came from the table through code.
    """

    question: str
    operation: Literal["value", "compare", "growth", "rank"] | None     = None
    indicator: str | None                                               = None
    countries: list[str]                                                = Field(default_factory=list)
    years: list[int]                                                    = Field(default_factory=list)
    facts: list[Fact]                                                   = Field(default_factory=list)
    result: Result | None                                               = None
    unit: str | None                                                    = None
    answer: str                                                         = ""
    verified: bool                                                      = False
    status: Literal["answered", "needs_clarification", "no_data", "out_of_scope"]  = "out_of_scope"
    reason: str                                                         = ""

    #----------------------------------------------------------------------
    # The plain dict that is printed and saved
    #----------------------------------------------------------------------
    def to_contract(self) -> dict[str, Any]:
        """
        Dump the answer as a JSON-ready dict with the contract's key order.

        Output:
            . contract      : question, operation, indicator, countries, years, facts, result, unit,
                              answer, verified, status, reason
        """
        return self.model_dump(mode="json")


#==========================================================================
# Build the answer from a graph state
#==========================================================================
def answer_from_state(state: dict[str, Any], *, units: dict[str, str]) -> Answer:
    """
    Turn the final state of the graph into the contract. The operation is the
    model's reading of the question, kept only when it is one of the four.

    Input:
        . state         : the graph's final state
        . units         : indicator -> unit, from the facts table
    Output:
        . answer        : the contract object
    """
    #----------------------------------------------------------------------
    # Read the state
    #----------------------------------------------------------------------
    operation = state.get("operation")
    indicator = state.get("indicator")
    result    = state.get("result")
    #----------------------------------------------------------------------
    # Write the contract
    #----------------------------------------------------------------------
    return Answer(
        question    = state["question"],
        operation   = operation if operation in ("value", "compare", "growth", "rank") else None,
        indicator   = indicator,
        countries   = state.get("countries", []),
        years       = state.get("years", []),
        facts       = [Fact(**fact) for fact in state.get("facts", [])],
        result      = Result(label=result["label"], value=result["value"]) if result else None,
        unit        = units.get(indicator) if indicator else None,
        answer      = state.get("answer", ""),
        verified    = state.get("verified", False),
        status      = state["status"],
        reason      = state.get("reason", ""),
    )
