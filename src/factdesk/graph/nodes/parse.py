"""Node parse: the MODEL reads the question into a Query."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    typing                          import  Any

from    langchain_core.exceptions       import  OutputParserException
from    pydantic                        import  ValidationError

from    factdesk.graph.state            import  FactState
from    factdesk.llm.chat_source        import  ChatSource
from    factdesk.llm.query              import  build_parse_chain


#==========================================================================
# Parse
#==========================================================================
def parse_node(state: FactState, *, chat: ChatSource) -> dict[str, Any]:
    """
    Run the parse chain. When the model's output is not a valid Query (an invalid
    value, or no tool call at all, which a small model sometimes does) the
    run goes on as operation "other" with the reason, so it still ends in a
    clear status (out_of_scope) and never in a crash.

    Input:
        . state         : holds "question"
        . chat          : the lazy chat model
    Output:
        . update        : query, operation, years, log (and reason when the output was invalid)
    """
    try:
        query = build_parse_chain(chat.get()).invoke({"question": state["question"]})
        failure = None if query is not None else "the model made no tool call"
    except (ValidationError, OutputParserException) as error:
        failure = type(error).__name__
    if failure:
        reason = f"the model's reading of the question was not a valid Query: {failure}"
        return {"query": {}, "operation": "other", "years": [], "reason": reason, "log": [f"parse: {reason}"]}
    return {
        "query": query.model_dump(),
        "operation": query.operation,
        "years": [int(year) for year in query.years],
        "log": [f"parse: {query.operation} | {query.indicator_text!r} | {query.country_texts} | {query.years}"],
    }
