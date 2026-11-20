"""Node explain: the MODEL writes 1-3 sentences from the facts table; it may write them again with a list of its wrong numbers."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    typing                          import  Any

from    langchain_core.output_parsers   import  StrOutputParser
from    langchain_core.prompts          import  ChatPromptTemplate

from    factdesk.compute.formatting     import  facts_lines
from    factdesk.compute.table          import  FactsTable
from    factdesk.config                 import  FormatConfig
from    factdesk.graph.state            import  FactState
from    factdesk.llm.chat_source        import  ChatSource
from    factdesk.llm.prompts            import  EXPLAIN_HUMAN, EXPLAIN_SYSTEM, RETRY_HUMAN


#==========================================================================
# Explain
#==========================================================================
def explain_node(
    state: FactState,
    *,
    chat: ChatSource,
    table: FactsTable,
    numbers: FormatConfig,
) -> dict[str, Any]:
    """
    Write the answer from the facts table. With state["bad_numbers"] set (a
    failed verify) the prompt names the wrong numbers and shows the last answer.

    Input:
        . state         : question, facts, result, tries, and after a failed verify: answer, bad_numbers
        . chat          : the lazy chat model
        . table         : names and titles for the facts table
        . numbers       : number formatting
    Output:
        . update        : answer, tries (one more), log
    """
    lines = "\n".join(
        facts_lines(state["facts"], state["result"], names=table.names, titles=table.titles, numbers=numbers)
    )
    if state.get("bad_numbers"):
        template = ChatPromptTemplate.from_messages([("system", EXPLAIN_SYSTEM), ("human", RETRY_HUMAN)])
        values   = {
            "question": state["question"], "facts": lines,
            "previous": state["answer"], "bad": ", ".join(state["bad_numbers"]),
        }
    else:
        template = ChatPromptTemplate.from_messages([("system", EXPLAIN_SYSTEM), ("human", EXPLAIN_HUMAN)])
        values   = {"question": state["question"], "facts": lines}
    answer = (template | chat.get() | StrOutputParser()).invoke(values).strip()
    tries  = state.get("tries", 0) + 1
    return {"answer": answer, "tries": tries, "log": [f"explain: try {tries}"]}
