"""v1: a free tool-calling agent. The model decides which tool, with which arguments, and writes the answer.

No link step, no verify step, no routing: whatever the model does is the result.
The contract is built afterwards from the tool results the agent happened to get.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
from    typing                          import  Any

from    langchain.agents                            import  create_agent
from    langchain_core.language_models.chat_models  import  BaseChatModel
from    langchain_core.messages                     import  AIMessage, BaseMessage, HumanMessage, ToolMessage
from    langchain_core.runnables                    import  RunnableConfig
from    langgraph.errors                            import  GraphRecursionError

from    factdesk.agents.tools           import  make_tools
from    factdesk.compute.table          import  FactsTable
from    factdesk.config                 import  FormatConfig, GraphConfig
from    factdesk.contract               import  Answer, Fact, Result
from    factdesk.llm.prompts            import  AGENT_SYSTEM


#==========================================================================
# Run the agent
#==========================================================================
def run_agent(
    question: str,
    *,
    chat: BaseChatModel,
    table: FactsTable,
    numbers: FormatConfig,
    graph: GraphConfig,
    run_config: RunnableConfig | None = None,
) -> Answer:
    """
    Run create_agent with the three tools on one question and turn what
    happened into the contract. A run that does not finish in
    graph.agent_max_steps steps is reported, not hidden.

    Input:
        . question      : the user's question
        . chat          : a chat model that can bind tools
        . table         : the facts table the tools read
        . numbers       : rounding of results
        . graph         : agent_max_steps
        . run_config    : tags, metadata and the tracer callback
    Output:
        . answer        : the contract; verified is always False (nothing checked it)
    """
    agent  = create_agent(chat, tools=make_tools(table, numbers), system_prompt=AGENT_SYSTEM)
    config = {**(run_config or {}), "recursion_limit": graph.agent_max_steps}
    try:
        messages = agent.invoke({"messages": [HumanMessage(question)]}, config=config)["messages"]
    except GraphRecursionError:
        return Answer(
            question    = question,
            status      = "no_data",
            reason      = f"v1: the agent did not finish in {graph.agent_max_steps} steps",
        )
    return answer_from_agent(question, messages, table=table)


#==========================================================================
# The contract of an agent run
#==========================================================================
def answer_from_agent(question: str, messages: list[BaseMessage], *, table: FactsTable) -> Answer:
    """
    Build the contract from the conversation: facts and result come from the
    tool results; the entities are what the agent actually looked up. Status is
    "no_data" when the last tool call returned an error, otherwise "answered":
    the agent has no way to decline, so a run without any tool call is
    "answered" too (and its numbers are unverified).

    Input:
        . question      : the user's question
        . messages      : the agent's messages
        . table         : units of the indicators
    Output:
        . answer        : the contract
    """

    #----------------------------------------------------------------------
    # Read the tool results
    #----------------------------------------------------------------------
    results: list[dict[str, Any]] = [
        json.loads(message.content) for message in messages if isinstance(message, ToolMessage)
    ]
    good  = [item for item in results if "facts" in item]
    final = next((message.text for message in reversed(messages) if isinstance(message, AIMessage) and message.text), "")
    facts: list[dict[str, Any]] = []
    for item in good:
        facts += [fact for fact in item["facts"] if fact not in facts]

    #----------------------------------------------------------------------
    # Entities are what the tools were asked for
    #----------------------------------------------------------------------
    result = next((item["result"] for item in reversed(good) if item.get("result")), None)
    failed = bool(results) and "error" in results[-1]
    return Answer(
        question    = question,
        operation   = good[-1]["operation"] if good else None,
        indicator   = facts[0]["indicator"] if facts else None,
        countries   = list(dict.fromkeys(fact["code"] for fact in facts)),
        years       = sorted({fact["year"] for fact in facts}),
        facts       = [Fact(**fact) for fact in facts],
        result      = Result(**result) if result else None,
        unit        = table.units[facts[0]["indicator"]] if facts else None,
        answer      = final,
        verified    = False,
        status      = "no_data" if failed else "answered",
        reason      = f"v1: last tool call failed: {results[-1]['error']}" if failed else "v1: not checked by code",
    )
