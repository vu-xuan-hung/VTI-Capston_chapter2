"""FakeChatModel: structured output and create_agent loops, by injected rules."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    typing                          import  Any, Literal

import  pytest
from    langchain.agents                import  create_agent
from    langchain.tools                 import  tool
from    langchain_core.messages         import  BaseMessage, HumanMessage, ToolMessage
from    pydantic                        import  BaseModel

from    factdesk.llm.fake_chat_model    import  FakeChatModel, FakeRules


#==========================================================================
# Test schema and tool
#==========================================================================
class Verdict(BaseModel):
    """
    Purpose: a small structured answer with an enum, text and a number.
    """

    label: Literal["open", "closed"]
    reason: str
    score: int


#==========================================================================
# Test tool
#==========================================================================
@tool
def lookup(code: str) -> str:
    """
    Look up the status of one case code.

    Input:
        . code          : the case code, e.g. A1
    Output:
        . status        : one sentence
    """
    return f"case {code} is open"


#==========================================================================
# Default rules fill any schema
#==========================================================================
def test_default_rules_fill_any_schema_with_placeholders() -> None:
    """
    Pin that with_structured_output works with no project rule at all.
    """
    answer = FakeChatModel().with_structured_output(Verdict).invoke("Is case A1 open?")
    assert answer == Verdict(label="open", reason="[FAKE]", score=0)


#==========================================================================
# A project rule answers by schema and prompt
#==========================================================================
def test_structured_rule_receives_schema_name_and_prompt() -> None:
    """
    Pin the structured(schema_name, prompt) contract of FakeRules.
    """

    #----------------------------------------------------------------------
    # A rule that answers by the prompt
    #----------------------------------------------------------------------
    seen: list[str] = []

    def structured(schema_name: str, prompt: str) -> dict[str, Any]:
        """
        Answer "closed" when the prompt says so.

        Input:
            . schema_name   : the Pydantic class name
            . prompt        : every message's text
        Output:
            . args          : the Verdict fields
        """
        seen.append(schema_name)
        label = "closed" if "closed" in prompt else "open"
        return {"label": label, "reason": "rule", "score": 2}

    #----------------------------------------------------------------------
    # Ask for a Verdict
    #----------------------------------------------------------------------
    chat   = FakeChatModel(rules=FakeRules(structured=structured))
    answer = chat.with_structured_output(Verdict).invoke("Case B2 was closed yesterday.")
    assert answer.label == "closed"
    assert seen == ["Verdict"]


#==========================================================================
# create_agent tool loop with project rules
#==========================================================================
def test_agent_loop_calls_the_tool_then_answers() -> None:
    """
    Pin a create_agent loop: rule asks for the tool, then the text rule answers.
    """

    #----------------------------------------------------------------------
    # A loop rule: call lookup once
    #----------------------------------------------------------------------
    def calls(messages: list[BaseMessage], tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """
        Call lookup once, then stop.

        Input:
            . messages      : the conversation so far
            . tools         : the bound tool schemas
        Output:
            . calls         : one lookup call, or None after the tool answered
        """
        if isinstance(messages[-1], ToolMessage):
            return None
        return [{"name": "lookup", "args": {"code": "A1"}}]

    #----------------------------------------------------------------------
    # Run the agent
    #----------------------------------------------------------------------
    rules  = FakeRules(tool_calls=calls, text=lambda prompt: f"final: {prompt.splitlines()[-1]}")
    agent  = create_agent(FakeChatModel(rules=rules), tools=[lookup], system_prompt="Use the tools.")
    result = agent.invoke({"messages": [HumanMessage("Status of A1?")]})
    tools  = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.content for message in tools] == ["case A1 is open"]
    assert result["messages"][-1].content == "final: case A1 is open"


#==========================================================================
# create_agent with the default rules
#==========================================================================
def test_default_rules_run_one_tool_then_stop() -> None:
    """
    Pin that the default loop rule calls the first tool once and terminates.
    """
    agent  = create_agent(FakeChatModel(), tools=[lookup])
    result = agent.invoke({"messages": [HumanMessage("Status?")]})
    kinds  = [type(message).__name__ for message in result["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert result["messages"][-1].content.startswith("[FAKE MODEL ANSWER]")
    assert result["messages"][-1].usage_metadata["total_tokens"] > 0


#==========================================================================
# Ambiguous forced tool
#==========================================================================
def test_forcing_any_of_several_tools_stops_with_a_reason() -> None:
    """
    Pin that tool_choice="any" over two tools is refused instead of guessed.
    """
    chat = FakeChatModel().bind_tools([lookup, Verdict], tool_choice="any")
    with pytest.raises(ValueError, match="over 2 tools"):
        chat.invoke("x")
