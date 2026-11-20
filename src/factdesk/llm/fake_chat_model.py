"""Answer like a chat model, by rules, for tests and for checking code fast.

The class never changes per project: a project passes FakeRules with its own
three functions (see llm/fake_rules.py). A rule left as None uses the kit
default, which works for any schema and any tool list.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    collections.abc                 import  Callable
from    dataclasses                     import  dataclass
from    typing                          import  Any

from    langchain_core.language_models.chat_models  import  BaseChatModel
from    langchain_core.messages                     import  AIMessage, BaseMessage, HumanMessage, ToolMessage
from    langchain_core.outputs                      import  ChatGeneration, ChatResult
from    langchain_core.utils.function_calling       import  convert_to_openai_tool


#==========================================================================
# Rule signatures
#==========================================================================
type StructuredRule = Callable[[str, str], dict[str, Any]]
type TextRule       = Callable[[str], str]
type ToolRule       = Callable[[list[BaseMessage], list[dict[str, Any]]], list[dict[str, Any]] | None]


#==========================================================================
# The project's rules
#==========================================================================
@dataclass(frozen=True)
class FakeRules:
    """
    Purpose: what the fake model answers, as three plain functions.
      . structured(schema_name, prompt) -> dict: the arguments of a forced
        tool call, i.e. the fields of with_structured_output(Schema)
      . text(prompt) -> str: a plain answer
      . tool_calls(messages, tools) -> list | None: inside a tool loop
        (create_agent), the calls to make as [{"name", "args"}], or None to
        answer with text(prompt)
    prompt is every message's text joined by newlines, the last one last.
    """

    structured: StructuredRule | None   = None
    text: TextRule | None               = None
    tool_calls: ToolRule | None         = None


#==========================================================================
# Rule-based chat model
#==========================================================================
class FakeChatModel(BaseChatModel):
    """
    Purpose: the same interface as a real chat model, with no download and
    the same answer every time; works with with_structured_output and inside
    create_agent tool loops.
    """

    rules: Any                          = FakeRules()

    #----------------------------------------------------------------------
    # Name of this model type in traces
    #----------------------------------------------------------------------
    @property
    def _llm_type(self) -> str:
        """
        Identify this wrapper in LangChain and LangSmith traces.

        Output:
            . name          : a fixed label
        """
        return "fake-rules-chat"

    #----------------------------------------------------------------------
    # Attach tools the way a real model does
    #----------------------------------------------------------------------
    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        """
        Bind tool schemas; structured output and create_agent both use this.

        Input:
            . tools         : LangChain tools, Pydantic classes or OpenAI dicts
            . tool_choice   : None, "auto", "any", "required", a name or a dict
            . kwargs        : other bound arguments, passed on
        Output:
            . model         : a Runnable that sends tools to _generate
        """
        schemas = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=schemas, tool_choice=tool_choice, **kwargs)

    #----------------------------------------------------------------------
    # Answer by the rules
    #----------------------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Fill a forced tool, call tools in a loop, or write text.

        Input:
            . messages      : the conversation
            . stop          : ignored
            . run_manager   : LangChain callback manager
            . tools         : bound OpenAI tool schemas, or None
            . tool_choice   : bound tool choice; a forced one means structured output
            . kwargs        : other bound arguments, ignored
        Output:
            . result        : one AIMessage, with tool_calls when a tool is called
        """

        #------------------------------------------------------------------
        # Choose the tool calls: forced (structured output) or by the loop rule
        #------------------------------------------------------------------
        prompt = "\n".join(message.text for message in messages)
        forced = _forced_tool(tools, tool_choice)
        if forced is not None:
            name = forced["function"]["name"]
            args = (
                self.rules.structured(name, prompt)
                if self.rules.structured
                else placeholder_args(forced["function"]["parameters"])
            )
            calls = [{"name": name, "args": args}]
        elif tools:
            pick  = self.rules.tool_calls or _call_first_tool_once
            calls = pick(messages, tools) or []
        else:
            calls = []
        #------------------------------------------------------------------
        # Write the message and rough token counts (words)
        #------------------------------------------------------------------
        if calls:
            message = AIMessage(content="", tool_calls=_with_ids(calls, len(messages)))
        else:
            write   = self.rules.text or _placeholder_text
            message = AIMessage(content=write(prompt))
        words = len(prompt.split())
        message.usage_metadata = {
            "input_tokens": words,
            "output_tokens": len(message.text.split()),
            "total_tokens": words + len(message.text.split()),
        }
        return ChatResult(generations=[ChatGeneration(message=message)])


#==========================================================================
# Placeholder arguments for any JSON schema
#==========================================================================
def placeholder_args(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Fill every property of a tool's parameter schema with a visible
    placeholder ("[FAKE]", 0, False, [], the first enum value), recursively.

    Input:
        . schema        : a JSON schema object, e.g. tool["function"]["parameters"]
    Output:
        . args          : one value per property
    """
    defs = schema.get("$defs", {})
    return {
        name: _placeholder(prop, defs) for name, prop in schema.get("properties", {}).items()
    }


#==========================================================================
# One placeholder value
#==========================================================================
def _placeholder(prop: dict[str, Any], defs: dict[str, Any]) -> Any:
    """
    Choose a placeholder for one property of a JSON schema.

    Input:
        . prop          : the property's schema
        . defs          : the $defs of the root schema, for $ref
    Output:
        . value         : a value of the property's type
    """
    if "$ref" in prop:
        return _placeholder(defs[prop["$ref"].split("/")[-1]], defs)
    if "enum" in prop:
        return prop["enum"][0]
    if "anyOf" in prop:
        options = [option for option in prop["anyOf"] if option.get("type") != "null"]
        return _placeholder(options[0], defs) if options else None
    kind = prop.get("type", "string")
    if kind == "object":
        return {name: _placeholder(sub, defs) for name, sub in prop.get("properties", {}).items()}
    values = {"string": "[FAKE]", "integer": 0, "number": 0.0, "boolean": False, "array": []}
    return values[kind]


#==========================================================================
# The tool a caller forces, if any
#==========================================================================
def _forced_tool(tools: list[dict[str, Any]] | None, tool_choice: Any) -> dict[str, Any] | None:
    """
    Return the tool that tool_choice forces, or None when the model may
    choose. Forcing "any" of several tools is ambiguous for rules: it stops.

    Input:
        . tools         : bound OpenAI tool schemas, or None
        . tool_choice   : None, "auto", "none", "any", "required", a name or a dict
    Output:
        . tool          : the forced tool schema, or None
    """
    if not tools or not tool_choice or tool_choice in ("auto", "none"):
        return None
    if tool_choice in ("any", "required", True):
        if len(tools) != 1:
            raise ValueError(
                f"fake model: tool_choice={tool_choice!r} over {len(tools)} tools; "
                "name the tool, or give FakeRules.tool_calls and leave tool_choice unset"
            )
        return tools[0]
    name = tool_choice["function"]["name"] if isinstance(tool_choice, dict) else tool_choice
    for tool in tools:
        if tool["function"]["name"] == name:
            return tool
    raise ValueError(f"fake model: tool_choice names {name!r}, which is not bound")


#==========================================================================
# Default tool-loop rule
#==========================================================================
def _call_first_tool_once(
    messages: list[BaseMessage], tools: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """
    Call the first tool once after each human message, then answer: enough
    for a create_agent loop to run tool -> answer and stop.

    Input:
        . messages      : the conversation so far
        . tools         : bound OpenAI tool schemas
    Output:
        . calls         : one placeholder call, or None once a tool answered
    """
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return None
        if isinstance(message, HumanMessage):
            break
    first = tools[0]["function"]
    return [{"name": first["name"], "args": placeholder_args(first["parameters"])}]


#==========================================================================
# Default text rule
#==========================================================================
def _placeholder_text(prompt: str) -> str:
    """
    Write an answer nobody could mistake for a real one.

    Input:
        . prompt        : every message's text, the last one last
    Output:
        . text          : "[FAKE MODEL ANSWER]" and the end of the prompt
    """
    last = " ".join(prompt.split()[-24:])
    return f"[FAKE MODEL ANSWER] {last}"


#==========================================================================
# Stable ids for tool calls
#==========================================================================
def _with_ids(calls: list[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    """
    Give each call an id that is unique in the conversation and the same on
    every run.

    Input:
        . calls         : [{"name", "args"}, ...] from a rule
        . turn          : the number of messages so far
    Output:
        . calls         : LangChain tool calls with id and type
    """
    return [
        {"name": call["name"], "args": call["args"], "id": f"call_{turn}_{index}", "type": "tool_call"}
        for index, call in enumerate(calls)
    ]
