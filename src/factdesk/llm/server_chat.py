"""Call the OpenAI-compatible model server as a LangChain chat model."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
from    typing                          import  Any

from    langchain_core.language_models.chat_models  import  BaseChatModel
from    langchain_core.messages                     import  AIMessage, BaseMessage, convert_to_openai_messages
from    langchain_core.outputs                      import  ChatGeneration, ChatResult
from    langchain_core.utils.function_calling       import  convert_to_openai_tool

from    factdesk.integrations.http_json  import  request_json


#==========================================================================
# Model server client
#==========================================================================
class ServerChat(BaseChatModel):
    """
    Purpose: what ChatOpenAI(base_url=...) does, in a few lines: the same
    request Langflow's OpenAI component sends, with tools and tool calls, so
    structured output and create_agent work through the server too.
    """

    base_url: str
    model_name: str
    max_tokens: int                     = 300
    timeout: float                      = 300.0

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
        return "local-server-chat"

    #----------------------------------------------------------------------
    # Attach tools in the OpenAI format
    #----------------------------------------------------------------------
    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        """
        Bind tool schemas; LangChain's "any" becomes OpenAI's "required" and
        a bare tool name becomes {"type": "function", "function": {"name"}}.

        Input:
            . tools         : LangChain tools, Pydantic classes or OpenAI dicts
            . tool_choice   : None, "auto", "none", "any", "required", a name or a dict
            . kwargs        : other bound arguments, passed on
        Output:
            . model         : a Runnable that sends tools to _generate
        """
        schemas = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice == "any":
            tool_choice = "required"
        elif isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
            tool_choice = {"type": "function", "function": {"name": tool_choice}}
        return self.bind(tools=schemas, tool_choice=tool_choice, **kwargs)

    #----------------------------------------------------------------------
    # Send one request
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
        POST /chat/completions with temperature 0 and read the answer back.

        Input:
            . messages      : the conversation
            . stop          : ignored; the server ends the turn
            . run_manager   : LangChain callback manager
            . tools         : bound OpenAI tool schemas, or None
            . tool_choice   : bound OpenAI tool choice, or None
            . kwargs        : other bound arguments, ignored
        Output:
            . result        : one AIMessage with tool_calls and usage_metadata
        """

        #------------------------------------------------------------------
        # Build the OpenAI request
        #------------------------------------------------------------------
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": convert_to_openai_messages(messages),
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice

        #------------------------------------------------------------------
        # Read the message, its tool calls and the token counts
        #------------------------------------------------------------------
        out    = request_json(f"{self.base_url}/chat/completions", body, timeout=self.timeout)
        answer = out["choices"][0]["message"]
        calls  = [
            {
                "name": call["function"]["name"],
                "args": json.loads(call["function"]["arguments"] or "{}"),
                "id": call["id"],
                "type": "tool_call",
            }
            for call in answer.get("tool_calls") or []
        ]

        usage   = out.get("usage", {})
        meta    = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        message = AIMessage(content=answer.get("content") or "", tool_calls=calls, usage_metadata=meta)
        return ChatResult(generations=[ChatGeneration(message=message)])
