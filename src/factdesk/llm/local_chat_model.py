"""Run a local Hugging Face chat model as a LangChain chat model."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  os
import  time
from    typing                          import  Any

import  torch
from    langchain_core.language_models.chat_models  import  BaseChatModel
from    langchain_core.messages                     import  AIMessage, BaseMessage, HumanMessage
from    langchain_core.messages                     import  SystemMessage, ToolMessage
from    langchain_core.outputs                      import  ChatGeneration, ChatResult
from    langchain_core.utils.function_calling       import  convert_to_openai_tool
from    transformers                                import  AutoModelForCausalLM, AutoTokenizer

from    factdesk.ai_logger              import  aiLogger
from    factdesk.llm.tool_text          import  parse_tool_calls, with_tool_instruction


#==========================================================================
# Local Transformers chat model
#==========================================================================
class LocalChatModel(BaseChatModel):
    """
    Purpose: a Qwen-style chat model on this laptop (no server, no key) with
    tool calling. Greedy decoding: the same input gives the same answer.
    """

    model_name: str
    max_new_tokens: int                 = 300
    device: str                         = "cpu"
    tok: Any                            = None
    model: Any                          = None

    #----------------------------------------------------------------------
    # Load the tokenizer and the weights once
    #----------------------------------------------------------------------
    def __init__(self, **kwargs: Any) -> None:
        """
        Load the weights from the Hugging Face cache (download on first use).

        Input:
            . kwargs        : model_name and optionally max_new_tokens
        """
        super().__init__(**kwargs)
        started     = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype       = torch.float16 if self.device == "cuda" else torch.bfloat16
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.tok    = AutoTokenizer.from_pretrained(self.model_name)
        weights     = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        self.model  = weights.to(self.device).eval()
        aiLogger.info(
            "Chat model %s on %s, loaded in %.0f s",
            self.model_name,
            self.device,
            time.time() - started,
        )

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
        return "local-transformers-chat"

    #----------------------------------------------------------------------
    # Attach tools
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
    # Generate one answer
    #----------------------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Decode greedily and turn <tool_call> blocks into tool calls.

        Input:
            . messages      : the conversation
            . stop          : ignored; the chat template ends the turn
            . run_manager   : LangChain callback manager
            . tools         : bound OpenAI tool schemas, or None
            . tool_choice   : bound tool choice, or None
            . max_new_tokens: per-call limit (the model server sends one)
            . kwargs        : other bound arguments, ignored
        Output:
            . result        : one AIMessage with tool_calls and usage_metadata
        """

        #------------------------------------------------------------------
        # Build the prompt with the chat template
        #------------------------------------------------------------------
        dicts = with_tool_instruction(_to_dicts(messages), tools, tool_choice)
        ids   = self.tok.apply_chat_template(
            dicts,
            tools                   = tools,
            add_generation_prompt   = True,
            return_tensors          = "pt",
            return_dict             = True,
        ).to(self.device)

        #------------------------------------------------------------------
        # Decode greedily
        #------------------------------------------------------------------
        with torch.inference_mode():
            out = self.model.generate(
                **ids,
                max_new_tokens  = max_new_tokens or self.max_new_tokens,
                do_sample       = False,
                pad_token_id    = self.tok.eos_token_id,
            )

        #------------------------------------------------------------------
        # Read the answer, the tool calls and the token counts
        #------------------------------------------------------------------
        size           = int(ids["input_ids"].shape[1])
        text           = self.tok.decode(out[0, size:], skip_special_tokens=True).strip()
        content, calls = parse_tool_calls(text)
        usage          = {
            "input_tokens": size,
            "output_tokens": int(out.shape[1]) - size,
            "total_tokens": int(out.shape[1]),
        }
        message = AIMessage(content=content, tool_calls=calls, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])


#==========================================================================
# LangChain messages -> chat-template dictionaries
#==========================================================================
def _to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """
    Convert messages to the role/content dictionaries of the chat template;
    earlier tool calls keep their arguments as a dict, as Qwen expects.

    Input:
        . messages      : system, human, AI and tool messages
    Output:
        . dicts         : chat-template messages in the same order
    """

    #----------------------------------------------------------------------
    # Map each message type to its template role
    #----------------------------------------------------------------------
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            out.append({"role": "system", "content": message.text})
        elif isinstance(message, HumanMessage):
            out.append({"role": "user", "content": message.text})
        elif isinstance(message, AIMessage):
            item: dict[str, Any] = {"role": "assistant", "content": message.text}
            if message.tool_calls:
                item["tool_calls"] = [
                    {"type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
                    for c in message.tool_calls
                ]
            out.append(item)
        elif isinstance(message, ToolMessage):
            out.append({"role": "tool", "content": message.text})
        else:
            raise ValueError(f"unsupported message type for a local model: {type(message).__name__}")
    return out
