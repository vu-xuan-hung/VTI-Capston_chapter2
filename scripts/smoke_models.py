"""Smoke-test a real model: structured output and a one-tool agent.

    python scripts/smoke_models.py local --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/smoke_models.py server            # needs: python main.py serve-model

Prints [OK]/[NO] and the time of each step. Downloads nothing when the model
is already in the Hugging Face cache.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  argparse
import  sys
import  time
from    typing                          import  Any, Literal

from    langchain.agents                            import  create_agent
from    langchain.tools                             import  tool
from    langchain_core.language_models.chat_models  import  BaseChatModel
from    langchain_core.messages                     import  HumanMessage, SystemMessage, ToolMessage
from    pydantic                                    import  BaseModel, Field

from    factdesk.config                 import  ServerConfig


#==========================================================================
# Structured answer
#==========================================================================
class Triage(BaseModel):
    """
    Purpose: the structured answer of the smoke test.
    """

    topic: Literal["billing", "delivery", "other"] = Field(description="what the message is about")
    urgent: bool                                    = Field(description="true when it must be handled today")
    summary: str                                    = Field(description="the request in at most 12 words")


#==========================================================================
# The one tool
#==========================================================================
@tool
def order_status(order_id: str) -> str:
    """
    Look up the shipping status of one order by its number.

    Input:
        . order_id      : the order number, digits only
    Output:
        . status        : one sentence
    """
    return f"Order {order_id} shipped on 2026-11-10 and arrives on 2026-11-18."


#==========================================================================
# Structured output
#==========================================================================
def check_structured(chat: BaseChatModel) -> bool:
    """
    Ask for a Triage of a late-delivery message and time it.

    Input:
        . chat          : any LangChain chat model
    Output:
        . passed        : True when the answer parsed into Triage (the plumbing);
                          whether the label is right is printed, not checked
    """
    started = time.time()
    answer  = chat.with_structured_output(Triage).invoke([
        SystemMessage("Classify the customer message."),
        HumanMessage("My parcel has not arrived after 10 days and I need it for a wedding tomorrow!"),
    ])
    passed  = isinstance(answer, Triage)
    right   = "right" if answer.topic == "delivery" else "WRONG (expected delivery)"
    print(f"[{'OK' if passed else 'NO'}] structured {time.time() - started:6.1f} s  {answer!r}  label {right}",
          flush=True)
    return passed


#==========================================================================
# One-tool agent
#==========================================================================
def check_agent(chat: BaseChatModel) -> bool:
    """
    Run create_agent with one tool on a question that needs it, and time it.

    Input:
        . chat          : any LangChain chat model with bind_tools
    Output:
        . passed        : True when the tool ran with 4521 and the answer has the date
    """
    started  = time.time()
    agent    = create_agent(chat, tools=[order_status], system_prompt="Use the tools for facts. Answer briefly.")
    result   = agent.invoke({"messages": [HumanMessage("Where is my order 4521?")]})
    messages: list[Any] = result["messages"]
    called   = [message.content for message in messages if isinstance(message, ToolMessage)]
    final    = messages[-1].content
    passed   = bool(called) and "4521" in called[0] and "18" in final
    print(f"[{'OK' if passed else 'NO'}] agent      {time.time() - started:6.1f} s  "
          f"tool={called}  answer={' '.join(final.split())[:120]!r}", flush=True)
    return passed


#==========================================================================
# Command line
#==========================================================================
def main(argv: list[str]) -> int:
    """
    Run both checks on a local model, or the structured check through the server.

    Input:
        . argv          : command-line arguments without the program name
    Output:
        . code          : 0 when every check passed, else 1
    """
    parser = argparse.ArgumentParser(description="Real-model smoke test")
    parser.add_argument("where", choices=["local", "server"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--agent", action="store_true", help="server: also run the agent check")
    args = parser.parse_args(argv)
    if args.where == "local":
        from    factdesk.llm.local_chat_model   import  LocalChatModel

        started = time.time()
        chat    = LocalChatModel(model_name=args.model, max_new_tokens=200)
        print(f"[OK] loaded {args.model} in {time.time() - started:.1f} s", flush=True)
        return 0 if all([check_structured(chat), check_agent(chat)]) else 1
    from    factdesk.llm.server_chat        import  ServerChat

    chat   = ServerChat(base_url=ServerConfig().model_url, model_name=args.model, max_tokens=200)
    checks = [check_structured(chat), *([check_agent(chat)] if args.agent else [])]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
