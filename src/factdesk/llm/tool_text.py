"""Turn tool calls into text for a local model, and its <tool_call> text back.

Hosted APIs force a tool call while decoding. A local Qwen model only sees the
tools in its chat template and writes <tool_call>{json}</tool_call> blocks;
when a call is required it is asked to in the system text.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
import  re
import  uuid
from    typing                          import  Any

from    factdesk.ai_logger              import  aiLogger


#==========================================================================
# Ask for a tool call when the caller requires one
#==========================================================================
def with_tool_instruction(
    dicts: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
) -> list[dict[str, Any]]:
    """
    Add one sentence to the system message when tool_choice forces a call
    (with_structured_output binds "any"; OpenAI clients send "required" or
    {"type": "function", "function": {"name": ...}}).

    Input:
        . dicts         : chat-template messages (role, content)
        . tools         : OpenAI tool schemas, or None
        . tool_choice   : None, "auto", "none", "any", "required", a name or a dict
    Output:
        . dicts         : the same messages, the system text extended if forced
    """
    if not tools or not tool_choice or tool_choice in ("auto", "none"):
        return dicts
    name = None
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name")
    elif tool_choice not in ("any", "required"):
        name = tool_choice
    if name:
        must = f"You must answer by calling the function {name} inside <tool_call></tool_call> tags."
    else:
        must = "You must answer by calling one of the functions inside <tool_call></tool_call> tags."
    if dicts and dicts[0]["role"] == "system":
        return [dict(dicts[0], content=f"{dicts[0]['content']}\n{must}"), *dicts[1:]]
    return [{"role": "system", "content": must}, *dicts]


#==========================================================================
# Read <tool_call> blocks out of generated text
#==========================================================================
def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Split generated text into its answer text and LangChain tool calls. An
    unreadable block stays in the text, where a person or a test sees it.

    Input:
        . text          : decoded model output
    Output:
        . content       : the text without the readable tool-call blocks
        . calls         : [{"name", "args", "id", "type": "tool_call"}, ...]
    """

    #----------------------------------------------------------------------
    # Parse each block; keep the unreadable ones as text
    #----------------------------------------------------------------------
    calls   = []
    content = text
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S):
        try:
            data = json.loads(match.group(1))
            call = {
                "name": data["name"],
                "args": data.get("arguments", {}),
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            aiLogger.warning("Unreadable tool call kept as text: %s", match.group(1)[:80])
            continue
        calls.append(call)
        content = content.replace(match.group(0), "")
    return content.strip(), calls
