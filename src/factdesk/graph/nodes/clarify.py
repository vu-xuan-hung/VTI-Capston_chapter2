"""Node clarify: stop and ask the user which country they mean (interrupt), then apply the choice."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                          import  annotations

from    os                                  import  name
from    pathlib                             import  Path
from    typing                              import  Any

from    langgraph.types                     import  interrupt

from    factdesk.graph.outbox               import  notify_once
from    factdesk.graph.state                import  FactState


#==========================================================================
# Clarify
#==========================================================================
def clarify_node(state: FactState, *, outbox: Path) -> dict[str, Any]:
    """
    Ask about the first ambiguous name: record the request in the outbox (once),
    interrupt with the candidates, and on resume store the chosen code in
    "resolved" for that name and clear "pending". The graph goes back to link,
    which asks again if another name is still ambiguous.

    Input:
        . state         : holds pending [{"text", "candidates"}], question, thread_id, resolved
        . outbox        : where the clarification request is recorded
    Output:
        . update        : resolved, pending, log
    """

    #----------------------------------------------------------------------
    # Record the clarification request before pausing
    #----------------------------------------------------------------------
    text_contry = state["pending"][0]
    notify_once(
        outbox,
        key=f"{state['thread_id']}:{text_contry['text']}",
        payload={
            "question": state["question"],
            "text": text_contry["text"],
            "options": text_contry["candidates"],
        },
    )
    choice = interrupt(
        {
            "question": state["question"],
            "text": text_contry["text"],
            "options": text_contry["candidates"],
        }
    )
    codes = [candidate["code"] for candidate in text_contry["candidates"]]
    if choice not in codes:
        raise ValueError(f"Choice {choice!r} is not one of {codes}")
    return {
        "resolved": {**state.get("resolved", {}), text_contry["text"]: choice},
        "pending": [],
        "log": [f"clarify: {text_contry['text']} -> {choice}"],
    }
