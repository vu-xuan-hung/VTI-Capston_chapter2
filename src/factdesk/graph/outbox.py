"""An outside effect that must not repeat: one request line per clarification.

When a graph resumes after an interrupt, the node that interrupted runs again
from its first line. Anything it did before the interrupt would happen twice.
notify_once writes a line only if its key is not there yet.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
from    pathlib                         import  Path
from    typing                          import  Any


#==========================================================================
# Write once
#==========================================================================
def notify_once(outbox: Path, *, key: str, payload: dict[str, Any]) -> bool:
    """
    Append {"key", **payload} to the outbox file unless a line with this key exists.
    It stands for a real effect (an email to the editor, a ticket).

    Input:
        . outbox        : the JSON-lines file, e.g. runtime/outbox.jsonl
        . key           : identity of the request, e.g. "<thread id>:<country text>"
        . payload       : what to record
    Output:
        . written       : True when a line was added, False when the key was already there
    """
    outbox.parent.mkdir(parents=True, exist_ok=True)
    lines = outbox.read_text(encoding="utf-8").splitlines() if outbox.exists() else []
    if any(json.loads(line)["key"] == key for line in lines):
        return False
    with outbox.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"key": key, **payload}, ensure_ascii=False) + "\n")
    return True
