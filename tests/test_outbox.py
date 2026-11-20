"""graph/outbox.py: an outside effect that must not repeat."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    pathlib                         import  Path

from    factdesk.graph.outbox           import  notify_once


#==========================================================================
# Write once
#==========================================================================
def test_notify_once_writes_a_key_only_the_first_time(tmp_path: Path) -> None:
    """
    Pin notify_once: the same key is written once, a new key is written.

    Input:
        . tmp_path      : temporary folder
    """
    outbox = tmp_path / "runtime" / "outbox.jsonl"
    assert notify_once(outbox, key="t1:Korea", payload={"text": "Korea"}) is True
    assert notify_once(outbox, key="t1:Korea", payload={"text": "Korea"}) is False
    assert notify_once(outbox, key="t2:Korea", payload={"text": "Korea"}) is True
    assert len(outbox.read_text(encoding="utf-8").splitlines()) == 2
