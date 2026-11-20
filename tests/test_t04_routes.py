"""T04: the three routers of the graph (run: pytest -k t04). Plain dicts in, a node name out."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  pytest
from    langgraph.graph                 import  END

from    factdesk.graph.routes           import  after_compute, after_link, after_verify


#==========================================================================
# After link
#==========================================================================
@pytest.mark.parametrize(("state", "node"), [
    ({"status": "out_of_scope", "pending": []}, "decline"),
    ({"status": "no_data", "unknown": ["Atlantis"]}, "decline"),
    ({"pending": [{"text": "Korea", "candidates": []}]}, "clarify"),
    ({"countries": ["VNM"], "pending": [], "unknown": []}, "compute"),
])
def test_t04_after_link_declines_asks_or_computes(state: dict, node: str) -> None:
    """
    Pin T04: a status set by link wins (decline), then pending names (clarify), else compute.

    Input:
        . state         : the state after link
        . node          : the expected next node
    """
    assert after_link(state) == node


#==========================================================================
# After compute
#==========================================================================
def test_t04_after_compute_explains_an_answer_and_declines_the_rest() -> None:
    """
    Pin T04: status answered goes to explain; no_data and out_of_scope go to decline.
    """
    assert after_compute({"status": "answered"}) == "explain"
    assert after_compute({"status": "no_data"}) == "decline"
    assert after_compute({"status": "out_of_scope"}) == "decline"


#==========================================================================
# After verify: the retry counter
#==========================================================================
def test_t04_after_verify_retries_until_max_tries_then_uses_the_template() -> None:
    """
    Pin T04: verified ends the graph; a wrong answer is written again while tries < max_tries; after
    the last try the template answer is used.
    """
    assert after_verify({"verified": True, "tries": 1, "max_tries": 2}) == END
    assert after_verify({"verified": False, "tries": 1, "max_tries": 2}) == "explain"
    assert after_verify({"verified": False, "tries": 2, "max_tries": 2}) == "template"
    assert after_verify({"verified": False, "tries": 1, "max_tries": 1}) == "template"
