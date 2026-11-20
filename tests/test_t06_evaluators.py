"""T06: result_exact and answer_numbers_verified (run: pytest -k t06). The other two evaluators are given."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  pytest

from    factdesk.evaluation.evaluators  import  answer_numbers_verified, entities_correct, result_exact, status_correct


#==========================================================================
# result_exact
#==========================================================================
def test_t06_result_exact_compares_the_number_within_the_tolerance() -> None:
    """
    Pin T06: the expected number, a rounding-sized difference and a wrong number; a case that
    expects no number needs a run that produced none.
    """
    ref = {"status": "answered", "result_value": 1.26}
    assert result_exact({}, {"result": {"label": "r", "value": 1.26}}, ref) is True
    assert result_exact({}, {"result": {"label": "r", "value": 1.2601}}, ref) is True
    assert result_exact({}, {"result": {"label": "r", "value": 1.3}}, ref) is False
    assert result_exact({}, {"result": None}, ref) is False
    none = {"status": "no_data", "result_value": None}
    assert result_exact({}, {"result": None}, none) is True
    assert result_exact({}, {"result": {"label": "r", "value": 5}}, none) is False


#==========================================================================
# answer_numbers_verified
#==========================================================================
FACTS = [{"code": "VNM", "year": 2015, "indicator": "population", "value": 92823254}]


#==========================================================================
# Answer numbers verified reads the answer against its own facts
#==========================================================================
@pytest.mark.parametrize(("answer", "ok"), [
    ("Viet Nam had 92,823,254 people in 2015.", True),
    ("Viet Nam had about 92.8 million people in 2015.", True),
    ("Viet Nam had 97 million people in 2015.", False),
    ("I do not know.", True),
])
def test_t06_answer_numbers_verified_reads_the_answer_against_its_own_facts(answer: str, ok: bool) -> None:
    """
    Pin T06: every number of the answer must be a fact (or the result, or a year of the run).

    Input:
        . answer        : the written answer
        . ok            : whether it passes
    """
    outputs = {"answer": answer, "facts": FACTS, "result": None, "years": [2015]}
    assert answer_numbers_verified({"question": "How many people in 2015?"}, outputs, {}) is ok


#==========================================================================
# An answer with numbers but no facts does not pass
#==========================================================================
def test_t06_an_answer_with_numbers_but_no_facts_does_not_pass() -> None:
    """
    Pin T06: an agent that answers from memory (no facts) with a number fails.
    """
    outputs = {"answer": "Canada has about 38 million people.", "facts": [], "result": None, "years": []}
    assert answer_numbers_verified({"question": "Population of Canada?"}, outputs, {}) is False


#==========================================================================
# A decline may repeat the year of the question
#==========================================================================
def test_t06_a_year_written_in_the_question_may_be_repeated_by_a_decline() -> None:
    """
    Pin T06: a run without facts (a correct hand-over) may repeat the year of the question, and only that.
    """
    outputs = {"answer": "There is no GDP for Viet Nam in 2024.", "facts": [], "result": None, "years": []}
    assert answer_numbers_verified({"question": "What was the GDP of Vietnam in 2024?"}, outputs, {}) is True
    assert answer_numbers_verified({"question": "What was the GDP of Vietnam in 2025?"}, outputs, {}) is False


#==========================================================================
# The two given evaluators
#==========================================================================
def test_given_evaluators_compare_entities_as_sets_and_the_status() -> None:
    """
    Pin entities_correct (indicator, countries, years; only the keys the case lists) and status_correct.
    """
    ref = {"status": "answered", "indicator": "gdp", "countries": ["JPN", "KOR"], "years": [2020]}
    out = {"status": "answered", "indicator": "gdp", "countries": ["KOR", "JPN"], "years": [2020]}
    assert entities_correct({}, out, ref) is True and status_correct({}, out, ref) is True
    assert entities_correct({}, {**out, "countries": ["JPN"]}, ref) is False
    assert entities_correct({}, {"indicator": None}, {"status": "out_of_scope", "indicator": None}) is True
    assert status_correct({}, {"status": "no_data"}, ref) is False
