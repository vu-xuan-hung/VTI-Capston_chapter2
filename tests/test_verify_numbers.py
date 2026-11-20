"""verify/numbers.py: reading numbers out of text and checking them; strict but not brittle."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  pytest

from    factdesk.config                 import  FormatConfig
from    factdesk.verify.numbers         import  extract_numbers, find_unverified


#==========================================================================
# Reading numbers
#==========================================================================
def test_extract_reads_commas_decimals_scale_words_and_percent() -> None:
    """
    Pin extract_numbers: "2,345" is one number, "97.5 million" carries its scale,
    "12%" is a percent, "1.26 times" is a plain decimal, a sign is ignored and
    a number glued to letters ("10km", "2nd") is not a number of the answer.
    """
    found = extract_numbers("It was 2,345 or 97.5 million, up 12% (-3.5) and 1.26 times, 10km, 2nd.", numbers=FormatConfig())
    assert [(item.value, item.scale, item.percent, item.decimals) for item in found] == [
        (2345.0, 1.0, False, 0), (97.5, 1e6, False, 1), (12.0, 1.0, True, 0), (3.5, 1.0, False, 1), (1.26, 1.0, False, 2),
    ]


#==========================================================================
# Checking numbers
#==========================================================================
FACTS = [{"code": "VNM", "year": 2015, "indicator": "population", "value": 92823254},
         {"code": "JPN", "year": 2020, "indicator": "gdp", "value": 4256410760723.75}]
RESULT = {"label": "ratio JPN/VNM", "value": 1.26}


#==========================================================================
# Numbers that are the facts rounded pass
#==========================================================================
@pytest.mark.parametrize("answer", [
    "Viet Nam had 92,823,254 people in 2015.",
    "About 92.8 million people lived in Viet Nam in 2015.",
    "Japan's GDP was $4.26 trillion in 2020, 1.26 times as much.",
    "Japan's GDP was 4.3 trillion dollars.",
    "Two countries: the ratio is 1.3 times.",
])
def test_numbers_that_are_the_facts_rounded_pass(answer: str) -> None:
    """
    Pin that a number passes when it is a fact, the result, a known year or a
    counting word, or the fact rounded to the digits shown.

    Input:
        . answer        : a written answer
    """
    assert find_unverified(answer, facts=FACTS, result=RESULT, years=[2015, 2020], numbers=FormatConfig()) == []


#==========================================================================
# Numbers that are not the facts are reported
#==========================================================================
@pytest.mark.parametrize(("answer", "bad"), [
    ("Viet Nam had 97,338,579 people.", ["97,338,579"]),
    ("Viet Nam had 97.5 million people.", ["97.5 million"]),
    ("Japan's GDP was $5.1 trillion.", ["5.1 trillion"]),
    ("The ratio was 1.5 times in 1998.", ["1.5", "1998"]),
    ("Growth was 12% and 14 countries joined.", ["12%", "14"]),
])
def test_numbers_that_are_not_the_facts_are_reported(answer: str, bad: list[str]) -> None:
    """
    Pin that a wrong number, a percent that is not the result, a year that is not
    in the question and a whole number above the counting-word limit are reported.

    Input:
        . answer        : a written answer
        . bad           : the numbers the check must report
    """
    assert find_unverified(answer, facts=FACTS, result=RESULT, years=[2015, 2020], numbers=FormatConfig()) == bad


#==========================================================================
# The rule is strict about digits
#==========================================================================
def test_a_number_may_differ_only_by_its_own_rounding() -> None:
    """
    Pin the tolerance: half a unit of the last digit shown, nothing more.
    """
    numbers = FormatConfig()
    ok      = find_unverified("It is 4.26 trillion.", facts=FACTS, result=None, years=[], numbers=numbers)
    far     = find_unverified("It is 4.27 trillion.", facts=FACTS, result=None, years=[], numbers=numbers)
    assert ok == [] and far == ["4.27 trillion"]
