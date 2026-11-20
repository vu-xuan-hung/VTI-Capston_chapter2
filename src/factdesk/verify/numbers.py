"""Read the numbers out of a written answer and check each one against the facts.

Strict but not brittle: "97.5 million", "97,500,000" and "97.5 M" style
writings are read by their own rounding (a number may differ from the fact by
half a unit of its last digit), "12%" is a number, "2,345" is one number, and
years and counting words ("two countries") are handled by rule. Anything
else that is not in the facts is reported. The same function is the
`verify` step of the graph and the core of the evaluator answer_numbers_verified.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  re
from    dataclasses                     import  dataclass
from    typing                          import  Any

from    factdesk.config                 import  FormatConfig


#==========================================================================
# One number found in a text
#==========================================================================
@dataclass(frozen=True)
class Mention:
    """
    Purpose: one number as written: its text, its value, how many decimals it
    shows, the scale word after it (million, ...) and whether it was a plain
    whole number (no comma, decimal point, scale word or percent sign).
    """

    text: str
    value: float
    decimals: int
    scale: float
    percent: bool
    plain: bool


#==========================================================================
# Find the numbers
#==========================================================================
def extract_numbers(text: str, *, numbers: FormatConfig) -> list[Mention]:
    """
    Find every number in a text, with the scale word or percent sign after it.
    A sign is ignored (direction is a word, not a number); a number glued to
    letters ("10km", "2nd") is not a number of the answer.

    Input:
        . text          : the written answer
        . numbers       : the scale words
    Output:
        . mentions      : in reading order
    """
    #----------------------------------------------------------------------
    # The pattern: a number, then an optional scale word or percent sign
    #----------------------------------------------------------------------
    scale_words = "|".join(sorted(numbers.scales, key=len, reverse=True))
    pattern     = re.compile(
        rf"(?<![\w.])(?P<int>\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.(?P<dec>\d+))?(?!\d)"
        rf"(?:\s*(?P<unit>%|percent\b|{scale_words}\b))?(?![A-Za-z]{{2,}})",
        re.IGNORECASE,
    )
    #----------------------------------------------------------------------
    # Read each match
    #----------------------------------------------------------------------
    found = []
    for match in pattern.finditer(text):
        digits  = match["int"].replace(",", "") + ("." + match["dec"] if match["dec"] else "")
        unit    = (match["unit"] or "").lower()
        found.append(Mention(
            text        = match[0].strip(),
            value       = float(digits),
            decimals    = len(match["dec"] or ""),
            scale       = numbers.scales.get(unit, 1.0),
            percent     = unit in ("%", "percent"),
            plain       = "," not in match["int"] and not match["dec"] and not unit,
        ))
    return found


#==========================================================================
# Does a number match a fact
#==========================================================================
def matches(mention: Mention, target: float, *, numbers: FormatConfig) -> bool:
    """
    Say whether a written number is the target rounded to the digits it shows.

    Input:
        . mention       : the number as written
        . target        : a fact or result value
        . numbers       : slack (1.0 = half a unit of the last digit)
    Output:
        . same          : True when |written - target| <= half a unit of its last digit
    """
    written = mention.value * mention.scale
    half    = 0.5 * 10 ** (-mention.decimals) * mention.scale * numbers.slack
    return abs(written - abs(target)) <= half + 1e-9 * max(1.0, abs(target))


#==========================================================================
# The check
#==========================================================================
def find_unverified(
    answer: str,
    *,
    facts: list[dict[str, Any]],
    result: dict[str, Any] | None,
    years: list[int],
    numbers: FormatConfig,
) -> list[str]:
    """
    Return the numbers of the answer that are neither a fact, nor the result,
    nor a year of the question, nor a small counting word ("two"): empty
    means the answer is verified.

    Input:
        . answer        : the model's text
        . facts         : [{"code", "year", "indicator", "value"}]
        . result        : {"label", "value", ...} or None
        . years         : years of the question
        . numbers       : scale words, small_int_max, year range, slack
    Output:
        . unverified    : the text of each number that is not accounted for
    """

    #----------------------------------------------------------------------
    # What the answer may contain
    #----------------------------------------------------------------------
    targets    = [fact["value"] for fact in facts] + ([result["value"]] if result else [])
    known_year = set(years) | {fact["year"] for fact in facts}

    #----------------------------------------------------------------------
    # Read each number by the rules
    #----------------------------------------------------------------------
    bad = []
    for mention in extract_numbers(answer, numbers=numbers):
        if mention.plain and mention.value <= numbers.small_int_max:
            continue
        if mention.plain and numbers.year_min <= mention.value <= numbers.year_max and int(mention.value) in known_year:
            continue
        if not any(matches(mention, target, numbers=numbers) for target in targets):
            bad.append(mention.text)
    return bad
