"""Write numbers for people, the same way every time: the prompt, the template answer and the check use it."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    typing                          import  Any

from    factdesk.config                 import  FormatConfig


#==========================================================================
# One amount of an indicator
#==========================================================================
def format_amount(indicator: str, value: float, numbers: FormatConfig) -> str:
    """
    Write a value of an indicator: people as a whole number with commas, GDP in
    millions, billions or trillions of US$ with money_decimals, GDP per person
    in US$ with per_person_decimals.

    Input:
        . indicator     : population, gdp or gdp_per_capita
        . value         : the number from the facts table
        . numbers       : decimals per kind
    Output:
        . text          : e.g. "92,677,076", "$4.26 trillion", "$39,986.93"
    """
    if indicator == "population":
        return f"{round(value):,}"
    if indicator == "gdp_per_capita":
        return f"${value:,.{numbers.per_person_decimals}f}"
    for word, scale in sorted(numbers.scales.items(), key=lambda item: -item[1]):
        if word != "thousand" and abs(value) >= scale:
            return f"${value / scale:,.{numbers.money_decimals}f} {word}"
    return f"${value:,.{numbers.money_decimals}f}"


#==========================================================================
# The computed result
#==========================================================================
def format_result(result: dict[str, Any], indicator: str, numbers: FormatConfig) -> str:
    """
    Write a result by its kind: ratio, percent, or an amount of the indicator.

    Input:
        . result        : {"label", "value", "kind"} from compute
        . indicator     : the indicator the question is about
        . numbers       : decimals per kind
    Output:
        . text          : e.g. "1.26", "24.33%", "$4.26 trillion"
    """
    if result["kind"] == "ratio":
        return f"{result['value']:.{numbers.ratio_decimals}f}"
    if result["kind"] == "percent":
        return f"{result['value']:.{numbers.percent_decimals}f}%"
    return format_amount(indicator, result["value"], numbers)


#==========================================================================
# The facts table shown to the model
#==========================================================================
def facts_lines(
    facts: list[dict[str, Any]],
    result: dict[str, Any] | None,
    *,
    names: dict[str, str],
    titles: dict[str, str],
    numbers: FormatConfig,
) -> list[str]:
    """
    One line per fact and one for the result, every number already formatted:
    the model copies them, the check reads them back.

    Input:
        . facts         : [{"code", "year", "indicator", "value"}]
        . result        : {"label", "value", "kind"} or None
        . names         : code -> country name
        . titles        : indicator -> title
        . numbers       : decimals per kind
    Output:
        . lines         : "Japan (JPN) | 2020 | GDP per person | $39,986.93" and "RESULT ratio JPN/KOR: 1.26"
    """
    lines = [
        f"{names[fact['code']]} ({fact['code']}) | {fact['year']} | {titles[fact['indicator']]} | "
        f"{format_amount(fact['indicator'], fact['value'], numbers)}"
        for fact in facts
    ]
    if result is not None:
        indicator = facts[0]["indicator"]
        lines.append(f"RESULT {result['label']}: {format_result(result, indicator, numbers)}")
    return lines


#==========================================================================
# The answer written by code
#==========================================================================
def template_answer(
    facts: list[dict[str, Any]],
    result: dict[str, Any] | None,
    *,
    names: dict[str, str],
    titles: dict[str, str],
    numbers: FormatConfig,
) -> str:
    """
    Write the answer by code, never by the model: one sentence per fact and
    one for the result. Plain, always correct, used when the model's numbers
    did not verify.

    Input:
        . facts         : [{"code", "year", "indicator", "value"}]
        . result        : {"label", "value", "kind"} or None
        . names         : code -> country name
        . titles        : indicator -> title
        . numbers       : decimals per kind
    Output:
        . text          : e.g. "Japan, 2020, GDP per person: $39,986.93. Result (ratio JPN/KOR): 1.26."
    """
    parts = [
        f"{names[fact['code']]}, {fact['year']}, {titles[fact['indicator']]}: "
        f"{format_amount(fact['indicator'], fact['value'], numbers)}."
        for fact in facts
    ]
    if result is not None and result["label"].split()[0] != facts[0]["indicator"]:
        parts.append(f"Result ({result['label']}): {format_result(result, facts[0]['indicator'], numbers)}.")
    return " ".join(parts)
