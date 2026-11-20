"""The four operations of the desk: value, compare, growth, rank. Pure code over the facts table.

A question the table cannot answer ends in a status with a reason that names
the missing cell; this module never estimates and never fills a gap.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    dataclasses                     import  dataclass, field
from    typing                          import  Any

from    factdesk.compute.table          import  FactsTable, indicator_value
from    factdesk.config                 import  FormatConfig


#==========================================================================
# What an operation returns
#==========================================================================
@dataclass(frozen=True)
class Computed:
    """
    Purpose: the outcome of one operation: "answered" with facts and a
    result, or "no_data" / "out_of_scope" with the reason and nothing else.
    """

    status: str
    reason: str                         = ""
    facts: list[dict[str, Any]]         = field(default_factory=list)
    result: dict[str, Any] | None       = None
    missing: dict[str, Any] | None      = None      # {"code", "year", "indicator"} of the first missing cell


#==========================================================================
# Run one operation
#==========================================================================
def compute(
    table: FactsTable,
    *,
    operation: str,
    indicator: str,
    countries: list[str],
    years: list[int],
    numbers: FormatConfig,
) -> Computed:
    """
    Check the shape of the request, read every cell it needs, then apply the
    operation. value: one country, one year. compare: two or more countries,
    one year. growth: one country, two years. rank: two or more countries, one year.

    Input:
        . table         : the facts table
        . operation     : value, compare, growth or rank
        . indicator     : population, gdp or gdp_per_capita
        . countries     : World Bank codes in the order of the question
        . years         : calendar years in the order of the question
        . numbers       : rounding of results
    Output:
        . computed      : status answered, no_data or out_of_scope
    """

    #----------------------------------------------------------------------
    # The shape of the request
    #----------------------------------------------------------------------
    shapes = {"value": (1, 1), "growth": (1, 2), "compare": (2, 1), "rank": (2, 1)}
    if operation not in shapes:
        return Computed("out_of_scope", f"operation {operation!r} is not one of {sorted(shapes)}")
    if indicator not in table.units:
        raise ValueError(f"indicator {indicator!r} is not in the table: {sorted(table.units)}")
    want_countries, want_years = shapes[operation]

    years        = sorted(set(years)) if operation == "growth" else years
    right_number = len(countries) == 1 if want_countries == 1 else len(countries) >= want_countries
    if len(years) != want_years or not right_number:
        return Computed(
            "out_of_scope",
            f"{operation} needs {want_countries}{'+' if want_countries > 1 else ''} country and {want_years} year, "
            f"got countries {countries} and years {years}",
        )

    #----------------------------------------------------------------------
    # Read every cell; the first gap ends the request
    #----------------------------------------------------------------------
    facts: list[dict[str, Any]] = []
    for code in countries:
        for year in years:
            value = indicator_value(table, code, year, indicator)
            if value is None:
                cell = {"code": code, "year": year, "indicator": indicator}
                return Computed("no_data", missing_reason(table, code, year, indicator), missing=cell)
            facts.append({"code": code, "year": year, "indicator": indicator, "value": _rounded(indicator, value)})
    if operation == "rank":
        facts = sorted(facts, key=_value, reverse=True)
    return Computed("answered", "", facts, _result(operation, facts, numbers))


#==========================================================================
# Why a cell is missing
#==========================================================================
def missing_reason(table: FactsTable, code: str, year: int, indicator: str) -> str:
    """
    Name the missing cell and say what the table does have for that country.

    Input:
        . table         : the facts table
        . code          : country code
        . year          : the year asked
        . indicator     : the indicator asked
    Output:
        . reason        : e.g. "no gdp for VNM in 2024; the latest gdp year for VNM is 2023"
    """
    if code not in table.names:
        return f"country code {code} is not in the table"
    needs = ("gdp", "population") if indicator == "gdp_per_capita" else (indicator,)
    lasts = [table.last_year.get((code, name)) for name in needs]
    if None in lasts:
        return f"no {indicator} for {code} in any year of the table"
    return f"no {indicator} for {code} in {year}; the latest {indicator} year for {code} is {min(lasts)}"


#==========================================================================
# Round a fact the way it is shown
#==========================================================================
def _rounded(indicator: str, value: float) -> float:
    """
    Round a value for the contract: people to a whole number, money to 2 decimals.

    Input:
        . indicator     : the indicator
        . value         : the raw number
    Output:
        . value         : an int-valued float for population, else 2 decimals
    """
    return int(round(value)) if indicator == "population" else round(value, 2)


#==========================================================================
# The result of an operation
#==========================================================================
def _result(operation: str, facts: list[dict[str, Any]], numbers: FormatConfig) -> dict[str, Any]:
    """
    Build the one result of an operation from its facts.

    Input:
        . operation     : value, compare, growth or rank
        . facts         : one fact per country and year, in request order
        . numbers       : decimals of ratios and percentages
    Output:
        . result        : {"label", "value", "kind"}; kind is amount, ratio or percent
    """
    #----------------------------------------------------------------------
    # Value, growth and rank
    #----------------------------------------------------------------------
    first = facts[0]
    if operation == "value":
        return {"label": f"{first['indicator']} {first['code']} {first['year']}", "value": first["value"], "kind": "amount"}
    if operation == "growth":
        last   = facts[-1]
        change = (last["value"] / first["value"] - 1) * 100
        return {
            "label": f"growth {first['code']} {first['year']}-{last['year']} (%)",
            "value": round(change, numbers.percent_decimals),
            "kind": "percent",
        }
    if operation == "rank":
        return {"label": f"highest {first['code']}", "value": first["value"], "kind": "amount"}
    #----------------------------------------------------------------------
    # Compare: the first over the second, or the largest over the smallest
    #----------------------------------------------------------------------
    big, small = (facts[0], facts[1]) if len(facts) == 2 else (max(facts, key=_value), min(facts, key=_value))
    return {
        "label": f"ratio {big['code']}/{small['code']}",
        "value": round(big["value"] / small["value"], numbers.ratio_decimals),
        "kind": "ratio",
    }


#==========================================================================
# Sort key
#==========================================================================
def _value(fact: dict[str, Any]) -> float:
    """
    Return the value of a fact, for max and min.

    Input:
        . fact          : one fact
    Output:
        . value         : its number
    """
    return fact["value"]
