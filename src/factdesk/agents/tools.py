"""The three tools of the free agent (v1): get_value, compare, growth.

Thin wrappers over compute(): the same arithmetic as the graph, but here the
MODEL picks the tool and its arguments. A tool never raises to the model: a
problem comes back as {"error": "..."} so the agent can read it and try again.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                          import  annotations

import  json

from    langchain.tools                     import  tool
from    langchain_core.tools                import  BaseTool

from    factdesk.compute.operations         import  compute
from    factdesk.compute.table              import  FactsTable
from    factdesk.config                     import  FormatConfig


#==========================================================================
# Shared by the three tools
#==========================================================================
def run_tool(
    table: FactsTable,
    numbers: FormatConfig,
    *,
    operation: str,
    indicator: str,
    countries: list[str],
    years: list[int],
) -> str:
    """
    Run one operation for a tool and write the outcome as JSON text: the
    facts and the result on success, {"error": reason} on any problem.

    Input:
        . table         : the facts table
        . numbers       : rounding of results
        . operation     : value, compare or growth
        . indicator     : as the model wrote it; "GDP per capita" is read as gdp_per_capita
        . countries     : codes as the model wrote them; case is ignored
        . years         : calendar years
    Output:
        . text          : {"operation", "facts", "result"} or {"error"}
    """
    #----------------------------------------------------------------------
    # Read what the model wrote
    #----------------------------------------------------------------------
    indicator = indicator.strip().lower().replace(" ", "_")
    codes = [code.strip().upper() for code in countries]
    if indicator not in table.units:
        return json.dumps(
            {"error": f"unknown indicator {indicator!r}; use one of {sorted(table.units)}"}
        )
    #----------------------------------------------------------------------
    # Run it; a problem comes back as JSON
    #----------------------------------------------------------------------
    computed = compute(
        table,
        operation=operation,
        indicator=indicator,
        countries=codes,
        years=years,
        numbers=numbers,
    )
    if computed.status != "answered":
        hint = (
            " Use ISO alpha-3 codes such as VNM, JPN, KOR."
            if "not in the table" in computed.reason
            else ""
        )
        return json.dumps({"error": computed.reason + hint})
    result = {"label": computed.result["label"], "value": computed.result["value"]}
    return json.dumps({"operation": operation, "facts": computed.facts, "result": result})


#==========================================================================
# The tool set
#==========================================================================
def make_tools(table: FactsTable, numbers: FormatConfig) -> list[BaseTool]:
    """
    Build the three tools over one facts table. get_value and growth are
    yours to write (T02); compare is given as the model answer.

    Input:
        . table         : the facts table
        . numbers       : rounding of results
    Output:
        . tools         : [get_value, compare, growth]
    """
    #----------------------------------------------------------------------
    # The tools you write: get_value and growth
    #----------------------------------------------------------------------

    @tool
    def get_value(country: str, indicator: str, year: int) -> str:
        """
        Get one indicator of one country in one year.

        Input:
            . country       : ISO alpha-3 code of the country
            . indicator     : population, gdp or gdp_per_capita
            . year          : calendar year
        Output:
            . json          : the fact, or {"error": ...}
        """
        return run_tool(
            table,
            numbers,
            operation="value",
            indicator=indicator,
            countries=[country],
            years=[year],
        )

    @tool
    def growth(country: str, indicator: str, year_from: int, year_to: int) -> str:
        """
        Get the growth of one indicator of one country between two years.

        Input:
            . country       : ISO alpha-3 code of the country
            . indicator     : population, gdp or gdp_per_capita
            . year_from     : calendar year to start from
            . year_to       : calendar year to end at
        Output:
            . json          : the percent growth, or {"error": ...}
        """
        return run_tool(
            table,
            numbers,
            operation="growth",
            indicator=indicator,
            countries=[country],
            years=[year_from, year_to],
        )

    #----------------------------------------------------------------------
    # Given: the compare tool, the model answer for the style
    #----------------------------------------------------------------------
    @tool
    def compare(country_a: str, country_b: str, indicator: str, year: int) -> str:
        """
        Compare one indicator of two countries in one year: both values and the ratio a / b.

        Input:
            . country_a     : ISO alpha-3 code of the first country
            . country_b     : ISO alpha-3 code of the second country
            . indicator     : population, gdp or gdp_per_capita
            . year          : calendar year
        Output:
            . json          : both facts and the ratio, or {"error": ...}
        """
        return run_tool(
            table,
            numbers,
            operation="compare",
            indicator=indicator,
            countries=[country_a, country_b],
            years=[year],
        )

    return [get_value, compare, growth]
