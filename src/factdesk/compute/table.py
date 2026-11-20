"""The facts table: population and GDP per country and year, read from data/facts.csv.

Every number the application shows comes from here, through code. A cell the
source does not have is None, never zero and never an estimate.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  csv
import  json
from    dataclasses                     import  dataclass
from    pathlib                         import  Path


#==========================================================================
# The table
#==========================================================================
@dataclass(frozen=True)
class FactsTable:
    """
    Purpose: population and GDP by (code, year), the country names, the
    unit of each indicator and the last year each country has data for.
    """

    cells: dict[tuple[str, int], dict[str, float | None]]
    names: dict[str, str]
    units: dict[str, str]
    titles: dict[str, str]
    last_year: dict[tuple[str, str], int]       # (code, indicator) -> latest year with a value


#==========================================================================
# Load the table
#==========================================================================
def load_facts(facts_path: Path, countries_path: Path, indicators_path: Path) -> FactsTable:
    """
    Read facts.csv, countries.json and indicators.json.

    Input:
        . facts_path    : data/facts.csv (code, year, population, gdp_usd)
        . countries_path: data/countries.json
        . indicators_path: data/indicators.json
    Output:
        . table         : the facts table
    """

    #----------------------------------------------------------------------
    # Names, units and titles
    #----------------------------------------------------------------------
    names      = {item["code"]: item["name"] for item in json.loads(countries_path.read_text(encoding="utf-8"))}
    indicators = json.loads(indicators_path.read_text(encoding="utf-8"))
    units      = {item["indicator"]: item["unit"] for item in indicators}
    titles     = {item["indicator"]: item["title"] for item in indicators}

    #----------------------------------------------------------------------
    # Cells: an empty text in the CSV is a missing value
    #----------------------------------------------------------------------
    cells: dict[tuple[str, int], dict[str, float | None]] = {}
    last_year: dict[tuple[str, str], int]                 = {}
    with facts_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            code, year = row["code"], int(row["year"])
            values     = {
                "population": float(row["population"]) if row["population"] else None,
                "gdp": float(row["gdp_usd"]) if row["gdp_usd"] else None,
            }
            cells[(code, year)] = values
            for indicator, value in values.items():
                if value is not None:
                    last_year[(code, indicator)] = max(year, last_year.get((code, indicator), year))
    return FactsTable(cells=cells, names=names, units=units, titles=titles, last_year=last_year)


#==========================================================================
# One value
#==========================================================================
def indicator_value(table: FactsTable, code: str, year: int, indicator: str) -> float | None:
    """
    Return the value of one indicator for one country and year, or None.
    gdp_per_capita is gdp / population of the same cell.

    Input:
        . table         : the facts table
        . code          : World Bank country code, e.g. KOR
        . year          : calendar year
        . indicator     : population, gdp or gdp_per_capita
    Output:
        . value         : the number, or None when the source has none
    """
    cell = table.cells.get((code, year))
    if cell is None:
        return None
    if indicator == "gdp_per_capita":
        if cell["gdp"] is None or cell["population"] is None:
            return None
        return cell["gdp"] / cell["population"]
    return cell[indicator]
