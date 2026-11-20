"""compute/: the table, the four operations, the formatting, and the dev and hold-out cases as an independent check."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
from    pathlib                         import  Path

import  pytest

from    factdesk.compute.formatting     import  format_amount, format_result, template_answer
from    factdesk.compute.operations     import  compute
from    factdesk.compute.table          import  FactsTable, indicator_value, load_facts
from    factdesk.config                 import  FormatConfig
from    helpers                         import  ROOT


#==========================================================================
# The real table
#==========================================================================
@pytest.fixture(scope="module")
def table() -> FactsTable:
    """
    Load the shipped facts table once for this module.

    Output:
        . table         : data/facts.csv, countries.json, indicators.json
    """
    return load_facts(ROOT / "data" / "facts.csv", ROOT / "data" / "countries.json", ROOT / "data" / "indicators.json")


#==========================================================================
# Values and gaps
#==========================================================================
def test_values_come_from_the_csv_and_gaps_are_none(table: FactsTable) -> None:
    """
    Pin indicator_value: a real cell, the derived GDP per person, and the gaps
    (GDP after 2023, North Korea's GDP, a year before 1990).

    Input:
        . table         : the facts table
    """
    assert indicator_value(table, "VNM", 2015, "population") == 92823254
    gdp, pop = indicator_value(table, "JPN", 2020, "gdp"), indicator_value(table, "JPN", 2020, "population")
    assert indicator_value(table, "JPN", 2020, "gdp_per_capita") == pytest.approx(gdp / pop)
    assert indicator_value(table, "VNM", 2024, "gdp") is None
    assert indicator_value(table, "VNM", 2024, "gdp_per_capita") is None
    assert indicator_value(table, "PRK", 2015, "gdp") is None
    assert indicator_value(table, "DEU", 1985, "population") is None


#==========================================================================
# The four operations
#==========================================================================
def test_the_four_operations(table: FactsTable) -> None:
    """
    Pin value, compare (ratio first/second), growth (percent) and rank
    (sorted high to low, the result is the highest).

    Input:
        . table         : the facts table
    """
    numbers = FormatConfig()
    value   = compute(table, operation="value", indicator="population", countries=["VNM"], years=[2015], numbers=numbers)
    assert value.status == "answered" and value.facts[0]["value"] == 92823254 and value.result["value"] == 92823254
    ratio   = compute(table, operation="compare", indicator="gdp_per_capita", countries=["JPN", "KOR"], years=[2020], numbers=numbers)
    assert ratio.result["label"] == "ratio JPN/KOR" and ratio.result["value"] == 1.26
    growth  = compute(table, operation="growth", indicator="population", countries=["POL"], years=[2020, 2000], numbers=numbers)
    assert growth.result["value"] == -1.94 and [fact["year"] for fact in growth.facts] == [2000, 2020]
    rank    = compute(table, operation="rank", indicator="gdp", countries=["ESP", "DEU", "FRA"], years=[2022], numbers=numbers)
    assert [fact["code"] for fact in rank.facts][0] == "DEU" and rank.result["label"] == "highest DEU"


#==========================================================================
# Honest hand-over
#==========================================================================
def test_missing_cells_and_wrong_shapes_name_what_is_wrong(table: FactsTable) -> None:
    """
    Pin that a missing cell is no_data with the cell named, an unknown code is
    no_data, and a request of the wrong shape is out_of_scope; nothing is estimated.

    Input:
        . table         : the facts table
    """
    numbers = FormatConfig()
    gap     = compute(table, operation="value", indicator="gdp", countries=["VNM"], years=[2024], numbers=numbers)
    assert gap.status == "no_data" and gap.facts == [] and gap.result is None
    assert gap.missing == {"code": "VNM", "year": 2024, "indicator": "gdp"} and "latest gdp year for VNM is 2023" in gap.reason
    unknown = compute(table, operation="value", indicator="gdp", countries=["XXX"], years=[2020], numbers=numbers)
    assert unknown.status == "no_data" and "XXX" in unknown.reason
    shape   = compute(table, operation="growth", indicator="gdp", countries=["VNM"], years=[2020], numbers=numbers)
    assert shape.status == "out_of_scope" and "growth needs" in shape.reason
    with pytest.raises(ValueError, match="not in the table"):
        compute(table, operation="value", indicator="inflation", countries=["VNM"], years=[2020], numbers=numbers)


#==========================================================================
# Formatting
#==========================================================================
def test_numbers_are_written_by_kind() -> None:
    """
    Pin the number formats: people with commas, GDP in trillions/billions/millions,
    GDP per person with 2 decimals, ratios and percentages.
    """
    numbers = FormatConfig()
    assert format_amount("population", 92823254, numbers) == "92,823,254"
    assert format_amount("gdp", 4256410760723.75, numbers) == "$4.26 trillion"
    assert format_amount("gdp", 410324028883.33, numbers) == "$410.32 billion"
    assert format_amount("gdp", 764804469.27, numbers) == "$764.80 million"
    assert format_amount("gdp_per_capita", 39986.934, numbers) == "$39,986.93"
    assert format_result({"kind": "ratio", "value": 1.26}, "gdp", numbers) == "1.26"
    assert format_result({"kind": "percent", "value": -1.94}, "gdp", numbers) == "-1.94%"


#==========================================================================
# Template answer holds exactly the table numbers
#==========================================================================
def test_template_answer_holds_exactly_the_table_numbers(table: FactsTable) -> None:
    """
    Pin the answer written by code: every fact once, the result once, named.

    Input:
        . table         : the facts table
    """
    numbers  = FormatConfig()
    computed = compute(table, operation="compare", indicator="gdp_per_capita", countries=["JPN", "KOR"], years=[2020], numbers=numbers)
    text     = template_answer(computed.facts, computed.result, names=table.names, titles=table.titles, numbers=numbers)
    assert text.startswith("Japan, 2020, GDP per person: $") and "Korea, Rep., 2020" in text and "(ratio JPN/KOR): 1.26." in text


#==========================================================================
# The cases are an independent check
#==========================================================================
@pytest.mark.parametrize("path", [ROOT / "data" / "dev_cases.json", ROOT / "holdout" / "holdout_cases.json"])
def test_cases_were_computed_independently_and_match_compute(path: Path, table: FactsTable) -> None:
    """
    Pin that the expected numbers of the cases (written by scripts/make_cases.py with
    plain arithmetic) equal what compute() returns, for every answered case.

    Input:
        . path          : a cases file; the hold-out file exists only in the solution repo
        . table         : the facts table
    """
    if not path.exists():
        pytest.skip(f"{path.name} is not part of this repo")
    for case in json.loads(path.read_text(encoding="utf-8")):
        want = case["outputs"]
        if want["status"] != "answered":
            continue
        got = compute(
            table,
            operation   = want["operation"],
            indicator   = want["indicator"],
            countries   = want["countries"],
            years       = want["years"],
            numbers     = FormatConfig(),
        )
        assert got.status == "answered" and got.result["value"] == pytest.approx(want["result_value"], rel=1e-4), case["id"]
