"""T02: the get_value and growth tools of the free agent (run: pytest -k t02)."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
from    pathlib                         import  Path

import  pytest

from    factdesk.agents.tools           import  make_tools
from    factdesk.compute.table          import  load_facts
from    factdesk.config                 import  FormatConfig
from    helpers                         import  ROOT, indexed_desk


#==========================================================================
# The tools over the real table
#==========================================================================
@pytest.fixture(scope="module")
def tools() -> dict:
    """
    Build the three tools over the shipped table and index them by name.

    Output:
        . tools         : name -> tool
    """
    table = load_facts(ROOT / "data" / "facts.csv", ROOT / "data" / "countries.json", ROOT / "data" / "indicators.json")
    return {tool.name: tool for tool in make_tools(table, FormatConfig())}


#==========================================================================
# get_value
#==========================================================================
def test_t02_get_value_returns_the_fact_as_json(tools: dict) -> None:
    """
    Pin T02: the tool is named get_value, typed (country, indicator, year), and
    returns JSON with the fact and a result; a sloppy indicator spelling is read.

    Input:
        . tools         : the three tools
    """
    assert sorted(tools) == ["compare", "get_value", "growth"]
    assert sorted(tools["get_value"].args) == ["country", "indicator", "year"]
    assert tools["get_value"].description.strip()
    out = json.loads(tools["get_value"].invoke({"country": "vnm", "indicator": "population", "year": 2015}))
    assert out["facts"] == [{"code": "VNM", "year": 2015, "indicator": "population", "value": 92823254}]
    assert out["operation"] == "value" and out["result"]["value"] == 92823254
    spaced = json.loads(tools["get_value"].invoke({"country": "JPN", "indicator": "GDP per capita", "year": 2020}))
    assert spaced["facts"][0]["indicator"] == "gdp_per_capita"


#==========================================================================
# Get value reports a problem as json not as a crash
#==========================================================================
def test_t02_get_value_reports_a_problem_as_json_not_as_a_crash(tools: dict) -> None:
    """
    Pin T02: an unknown country name, an unknown indicator and a year without data come back as
    {"error": ...}, so the agent can read the error and try again.

    Input:
        . tools         : the three tools
    """
    name = json.loads(tools["get_value"].invoke({"country": "Vietnam", "indicator": "population", "year": 2015}))
    assert "ISO alpha-3" in name["error"]
    kind = json.loads(tools["get_value"].invoke({"country": "VNM", "indicator": "inflation", "year": 2015}))
    assert "unknown indicator" in kind["error"]
    gap  = json.loads(tools["get_value"].invoke({"country": "VNM", "indicator": "gdp", "year": 2024}))
    assert "latest gdp year for VNM is 2023" in gap["error"]


#==========================================================================
# growth
#==========================================================================
def test_t02_growth_returns_both_facts_and_the_percent(tools: dict) -> None:
    """
    Pin T02: growth(country, indicator, year_from, year_to) returns both years and the percent change.

    Input:
        . tools         : the three tools
    """
    assert sorted(tools["growth"].args) == ["country", "indicator", "year_from", "year_to"]
    out = json.loads(tools["growth"].invoke({"country": "POL", "indicator": "population", "year_from": 2000, "year_to": 2020}))
    assert [fact["year"] for fact in out["facts"]] == [2000, 2020] and out["result"]["value"] == -1.94
    wrong = json.loads(tools["growth"].invoke({"country": "POL", "indicator": "population", "year_from": 2020, "year_to": 2020}))
    assert "error" in wrong


#==========================================================================
# The agent that uses them (v1) ends in a contract
#==========================================================================
def test_t02_free_agent_uses_the_tools_and_its_contract_is_unverified(tmp_path: Path) -> None:
    """
    Pin v1 on the fake model: the agent calls a tool, the contract holds the tool's facts, and
    verified stays False because nothing checks the agent.

    Input:
        . tmp_path      : temporary folder
    """
    desk   = indexed_desk(tmp_path)
    answer = desk.ask_agent("What was Vietnam's population in 2015?")
    assert answer.status == "answered" and answer.verified is False
    assert answer.countries == ["VNM"] and answer.years == [2015] and answer.facts[0].value == 92823254
    assert answer.reason == "v1: not checked by code"
