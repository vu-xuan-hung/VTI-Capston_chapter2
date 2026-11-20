"""The prepared data: countries, aggregates, facts, indexes, cases, and the prepare script itself."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  csv
import  importlib.util
import  json
import  re
from    dataclasses                     import  replace
from    pathlib                         import  Path

from    helpers                         import  ROOT, project_config


#==========================================================================
# Read the prepared files
#==========================================================================
def load_countries() -> list[dict]:
    """
    Read data/countries.json.

    Output:
        . countries     : [{code, name, aliases}]
    """
    return json.loads((ROOT / "data" / "countries.json").read_text(encoding="utf-8"))


#==========================================================================
# Aggregates are gone
#==========================================================================
def test_no_aggregate_survives_in_countries(tmp_path: Path) -> None:
    """
    Pin that no region, income group or World Bank group is a country, that
    the World and Kosovo are kept, and that every dropped code is really absent.

    Input:
        . tmp_path      : temporary folder for the config
    """
    config  = project_config(tmp_path)
    codes   = {item["code"] for item in load_countries()}
    words   = {"income", "&", "total", "members", "ida", "ibrd", "area", "dividend", "small", "excluding",
               "classification", "union", "oecd", "only", "blend", "indebted", "world"}
    for item in load_countries():
        if item["code"] != "WLD":
            assert not words & set(re.findall(r"[a-z]+|&", item["name"].lower())), item
    assert not codes & set(config.facts.aggregate_codes)
    assert {"WLD", "XKX", "VNM", "KOR", "PRK"} <= codes
    assert len(codes) == 217


#==========================================================================
# Facts
#==========================================================================
def test_facts_cover_1990_and_later_and_missing_cells_are_empty() -> None:
    """
    Pin the shape of facts.csv: years from 1990, population to 2025, GDP to 2023
    (2024 and 2025 have an empty GDP cell), North Korea has no GDP at all.
    """
    with (ROOT / "data" / "facts.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    years = {int(row["year"]) for row in rows}
    assert min(years) == 1990 and max(years) == 2025
    by_key = {(row["code"], int(row["year"])): row for row in rows}
    assert by_key[("VNM", 2024)]["population"] and by_key[("VNM", 2024)]["gdp_usd"] == ""
    assert by_key[("VNM", 2023)]["gdp_usd"]
    assert all(by_key[("PRK", year)]["gdp_usd"] == "" for year in range(1990, 2026))


#==========================================================================
# Aliases
#==========================================================================
def test_everyday_names_are_aliases_and_korea_and_congo_are_shared() -> None:
    """
    Pin the alias table: the hand-written names reach their code, and the names
    shared by two countries (Korea, Congo) are the ambiguous ones.
    """
    from    factdesk.knowledge.names        import  load_country_index, normalize

    index = load_country_index(ROOT / "data" / "countries.json")
    assert index[normalize("S. Korea")] == ["KOR"] and index[normalize("Vietnam")] == ["VNM"]
    assert index[normalize("Russia")] == ["RUS"] and index[normalize("Ivory Coast")] == ["CIV"]
    assert index[normalize("the United Kingdom")] == ["GBR"] and index[normalize("USA")] == ["USA"]
    shared = sorted(key for key, codes in index.items() if len(codes) > 1)
    assert shared == ["congo", "korea"]


#==========================================================================
# The prepare script is deterministic
#==========================================================================
def test_prepare_data_reproduces_the_shipped_files(tmp_path: Path) -> None:
    """
    Run scripts/prepare_data.py into a temporary folder and compare every
    byte with the files in data/.

    Input:
        . tmp_path      : temporary folder for the output
    """
    #----------------------------------------------------------------------
    # Load the script and run it into a temporary folder
    #----------------------------------------------------------------------
    spec   = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts" / "prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = project_config(tmp_path)
    paths  = replace(
        config.paths,
        facts_path      = tmp_path / "facts.csv",
        countries_path  = tmp_path / "countries.json",
        docs_dir        = tmp_path / "docs",
    )
    out    = replace(config, paths=paths)
    module.write_tables(out)
    # documents are built from the shipped authored files, so point them back at data/
    shipped = replace(paths, indicators_path=ROOT / "data" / "indicators.json", notes_path=ROOT / "data" / "notes.json")
    module.write_documents(replace(config, paths=shipped))
    #----------------------------------------------------------------------
    # Compare byte for byte
    #----------------------------------------------------------------------
    assert (tmp_path / "facts.csv").read_bytes() == (ROOT / "data" / "facts.csv").read_bytes()
    assert (tmp_path / "countries.json").read_bytes() == (ROOT / "data" / "countries.json").read_bytes()
    for name in ("countries", "indicators", "notes"):
        assert (tmp_path / "docs" / f"{name}.json").read_bytes() == (ROOT / "data" / "docs" / f"{name}.json").read_bytes()
