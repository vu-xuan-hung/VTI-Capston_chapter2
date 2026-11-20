"""Prepare the data: raw World Bank CSV files -> facts.csv, countries.json, knowledge base documents.

Usage: python scripts/prepare_data.py
Reads configs/app.toml ([facts]) and data/raw/; writes data/facts.csv, data/countries.json and data/docs/*.json.
Deterministic: the same raw files give the same output bytes. Stdlib only (the learners have not met pandas yet).
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  csv
import  json
import  sys
from    pathlib                         import  Path

from    factdesk.config                 import  Config, load_config


#==========================================================================
# The project folder
#==========================================================================
def root() -> Path:
    """
    Locate the project folder (the parent of scripts/).

    Output:
        . root          : the folder that holds configs/ and data/
    """
    return Path(__file__).resolve().parents[1]


#==========================================================================
# Read one raw World Bank file
#==========================================================================
def read_series(path: Path) -> tuple[dict[str, str], dict[tuple[str, int], str]]:
    """
    Read Country Name, Country Code, Year, Value rows without changing the numbers.

    Input:
        . path          : population.csv or gdp.csv
    Output:
        . names         : code -> the file's country name
        . cells         : (code, year) -> value text exactly as in the file
    """
    names: dict[str, str]                = {}
    cells: dict[tuple[str, int], str]    = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            names[row["Country Code"]]                          = row["Country Name"]
            cells[(row["Country Code"], int(row["Year"]))]      = row["Value"]
    return names, cells


#==========================================================================
# Which codes are countries
#==========================================================================
def choose_codes(names: dict[str, str], *, config: Config) -> list[str]:
    """
    Keep every code of population.csv that is not an aggregate; keep WLD.

    Input:
        . names         : code -> name from population.csv
        . config        : facts.aggregate_codes and facts.keep_aggregates
    Output:
        . codes         : sorted country codes plus the World
    """
    unknown = sorted(set(config.facts.aggregate_codes) - set(names) - {"FCS"})
    if unknown:
        raise ValueError(f"aggregate_codes not in population.csv: {unknown}")
    drop = set(config.facts.aggregate_codes) - set(config.facts.keep_aggregates)
    return sorted(code for code in names if code not in drop)


#==========================================================================
# Aliases of one country
#==========================================================================
def derive_aliases(code: str, name: str, other_names: list[str], *, hand: list[str]) -> list[str]:
    """
    Collect the names a person may write: the code, the World Bank name, the part
    before a comma or bracket, "St." spelled "Saint", the name in the other file,
    and the hand-written everyday names. Order is stable and duplicates are removed.

    Input:
        . code          : the World Bank code, e.g. KOR
        . name          : the population.csv name, e.g. "Korea, Rep."
        . other_names   : the name of the same code in gdp.csv (may differ)
        . hand          : everyday names from configs/app.toml
    Output:
        . aliases       : the code first, then the names
    """
    found = [code, name, *other_names, *hand]
    for text in [name, *other_names]:
        found += [text.split(",")[0], text.split("(")[0].strip()]
        found += [item.replace("St.", "Saint") for item in (text, text.split(",")[0]) if "St." in item]
    seen: list[str] = []
    for item in found:
        if item and item not in seen:
            seen.append(item)
    return seen


#==========================================================================
# Write the three tables
#==========================================================================
def write_tables(config: Config) -> tuple[int, int]:
    """
    Write data/facts.csv (years from facts.min_year) and data/countries.json.

    Input:
        . config        : paths and the facts section
    Output:
        . counts        : (number of countries, number of facts rows)
    """

    #----------------------------------------------------------------------
    # Read both raw files
    #----------------------------------------------------------------------
    pop_names, pop = read_series(config.paths.raw_dir / config.facts.population_csv)
    gdp_names, gdp = read_series(config.paths.raw_dir / config.facts.gdp_csv)
    codes          = choose_codes(pop_names, config=config)
    stray          = sorted(set(gdp_names) - set(pop_names) - set(config.facts.aggregate_codes))
    if stray:
        raise ValueError(f"gdp.csv has codes that are neither in population.csv nor aggregates: {stray}")

    #----------------------------------------------------------------------
    # facts.csv: one row per code and year, an empty cell when the source has none
    #----------------------------------------------------------------------
    years = sorted({year for _, year in pop if year >= config.facts.min_year})
    rows  = [
        [code, year, pop.get((code, year), ""), gdp.get((code, year), "")]
        for code in codes
        for year in years
        if (code, year) in pop or (code, year) in gdp
    ]
    with config.paths.facts_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["code", "year", "population", "gdp_usd"])
        writer.writerows(rows)

    #----------------------------------------------------------------------
    # countries.json with aliases
    #----------------------------------------------------------------------
    countries = [
        {
            "code": code,
            "name": pop_names[code],
            "aliases": derive_aliases(
                code,
                pop_names[code],
                [gdp_names[code]] if gdp_names.get(code, pop_names[code]) != pop_names[code] else [],
                hand = config.facts.aliases.get(code, []),
            ),
        }
        for code in codes
    ]
    config.paths.countries_path.write_text(json.dumps(countries, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    return len(countries), len(rows)


#==========================================================================
# Knowledge base documents
#==========================================================================
def write_documents(config: Config) -> int:
    """
    Write data/docs/countries.json, indicators.json and notes.json in the kit's
    document format: source_id, title, text and metadata keys (kind, code or indicator).

    Input:
        . config        : paths
    Output:
        . count         : number of documents written
    """

    #----------------------------------------------------------------------
    # One document per country, indicator and note
    #----------------------------------------------------------------------
    countries  = json.loads(config.paths.countries_path.read_text(encoding="utf-8"))
    indicators = json.loads(config.paths.indicators_path.read_text(encoding="utf-8"))
    notes      = json.loads(config.paths.notes_path.read_text(encoding="utf-8"))
    groups     = {
        "countries": [
            {
                "source_id": f"country-{item['code']}",
                "title": item["name"],
                "text": f"{item['name']} ({item['code']}). Also called: {', '.join(item['aliases'][1:])}.",
                "kind": "country",
                "code": item["code"],
            }
            for item in countries
        ],
        "indicators": [
            {
                "source_id": f"indicator-{item['indicator']}",
                "title": item["title"],
                "text": f"{item['title']}. {item['description']} Also called: {', '.join(item['synonyms'])}.",
                "kind": "indicator",
                "indicator": item["indicator"],
            }
            for item in indicators
        ],
        "notes": [
            {"source_id": f"note-{item['key']}", "title": item["title"], "text": item["text"], "kind": "note"}
            for item in notes
        ],
    }

    #----------------------------------------------------------------------
    # Write them
    #----------------------------------------------------------------------
    config.paths.docs_dir.mkdir(parents=True, exist_ok=True)
    for name, docs in groups.items():
        (config.paths.docs_dir / f"{name}.json").write_text(
            json.dumps(docs, indent=1, ensure_ascii=True) + "\n", encoding="utf-8"
        )
    return sum(len(docs) for docs in groups.values())


#==========================================================================
# Command line
#==========================================================================
def main() -> int:
    """
    Prepare every data file from data/raw and print the counts.

    Output:
        . code          : 0 when written
    """
    config = load_config(root() / "configs" / "app.toml", root=root())

    countries, rows = write_tables(config)
    documents       = write_documents(config)
    print(f"[OK] {countries} countries, {rows} facts rows, {documents} knowledge base documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
