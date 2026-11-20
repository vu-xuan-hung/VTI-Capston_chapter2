"""Names to codes: the plain-text side of linking (the vector side is in linker.py).

normalize() makes "S. Korea", "s korea" and "  S.  KOREA " the same key;
the alias index maps a key to every code that owns it. A key owned by two
countries ("korea", "congo") is what makes a question ambiguous.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
import  re
import  unicodedata
from    pathlib                         import  Path


#==========================================================================
# Normalise a name
#==========================================================================
def normalize(text: str) -> str:
    """
    Lower-case, drop accents and punctuation, collapse spaces, drop a leading "the".

    Input:
        . text          : a name as written, e.g. "S. Korea" or "the Republic of Korea"
    Output:
        . key           : e.g. "s korea", "republic of korea"
    """
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    key   = re.sub(r"[^a-z0-9]+", " ", plain.lower()).strip()
    return key[4:] if key.startswith("the ") else key


#==========================================================================
# Alias index
#==========================================================================
def build_alias_index(entries: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Map every normalised alias to the sorted codes that own it.

    Input:
        . entries       : code (or indicator id) -> its names
    Output:
        . index         : key -> codes; more than one code means the name is shared
    """
    index: dict[str, set[str]] = {}
    for code, aliases in entries.items():
        for alias in aliases:
            index.setdefault(normalize(alias), set()).add(code)
    return {key: sorted(codes) for key, codes in index.items()}


#==========================================================================
# Load the two indexes from the prepared data
#==========================================================================
def load_country_index(countries_path: Path) -> dict[str, list[str]]:
    """
    Build the alias index of countries from data/countries.json.

    Input:
        . countries_path: data/countries.json
    Output:
        . index         : normalised alias -> codes
    """
    countries = json.loads(countries_path.read_text(encoding="utf-8"))
    return build_alias_index({item["code"]: item["aliases"] for item in countries})


#==========================================================================
# The indicator index
#==========================================================================
def load_indicator_index(indicators_path: Path) -> dict[str, list[str]]:
    """
    Build the alias index of indicators from data/indicators.json (id, title, synonyms).

    Input:
        . indicators_path: data/indicators.json
    Output:
        . index         : normalised synonym -> indicator ids
    """
    indicators = json.loads(indicators_path.read_text(encoding="utf-8"))
    return build_alias_index({
        item["indicator"]: [item["indicator"].replace("_", " "), item["title"], *item["synonyms"]]
        for item in indicators
    })
