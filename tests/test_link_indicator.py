"""link_indicator: the quantity of a question to an indicator id (synonyms first, then the nearest indicator chunk)."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    pathlib                         import  Path

import  pytest

from    factdesk.config                 import  GraphConfig
from    factdesk.knowledge.linker       import  link_indicator
from    factdesk.knowledge.names        import  load_indicator_index
from    helpers                         import  ROOT, indexed_desk


#==========================================================================
# Synonyms and the nearest chunk
#==========================================================================
@pytest.mark.parametrize(("text", "indicator"), [
    ("GDP per person", "gdp_per_capita"),
    ("GDP per head", "gdp_per_capita"),
    ("economy size", "gdp"),
    ("the economy", "gdp"),
    ("Population", "population"),
    ("unemployment rate", None),
])
def test_indicator_is_linked_by_synonym_or_not_at_all(tmp_path: Path, text: str, indicator: str | None) -> None:
    """
    Pin link_indicator: a synonym of the data is linked, a quantity the data does not have is None.

    Input:
        . tmp_path      : temporary folder
        . text          : the quantity as written in a question
        . indicator     : the expected indicator id, or None
    """
    store = indexed_desk(tmp_path).store
    index = load_indicator_index(ROOT / "data" / "indicators.json")
    assert link_indicator(text, index=index, store=store, graph=GraphConfig(indicator_min_score=0.6)) == indicator
