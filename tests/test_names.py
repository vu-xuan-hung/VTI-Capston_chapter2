"""knowledge/names.py: normalising names and the alias index (no vector search here)."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  pytest

from    factdesk.knowledge.names        import  build_alias_index, normalize


#==========================================================================
# Normalise
#==========================================================================
@pytest.mark.parametrize(("text", "key"), [
    ("S. Korea", "s korea"),
    ("  S.  KOREA ", "s korea"),
    ("the Republic of Korea", "republic of korea"),
    ("Côte d'Ivoire", "cote d ivoire"),
    ("Korea, Rep.", "korea rep"),
])
def test_normalize_gives_one_key_per_spelling(text: str, key: str) -> None:
    """
    Pin normalize: lower case, no accents or punctuation, one space, no leading "the".

    Input:
        . text          : a name as written
        . key           : the expected key
    """
    assert normalize(text) == key


#==========================================================================
# Alias index
#==========================================================================
def test_alias_index_lists_every_owner_of_a_key() -> None:
    """
    Pin build_alias_index: a key owned by two codes lists both, sorted.
    """
    index = build_alias_index({"KOR": ["Korea", "South Korea"], "PRK": ["Korea", "North Korea"], "JPN": ["Japan"]})
    assert index["korea"] == ["KOR", "PRK"] and index["south korea"] == ["KOR"] and index["japan"] == ["JPN"]
