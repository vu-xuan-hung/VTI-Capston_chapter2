"""T03: link_country: alias first, then a filtered vector search, then the ambiguity rule (run: pytest -k t03)."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    pathlib                         import  Path

import  pytest

from    factdesk.config                  import  GraphConfig
from    factdesk.knowledge.chunking      import  Chunk
from    factdesk.knowledge.linker        import  link_country
from    factdesk.knowledge.vector_store  import  VectorStore
from    helpers                          import  hash_store


#==========================================================================
# A small entity collection
#==========================================================================
@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    """
    Index four country chunks and one note chunk with hashing vectors.

    Input:
        . tmp_path      : temporary folder
    Output:
        . store         : the filled store
    """
    texts = {
        "KOR": "Korea, Rep. (KOR). Also called: South Korea, Republic of Korea.",
        "PRK": "Korea, Dem. People's Rep. (PRK). Also called: North Korea, DPRK.",
        "JPN": "Japan (JPN). Also called: Nippon.",
        "VNM": "Viet Nam (VNM). Also called: Vietnam.",
    }
    chunks = [Chunk(f"country-{code}#001", f"country-{code}", code, text, {"kind": "country", "code": code}) for code, text in texts.items()]
    chunks.append(Chunk("note-1#001", "note-1", "note", "Japan is a country with a large economy.", {"kind": "note"}))
    filled = hash_store(tmp_path, collection="link-kb")
    filled.index(chunks)
    return filled


NAMES = {"KOR": "Korea, Rep.", "PRK": "Korea, Dem. People's Rep.", "JPN": "Japan", "VNM": "Viet Nam"}
INDEX = {"korea": ["KOR", "PRK"], "s korea": ["KOR"], "japan": ["JPN"], "vietnam": ["VNM"]}


#==========================================================================
# Alias first
#==========================================================================
def test_t03_an_exact_alias_links_without_a_search(store: VectorStore) -> None:
    """
    Pin T03: a normalised alias with one owner is linked (score 1.0, method alias); an alias with two owners is ambiguous.

    Input:
        . store         : the filled store
    """
    one = link_country("S. Korea", index=INDEX, names=NAMES, store=store, graph=GraphConfig())
    assert (one.status, one.code, one.method) == ("linked", "KOR", "alias") and one.candidates[0].score == 1.0
    two = link_country("Korea", index=INDEX, names=NAMES, store=store, graph=GraphConfig())
    assert (two.status, two.code, two.method) == ("ambiguous", None, "alias")
    assert [candidate.code for candidate in two.candidates] == ["KOR", "PRK"]


#==========================================================================
# Then the vector search
#==========================================================================
def test_t03_a_clear_vector_match_is_linked(store: VectorStore) -> None:
    """
    Pin T03: no alias, one clearly best country chunk: linked, method vector.

    Input:
        . store         : the filled store
    """
    link = link_country("Nippon", index=INDEX, names=NAMES, store=store, graph=GraphConfig())
    assert (link.status, link.code, link.method) == ("linked", "JPN", "vector")
    assert link.candidates[0].code == "JPN" and 0.45 < link.candidates[0].score < 1.0


#==========================================================================
# Two close countries are ambiguous and both are candidates
#==========================================================================
def test_t03_two_close_countries_are_ambiguous_and_both_are_candidates(store: VectorStore) -> None:
    """
    Pin T03: the runner-up within graph.ambiguity_margin of the best makes the link ambiguous, with both as candidates.

    Input:
        . store         : the filled store
    """
    link = link_country("Rep. Korea people", index=INDEX, names=NAMES, store=store, graph=GraphConfig())
    assert link.status == "ambiguous" and link.code is None and link.method == "vector"
    assert sorted(candidate.code for candidate in link.candidates) == ["KOR", "PRK"]
    wide = link_country("Rep. Korea people", index=INDEX, names=NAMES, store=store, graph=GraphConfig(ambiguity_margin=0.001, country_confident_score=0.5))
    assert wide.status == "linked"


#==========================================================================
# A weak match is a guess
#==========================================================================
def test_t03_a_weak_vector_match_is_only_a_guess_the_user_must_confirm(store: VectorStore) -> None:
    """
    Pin T03: a best score between country_min_score and country_confident_score is ambiguous with
    that one candidate; above it the same text is linked.

    Input:
        . store         : the filled store
    """
    guess = link_country("Nippon", index=INDEX, names=NAMES, store=store, graph=GraphConfig(country_confident_score=0.9))
    assert (guess.status, guess.code) == ("ambiguous", None) and [candidate.code for candidate in guess.candidates] == ["JPN"]
    sure  = link_country("Nippon", index=INDEX, names=NAMES, store=store, graph=GraphConfig(country_confident_score=0.7))
    assert (sure.status, sure.code) == ("linked", "JPN")


#==========================================================================
# Nothing close is unknown and only country chunks are searched
#==========================================================================
def test_t03_nothing_close_is_unknown_and_only_country_chunks_are_searched(store: VectorStore) -> None:
    """
    Pin T03: a best score below graph.country_min_score is unknown; the note chunk is never a candidate (kind filter).

    Input:
        . store         : the filled store
    """
    nothing = link_country("Atlantis", index=INDEX, names=NAMES, store=store, graph=GraphConfig())
    assert (nothing.status, nothing.code) == ("unknown", None)
    note = link_country("Japan is a country with a large economy", index=INDEX, names=NAMES, store=store, graph=GraphConfig())
    assert all(candidate.code in NAMES for candidate in note.candidates)
