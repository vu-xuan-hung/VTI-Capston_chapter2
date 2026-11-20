"""Link the names in a question to row keys: country codes and indicator ids.

The vector database is used here in a new way: not to find passages but to
decide which row a name means. Alias match first (exact, cheap, certain),
then a vector search filtered to country chunks, then the ambiguity rule.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                          import  annotations

from    dataclasses                         import  dataclass, field

from    factdesk.config                     import  GraphConfig
from    factdesk.knowledge.names            import  normalize
from    factdesk.knowledge.vector_store     import  VectorStore


#==========================================================================
# One candidate row
#==========================================================================
@dataclass(frozen=True)
class Candidate:
    """
    Purpose: one row a name may mean: its code, its name and the score of the
    match (1.0 for an exact alias, the cosine similarity for a vector hit).
    """

    code: str
    name: str
    score: float


#==========================================================================
# The result of linking one name
#==========================================================================
@dataclass(frozen=True)
class Link:
    """
    Purpose: what linking one name decided: "linked" (code set), "ambiguous"
    (several candidates, ask the user) or "unknown" (nothing close enough),
    and by which method.
    """

    text: str
    status: str
    code: str | None
    method: str
    candidates: list[Candidate] = field(default_factory=list)


#==========================================================================
# Link one country name
#==========================================================================
def link_country(
    text: str,
    *,
    index: dict[str, list[str]],
    names: dict[str, str],
    store: VectorStore,
    graph: GraphConfig,
) -> Link:
    """
    Link one country name from a question to a country code.

    1. Exact alias: the normalised text is in the index. One code = linked
       (score 1.0, method "alias"); several codes = ambiguous.
    2. Otherwise search the vector store, only chunks with kind = country,
       k = graph.link_k. One country can have several chunks: keep its best.
    3. Best score below graph.country_min_score (or no hit) = unknown.
       The second-best country within graph.ambiguity_margin of the best =
       ambiguous, candidates = every country within the margin. A best score
       below graph.country_confident_score is only a guess ("Taiwan" -> Guam):
       ambiguous too, with the best candidate, so the user confirms it.
       Otherwise linked.

    Input:
        . text          : the name as the question wrote it, e.g. "S. Korea"
        . index         : normalised alias -> codes (names.load_country_index)
        . names         : code -> country name
        . store         : the entity collection
        . graph         : link_k, country_min_score, country_confident_score, ambiguity_margin
    Output:
        . link          : status linked, ambiguous or unknown
    """
    alias = index.get(normalize(text), [])
    #----------------------------------------------------------------------
    # Search the collection when the alias table has no entry
    #----------------------------------------------------------------------
    if not alias:
        hits = store.search(text, k=graph.link_k, where={"kind": "country"})
        best_score = 0.0
        best_code = None
        candidates = {}
        for hit in hits:
            code = hit.metadata["code"]
            score = hit.score
            if score > best_score:
                best_score = score
                best_code = code
            previous = candidates.get(code)
            if previous is None or score > previous.score:
                candidates[code] = Candidate(code=code, name=names[code], score=score)

        if best_score < graph.country_min_score:
            return Link(
                text=text,
                status="unknown",
                code=None,
                method="vector",
                candidates=list(candidates.values()),
            )
        ambiguous_candidates = [
            c for c in candidates.values() if c.score >= best_score - graph.ambiguity_margin
        ]
        if len(ambiguous_candidates) > 1 or best_score < graph.country_confident_score:
            return Link(
                text=text,
                status="ambiguous",
                code=None,
                method="vector",
                candidates=ambiguous_candidates,
            )
        else:
            return Link(
                text=text,
                status="linked",
                code=best_code,
                method="vector",
                candidates=list(candidates.values()),
            )

    if len(alias) == 1:
        return Link(
            text=text,
            status="linked",
            code=alias[0],
            method="alias",
            candidates=[Candidate(code=alias[0], name=names[alias[0]], score=1.0)],
        )
    else:
        return Link(
            text=text,
            status="ambiguous",
            code=None,
            method="alias",
            candidates=[Candidate(code=code, name=names[code], score=1.0) for code in alias],
        )


#==========================================================================
# Link the indicator name
#==========================================================================
def link_indicator(
    text: str,
    *,
    index: dict[str, list[str]],
    store: VectorStore,
    graph: GraphConfig,
) -> str | None:
    """
    Link the quantity of a question ("GDP per person", "economy size") to an
    indicator id: exact synonym first, then the nearest indicator chunk
    above graph.indicator_min_score; None means the data has no such indicator.

    Input:
        . text          : the quantity as the question wrote it
        . index         : normalised synonym -> indicator ids
        . store         : the entity collection
        . graph         : indicator_min_score
    Output:
        . indicator     : population, gdp, gdp_per_capita, or None
    """
    ids = index.get(normalize(text), [])
    if len(ids) == 1:
        return ids[0]
    hits = store.search(text, k=1, where={"kind": "indicator"})
    if hits and hits[0].score >= graph.indicator_min_score:
        return hits[0].metadata["indicator"]
    return None
