"""Node link: CODE and the vector database turn names into row keys."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    dataclasses                     import  asdict
from    typing                          import  Any

from    factdesk.config                  import  GraphConfig
from    factdesk.graph.state             import  FactState
from    factdesk.knowledge.linker        import  Candidate, Link, link_country, link_indicator
from    factdesk.knowledge.vector_store  import  VectorStore


#==========================================================================
# Link
#==========================================================================
def link_node(
    state: FactState,
    *,
    store: VectorStore,
    names: dict[str, str],
    country_index: dict[str, list[str]],
    indicator_index: dict[str, list[str]],
    graph: GraphConfig,
) -> dict[str, Any]:
    """
    Link the indicator and every country text. Declines (state["status"]) when
    the question is not a data question, the indicator is not in the data, no
    country is named, or a country matches nothing. Ambiguous names are left in
    "pending" for the clarify step; names the user already chose are linked
    with method "user".

    Input:
        . state         : holds operation, query and, after a clarify, resolved
        . store         : the entity collection
        . names         : code -> country name
        . country_index : normalised alias -> codes
        . indicator_index: normalised synonym -> indicator ids
        . graph         : linking thresholds
    Output:
        . update        : links, pending, unknown, countries, indicator; status and reason when declined
    """

    #----------------------------------------------------------------------
    # Questions that are not data questions end here
    #----------------------------------------------------------------------
    query = state["query"]
    if state["operation"] == "other":
        reason = state.get("reason") or "the question does not ask for the population or GDP of a country"
        return {"status": "out_of_scope", "reason": reason, "log": [f"link: out_of_scope ({reason})"]}
    indicator = link_indicator(query["indicator_text"], index=indicator_index, store=store, graph=graph)
    if indicator is None:
        reason = f"the data has no indicator like {query['indicator_text']!r}"
        return {"status": "out_of_scope", "reason": reason, "log": [f"link: out_of_scope ({reason})"]}
    if not query["country_texts"]:
        return {"status": "out_of_scope", "reason": "no country is named", "log": ["link: no country named"]}

    #----------------------------------------------------------------------
    # Link each country text; a name the user chose stays chosen
    #----------------------------------------------------------------------
    resolved = state.get("resolved", {})
    links: list[Link] = []
    for text in query["country_texts"]:
        if text in resolved:
            links.append(Link(text, "linked", resolved[text], "user", [Candidate(resolved[text], names[resolved[text]], 1.0)]))
        else:
            links.append(link_country(text, index=country_index, names=names, store=store, graph=graph))
    update: dict[str, Any] = {
        "indicator": indicator,
        "links": [asdict(link) for link in links],
        "countries": [link.code for link in links if link.status == "linked"],
        "pending": [
            {"text": link.text, "candidates": [asdict(item) for item in link.candidates]}
            for link in links
            if link.status == "ambiguous"
        ],
        "unknown": [link.text for link in links if link.status == "unknown"],
        "log": [f"link: {indicator} | " + ", ".join(f"{link.text}->{link.code or link.status}" for link in links)],
    }
    if update["unknown"]:
        update["status"] = "no_data"
        update["reason"] = f"no country in the table matches {update['unknown']}"
    return update
