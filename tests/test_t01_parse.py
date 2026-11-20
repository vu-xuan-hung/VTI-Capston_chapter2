"""T01: the Query schema and the parse chain (run: pytest -k t01)."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  pytest

from    factdesk.llm.chat_source        import  ChatSource
from    factdesk.config                 import  ModelConfig, ServerConfig
from    factdesk.llm.query              import  Query, build_parse_chain


#==========================================================================
# The chain returns a Query
#==========================================================================
def test_t01_parse_chain_returns_a_query_of_the_question() -> None:
    """
    Pin T01: prompt | structured model gives a Query whose fields are the
    operation, the quantity, the country names and the years of the question.
    """
    chat  = ChatSource(ModelConfig(backend="fake"), ServerConfig()).get()
    query = build_parse_chain(chat).invoke({"question": "Compare the GDP per person of Japan and South Korea in 2020."})
    assert isinstance(query, Query)
    assert (query.operation, query.indicator_text, query.country_texts, query.years) == (
        "compare", "GDP per person", ["Japan", "South Korea"], [2020])


#==========================================================================
# The schema is small and flat
#==========================================================================
def test_t01_query_has_four_fields_and_an_operation_enum() -> None:
    """
    Pin T01: four fields, operation limited to five values, years as integers,
    and an invalid operation is refused (not repaired).
    """
    chat = ChatSource(ModelConfig(backend="fake"), ServerConfig()).get()
    build_parse_chain(chat)
    assert list(Query.model_fields) == ["operation", "indicator_text", "country_texts", "years"]
    with pytest.raises(ValueError):
        Query(operation="sum", indicator_text="gdp", country_texts=[], years=[])
    assert Query(operation="rank", indicator_text="gdp", country_texts=["A"], years=["2020"]).years == [2020]
