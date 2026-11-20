"""Flow reader on a real Langflow 1.12.6 export (Lesson 2.9's RAG flow)."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  copy
import  json

import  pytest
from    helpers                         import  FIXTURES

from    factdesk.integrations.flow_reader  import  node_id, read_flow, tweak


#==========================================================================
# Fixture flow
#==========================================================================
def _flow() -> dict:
    """
    Load the exported flow fixture.

    Output:
        . flow          : the flow JSON
    """
    return json.loads((FIXTURES / "flows" / "company_x_rag.json").read_text(encoding="utf-8"))


#==========================================================================
# Components, settings, edges
#==========================================================================
def test_real_export_lists_components_settings_and_edges() -> None:
    """
    Pin the summary of a real export: 7 components, 7 edges, key settings.
    """
    summary    = read_flow(_flow())
    components = {item.name: item for item in summary.components}
    assert summary.name == "Company X RAG"
    assert len(summary.components) == 7
    assert components["Chroma DB"].settings["collection_name"] == "company-x-kb"
    assert components["Chroma DB"].settings["number_of_results"] == 3
    assert components["OpenAI"].settings["openai_api_base"] == "http://127.0.0.1:8000/v1"
    assert "code" not in components["Parser"].settings
    assert "Chat Input.message -> Chroma DB.search_query" in summary.edges
    assert len(summary.edges) == 7
    assert summary.secrets == []


#==========================================================================
# A real key in a password field
#==========================================================================
def test_a_real_key_in_a_password_field_is_flagged_and_masked() -> None:
    """
    Pin the secrets check: the field is named and its value never shown.
    """
    flow = copy.deepcopy(_flow())
    for node in flow["data"]["nodes"]:
        if node["data"]["type"] == "OpenAIModel":
            node["data"]["node"]["template"]["api_key"]["value"] = "sk-" + "b" * 24
    summary = read_flow(flow)
    assert summary.secrets == ["OpenAI.api_key"]
    openai = next(item for item in summary.components if item.kind == "OpenAIModel")
    assert openai.settings["api_key"] == "***"


#==========================================================================
# Tweaks
#==========================================================================
def test_tweak_targets_the_component_by_type() -> None:
    """
    Pin the tweak shape {node id: {field: value}} and the clear error.
    """
    flow = _flow()
    assert tweak(flow, kind="Chroma", field="number_of_results", value=1) == {
        "ChromaVectorStoreComponent-JeBU7": {"number_of_results": 1}
    }
    with pytest.raises(ValueError, match="no component of type"):
        node_id(flow, "Nope")
