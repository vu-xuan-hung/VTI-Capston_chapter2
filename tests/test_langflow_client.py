"""LangflowClient against a fake Langflow over real HTTP (Langflow is not installed here)."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    collections.abc                 import  Iterator

import  pytest
from    fake_langflow                   import  FLOW_ID, FLOW_NAME, FakeLangflow

from    factdesk.integrations.flow_reader      import  read_flow, tweak
from    factdesk.integrations.http_json        import  HttpJsonError, request_json
from    factdesk.integrations.langflow_client  import  LangflowClient


#==========================================================================
# Server fixture
#==========================================================================
@pytest.fixture
def langflow(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeLangflow]:
    """
    Start the fake Langflow with a test key in the environment.

    Input:
        . monkeypatch   : sets LANGFLOW_API_KEY for this test only
    Output:
        . server        : the running fake
    """
    monkeypatch.setenv("LANGFLOW_API_KEY", "test-value")
    server = FakeLangflow()
    yield server
    server.close()


#==========================================================================
# Find a flow
#==========================================================================
def test_find_flow_by_name_and_clear_error(langflow: FakeLangflow) -> None:
    """
    Pin lookup by name and the error that lists the existing flows.

    Input:
        . langflow      : the running fake
    """
    client = LangflowClient(langflow.url)
    assert client.find_flow(FLOW_NAME) == FLOW_ID
    with pytest.raises(ValueError, match="no flow named 'other'.*factdesk-rag"):
        client.find_flow("other")


#==========================================================================
# Run with a tweak and a session
#==========================================================================
def test_run_flow_sends_tweaks_and_session_and_returns_the_text(langflow: FakeLangflow) -> None:
    """
    Pin the run body (chat in/out, tweaks, session_id) and the stripped answer.

    Input:
        . langflow      : the running fake
    """
    client = LangflowClient(langflow.url)
    flow   = client.get_flow(FLOW_ID)
    tweaks = tweak(flow, kind="Chroma", field="number_of_results", value=1)
    answer = client.run_flow(FLOW_ID, "question", tweaks=tweaks, session_id="customer-1")
    sent   = langflow.requests[-1]
    assert answer == langflow.answer
    assert sent["tweaks"] == {"Chroma-b": {"number_of_results": 1}}
    assert (sent["session_id"], sent["input_type"], sent["output_type"]) == ("customer-1", "chat", "chat")
    client.run_flow(FLOW_ID, "again")
    assert "session_id" not in langflow.requests[-1] and "tweaks" not in langflow.requests[-1]


#==========================================================================
# Download a flow (gzipped)
#==========================================================================
def test_get_flow_reads_a_gzipped_answer(langflow: FakeLangflow) -> None:
    """
    Pin that a gzipped flow is decoded and can be read by flow_reader.

    Input:
        . langflow      : the running fake
    """
    summary = read_flow(LangflowClient(langflow.url).get_flow(FLOW_ID))
    assert [item.name for item in summary.components] == ["Chat Input", "Chroma DB", "OpenAI"]
    assert summary.components[1].settings == {"collection_name": "factdesk-kb", "number_of_results": 3}


#==========================================================================
# No key
#==========================================================================
def test_missing_key_is_one_clear_sentence(langflow: FakeLangflow, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Pin the error before any request when LANGFLOW_API_KEY is missing, and
    the server's own words on a 403.

    Input:
        . langflow      : the running fake
        . monkeypatch   : removes the key
    """
    monkeypatch.delenv("LANGFLOW_API_KEY")
    with pytest.raises(ValueError, match="LANGFLOW_API_KEY is not set"):
        LangflowClient(langflow.url).find_flow(FLOW_NAME)
    with pytest.raises(HttpJsonError, match="403.*no key"):
        request_json(f"{langflow.url}/api/v1/flows/{FLOW_ID}")
