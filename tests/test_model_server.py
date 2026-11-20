"""Model server over real HTTP with the fake backend: chat, tools, stream, embeddings."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  base64
import  json
import  threading
import  urllib.request
from    collections.abc                 import  Iterator

import  numpy                           as      np
import  pytest
from    langchain.agents                import  create_agent
from    langchain.tools                 import  tool
from    langchain_core.messages         import  HumanMessage, ToolMessage
from    pydantic                        import  BaseModel

from    factdesk.integrations.http_json          import  HttpJsonError, request_json
from    factdesk.integrations.local_models       import  fake_backend
from    factdesk.llm.fake_chat_model             import  FakeRules
from    factdesk.integrations.model_server       import  make_server
from    factdesk.integrations.server_embeddings  import  ServerEmbeddings
from    factdesk.knowledge.hash_embeddings       import  HashEmbeddings
from    factdesk.llm.server_chat                 import  ServerChat


#==========================================================================
# Test schema and tool
#==========================================================================
class Ticket(BaseModel):
    """
    Purpose: a structured answer that must travel through the server as a tool call.
    """

    topic: str
    urgent: bool


#==========================================================================
# Test tool
#==========================================================================
@tool
def weather(city: str) -> str:
    """
    Tell the weather of a city.

    Input:
        . city          : a city name
    Output:
        . weather       : one sentence
    """
    return f"Sunny in {city}."


#==========================================================================
# Server fixture
#==========================================================================
@pytest.fixture
def base_url() -> Iterator[str]:
    """
    Serve the fake backend on a free port for one test.

    Output:
        . base_url      : e.g. http://127.0.0.1:PORT/v1
    """
    server = make_server(fake_backend(rules=FakeRules()), host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


#==========================================================================
# /v1/models
#==========================================================================
def test_models_lists_chat_and_embeddings(base_url: str) -> None:
    """
    Pin the model list Langflow reads.

    Input:
        . base_url      : the running fake server
    """
    names = [item["id"] for item in request_json(f"{base_url}/models")["data"]]
    assert names == ["fake-chat", "hash-64"]


#==========================================================================
# Plain chat
#==========================================================================
def test_server_chat_returns_text_and_usage(base_url: str) -> None:
    """
    Pin a plain chat round trip with token counts.

    Input:
        . base_url      : the running fake server
    """
    answer = ServerChat(base_url=base_url, model_name="fake-chat").invoke([HumanMessage("hello server")])
    assert answer.content == "[FAKE MODEL ANSWER] hello server"
    assert answer.usage_metadata["total_tokens"] > 0


#==========================================================================
# Structured output through the server
#==========================================================================
def test_structured_output_travels_as_a_forced_tool_call(base_url: str) -> None:
    """
    Pin with_structured_output over HTTP: tools + tool_choice out, tool_calls back.

    Input:
        . base_url      : the running fake server
    """
    chat   = ServerChat(base_url=base_url, model_name="fake-chat")
    answer = chat.with_structured_output(Ticket).invoke("The printer is on fire")
    assert answer == Ticket(topic="[FAKE]", urgent=False)


#==========================================================================
# create_agent through the server
#==========================================================================
def test_agent_tool_loop_runs_through_the_server(base_url: str) -> None:
    """
    Pin a create_agent loop over HTTP: the tool call, the tool message, the answer.

    Input:
        . base_url      : the running fake server
    """
    agent  = create_agent(ServerChat(base_url=base_url, model_name="fake-chat"), tools=[weather])
    result = agent.invoke({"messages": [HumanMessage("Weather in Hue?")]})
    tools  = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.content for message in tools] == ["Sunny in [FAKE]."]
    assert result["messages"][-1].content.startswith("[FAKE MODEL ANSWER]")


#==========================================================================
# Embeddings, floats and base64
#==========================================================================
def test_embeddings_match_local_vectors_in_both_encodings(base_url: str) -> None:
    """
    Pin that the server's vectors equal the local ones, as floats and as base64.

    Input:
        . base_url      : the running fake server
    """
    local  = HashEmbeddings().embed_query("refund policy")
    remote = ServerEmbeddings(base_url, "hash-64").embed_query("refund policy")
    body   = {"model": "hash-64", "input": "refund policy", "encoding_format": "base64"}
    packed = request_json(f"{base_url}/embeddings", body)["data"][0]["embedding"]
    floats = np.frombuffer(base64.b64decode(packed), dtype="<f4")
    assert remote == pytest.approx(local)
    assert floats.tolist() == pytest.approx(local, abs=1e-6)


#==========================================================================
# stream=true
#==========================================================================
def test_stream_sends_tool_calls_and_done(base_url: str) -> None:
    """
    Pin the Server-Sent Events answer, with a tool call delta.

    Input:
        . base_url      : the running fake server
    """

    #----------------------------------------------------------------------
    # A streamed request with one tool
    #----------------------------------------------------------------------
    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "tools": [{"type": "function", "function": {"name": "weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}}}}],
        "stream": True,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data    = json.dumps(body).encode(),
        headers = {"Content-Type": "application/json"},
    )
    #----------------------------------------------------------------------
    # Read the events
    #----------------------------------------------------------------------
    with urllib.request.urlopen(request) as answer:
        events = [line[6:] for line in answer.read().decode().splitlines() if line.startswith("data: ")]
    first = json.loads(events[0])["choices"][0]["delta"]
    assert first["tool_calls"][0]["function"]["name"] == "weather"
    assert json.loads(events[1])["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1] == "[DONE]"


#==========================================================================
# Errors go back to the caller
#==========================================================================
def test_errors_come_back_with_status_and_reason(base_url: str) -> None:
    """
    Pin 500 with the reason for a bad body, 404 for an unknown path.

    Input:
        . base_url      : the running fake server
    """
    with pytest.raises(HttpJsonError, match="500.*KeyError"):
        request_json(f"{base_url}/chat/completions", {"model": "fake-chat"})
    with pytest.raises(HttpJsonError, match="404"):
        request_json(f"{base_url}/unknown", {})
