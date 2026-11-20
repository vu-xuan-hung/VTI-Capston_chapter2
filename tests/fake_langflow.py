"""A small fake Langflow behind real HTTP: the endpoints LangflowClient uses."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  gzip
import  json
import  threading
from    http.server                     import  BaseHTTPRequestHandler, ThreadingHTTPServer
from    typing                          import  Any


FLOW_ID   = "flow-1"
FLOW_NAME = "factdesk-rag"


#==========================================================================
# A tiny flow
#==========================================================================
def tiny_flow() -> dict[str, Any]:
    """
    Build a three-component flow: Chat Input -> Chroma DB -> OpenAI.

    Output:
        . flow          : a flow JSON with data.nodes and data.edges
    """

    #----------------------------------------------------------------------
    # One node in the export shape
    #----------------------------------------------------------------------
    def node(node_id: str, kind: str, name: str, template: dict[str, Any]) -> dict[str, Any]:
        """
        Write one node in Langflow's export shape.

        Input:
            . node_id       : component id
            . kind          : component type
            . name          : display name
            . template      : the fields
        Output:
            . node          : one entry of data.nodes
        """
        return {"id": node_id, "data": {"type": kind, "node": {"display_name": name, "template": template}}}

    #----------------------------------------------------------------------
    # Nodes and edges
    #----------------------------------------------------------------------
    nodes = [
        node("ChatInput-a", "ChatInput", "Chat Input", {}),
        node("Chroma-b", "Chroma", "Chroma DB", {
            "collection_name": {"value": "factdesk-kb"},
            "number_of_results": {"value": 3, "advanced": True},
        }),
        node("OpenAIModel-c", "OpenAIModel", "OpenAI", {
            "api_key": {"value": "local", "password": True},
            "model_name": {"value": "Qwen"},
        }),
    ]
    edges = [
        {"source": "ChatInput-a", "target": "Chroma-b",
         "data": {"sourceHandle": {"name": "message"}, "targetHandle": {"fieldName": "search_query"}}},
        {"source": "Chroma-b", "target": "OpenAIModel-c",
         "data": {"sourceHandle": {"name": "search_results"}, "targetHandle": {"fieldName": "input_value"}}},
    ]
    return {"id": FLOW_ID, "name": FLOW_NAME, "data": {"nodes": nodes, "edges": edges}}


#==========================================================================
# Fake Langflow server
#==========================================================================
class FakeLangflow:
    """
    Purpose: answer /health_check, the flow list, one flow (gzipped, as
    Langflow does) and /api/v1/run; it remembers every run body it received.
    """

    #----------------------------------------------------------------------
    # Start the server on a free port
    #----------------------------------------------------------------------
    def __init__(self) -> None:
        """
        Serve in a background thread until close().
        """

        #------------------------------------------------------------------
        # What the fake remembers and answers
        #------------------------------------------------------------------
        self.requests: list[dict[str, Any]] = []
        self.answer = "The kit answers from the flow."
        owner       = self

        #------------------------------------------------------------------
        # The endpoints
        #------------------------------------------------------------------
        class Handler(BaseHTTPRequestHandler):
            """
            Purpose: the four endpoints; a missing key is a 403 like Langflow.
            """

            def _send(self, code: int, value: Any, *, zipped: bool = False) -> None:
                """
                Write one JSON answer, gzipped when asked.

                Input:
                    . code          : HTTP status
                    . value         : JSON body
                    . zipped        : gzip the body
                """
                data = json.dumps(value).encode()
                data = gzip.compress(data) if zipped else data
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                """
                Answer the health check, the flow list and one flow.
                """
                if self.path == "/health_check":
                    self._send(200, {"status": "ok"})
                elif not self.headers.get("x-api-key"):
                    self._send(403, {"detail": "no key"})
                elif self.path.startswith("/api/v1/flows/?"):
                    self._send(200, [{"id": FLOW_ID, "name": FLOW_NAME, "updated_at": "2026-11-01"}])
                elif self.path == f"/api/v1/flows/{FLOW_ID}":
                    self._send(200, tiny_flow(), zipped=True)
                else:
                    self._send(404, {"detail": "not found"})

            def do_POST(self) -> None:
                """
                Record a run request and answer with the fixed text.
                """
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(body)
                message = {"text": owner.answer + "  "}
                self._send(200, {"outputs": [{"outputs": [{"results": {"message": message}}]}]})

            def log_message(self, format: str, *args: Any) -> None:
                """
                Keep the test output quiet.

                Input:
                    . format        : printf-style format
                    . args          : its values
                """
                return

        #------------------------------------------------------------------
        # Serve on a free port
        #------------------------------------------------------------------
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url     = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    #----------------------------------------------------------------------
    # Stop the server
    #----------------------------------------------------------------------
    def close(self) -> None:
        """
        Shut the server down and free the port.
        """
        self._server.shutdown()
        self._server.server_close()
