"""Serve a model in the OpenAI format, so that Langflow and the code call one model.

Endpoints: GET /v1/models, POST /v1/chat/completions (stream and tools too),
POST /v1/embeddings: what Langflow's OpenAI and OpenAI Embeddings components
call. The standard library is enough.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
from    http.server                     import  BaseHTTPRequestHandler, ThreadingHTTPServer
from    typing                          import  Any

from    factdesk.ai_logger                   import  aiLogger
from    factdesk.integrations.model_backend  import  ModelBackend


#==========================================================================
# Build a server around a backend
#==========================================================================
def make_server(backend: ModelBackend, *, host: str, port: int) -> ThreadingHTTPServer:
    """
    Create the HTTP server; the caller runs serve_forever().

    Input:
        . backend       : the models that answer
        . host          : address to listen on, normally 127.0.0.1
        . port          : TCP port; 0 picks a free one (tests)
    Output:
        . server        : a ThreadingHTTPServer, not yet serving
    """

    class Handler(_OpenAiHandler):
        """
        Purpose: the handler bound to this backend.
        """

        models = backend

    return ThreadingHTTPServer((host, port), Handler)


#==========================================================================
# Request handler
#==========================================================================
class _OpenAiHandler(BaseHTTPRequestHandler):
    """
    Purpose: route the three OpenAI endpoints to the backend; an error goes
    back to the caller as JSON, where Langflow and ServerChat show it.
    """

    models: ModelBackend

    #----------------------------------------------------------------------
    # GET /v1/models
    #----------------------------------------------------------------------
    def do_GET(self) -> None:
        """
        List the chat and the embedding model.
        """
        if self.path.rstrip("/").endswith("/models"):
            names = [self.models.chat_name, self.models.embed_name]
            data  = [{"id": name, "object": "model", "owned_by": "local"} for name in names]
            self._send(200, {"object": "list", "data": data})
        else:
            self._send(404, {"error": {"message": f"unknown path {self.path}"}})

    #----------------------------------------------------------------------
    # POST /v1/chat/completions and /v1/embeddings
    #----------------------------------------------------------------------
    def do_POST(self) -> None:
        """
        Answer one request; a bad request or a model failure is a 500 with
        the reason, logged here too.
        """
        try:
            #--------------------------------------------------------------
            # Read the JSON body
            #--------------------------------------------------------------
            size = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(size) or b"{}")
            #--------------------------------------------------------------
            # Route it to the backend
            #--------------------------------------------------------------
            if self.path.endswith("/chat/completions"):
                result = self.models.chat(body)
                if body.get("stream"):
                    self._stream(result)
                else:
                    self._send(200, result)
            elif self.path.endswith("/embeddings"):
                self._send(200, self.models.embeddings(body))
            else:
                self._send(404, {"error": {"message": f"unknown path {self.path}"}})
        except (KeyError, ValueError, TypeError, RuntimeError) as error:
            aiLogger.exception("Request failed")
            self._send(500, {"error": {"message": f"{type(error).__name__}: {error}"}})

    #----------------------------------------------------------------------
    # Send one JSON answer
    #----------------------------------------------------------------------
    def _send(self, code: int, value: dict[str, Any]) -> None:
        """
        Write a JSON response with its length.

        Input:
            . code          : HTTP status
            . value         : JSON body
        """
        data = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    #----------------------------------------------------------------------
    # stream=true: the whole answer as one chunk
    #----------------------------------------------------------------------
    def _stream(self, result: dict[str, Any]) -> None:
        """
        Send a finished completion as Server-Sent Events: one delta with the
        content and tool calls, one with the finish reason, then [DONE].

        Input:
            . result        : a chat.completion from the backend
        """

        #------------------------------------------------------------------
        # Build the two chunks
        #------------------------------------------------------------------
        message = result["choices"][0]["message"]
        delta   = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = [
                dict(call, index=index) for index, call in enumerate(message["tool_calls"])
            ]
        finish = result["choices"][0]["finish_reason"]
        chunks = [
            {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": result["usage"]},
        ]
        #------------------------------------------------------------------
        # Send them as events
        #------------------------------------------------------------------
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            chunk.update(
                id      = result["id"],
                object  = "chat.completion.chunk",
                created = result["created"],
                model   = result["model"],
            )
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    #----------------------------------------------------------------------
    # Silence the default access log
    #----------------------------------------------------------------------
    def log_message(self, format: str, *args: Any) -> None:
        """
        Drop the per-request access line; the backend logs one short line.

        Input:
            . format        : printf-style format from BaseHTTPRequestHandler
            . args          : its values
        """
        return
