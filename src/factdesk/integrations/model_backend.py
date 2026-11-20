"""Answer OpenAI-format requests with any LangChain chat model and embeddings.

The model server (model_server.py) only speaks HTTP; this class turns one
OpenAI request body into one LangChain call and back. The local backend uses
LocalChatModel + HuggingFaceEmbeddings, the fake one FakeChatModel +
HashEmbeddings: the same code path either way.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  base64
import  json
import  threading
import  time
import  uuid
from    typing                          import  Any

import  numpy                                       as      np
from    langchain_core.embeddings                   import  Embeddings
from    langchain_core.language_models.chat_models  import  BaseChatModel
from    langchain_core.messages                     import  convert_to_messages

from    factdesk.ai_logger              import  aiLogger


#==========================================================================
# OpenAI-format backend
#==========================================================================
class ModelBackend:
    """
    Purpose: answer /chat/completions and /embeddings bodies. One request at a
    time: a laptop holds one model in memory.
    """

    #----------------------------------------------------------------------
    # Keep the models
    #----------------------------------------------------------------------
    def __init__(
        self,
        *,
        chat: BaseChatModel,
        embeddings: Embeddings,
        chat_name: str,
        embed_name: str,
    ) -> None:
        """
        Wrap a chat model and an embedding model.

        Input:
            . chat          : any LangChain chat model with bind_tools
            . embeddings    : any LangChain embeddings
            . chat_name     : the name /v1/models lists for chat
            . embed_name    : the name /v1/models lists for embeddings
        """
        self.chat_name   = chat_name
        self.embed_name  = embed_name
        self._chat       = chat
        self._embeddings = embeddings
        self._lock       = threading.Lock()

    #----------------------------------------------------------------------
    # Answer one chat request
    #----------------------------------------------------------------------
    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        Run one chat completion; tools and tool_choice are passed to the
        model, its tool calls come back in the OpenAI format.

        Input:
            . body          : OpenAI chat request (messages, tools, tool_choice, max_tokens)
        Output:
            . completion    : OpenAI chat.completion object
        """

        #------------------------------------------------------------------
        # Build the call from the request
        #------------------------------------------------------------------
        messages = convert_to_messages([_text_parts_joined(item) for item in body["messages"]])
        model: Any = self._chat
        if body.get("tools"):
            model = model.bind_tools(body["tools"], tool_choice=body.get("tool_choice"))
        limit = body.get("max_tokens") or body.get("max_completion_tokens")
        if limit:
            model = model.bind(max_new_tokens=int(limit))

        #------------------------------------------------------------------
        # Run it, one request at a time
        #------------------------------------------------------------------
        started = time.time()
        with self._lock:
            answer = model.invoke(messages)
        usage = answer.usage_metadata or {}
        aiLogger.info(
            "chat: %s tokens in, %s out, %.1f s",
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            time.time() - started,
        )

        #------------------------------------------------------------------
        # Write the OpenAI answer
        #------------------------------------------------------------------
        message: dict[str, Any] = {"role": "assistant", "content": answer.text}
        if answer.tool_calls:
            message["content"]    = answer.text or None
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                }
                for call in answer.tool_calls
            ]
        finish = "tool_calls" if answer.tool_calls else "stop"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.chat_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    #----------------------------------------------------------------------
    # Embed texts
    #----------------------------------------------------------------------
    def embeddings(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        Embed one text or a list of texts. The openai Python client asks for
        encoding_format "base64" by default: float32 bytes, base64-encoded.

        Input:
            . body          : OpenAI embeddings request (input, encoding_format)
        Output:
            . answer        : OpenAI list of embedding objects
        """

        #------------------------------------------------------------------
        # Embed the texts
        #------------------------------------------------------------------
        texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("embeddings: input must be text; switch tiktoken off in the client")
        with self._lock:
            vectors = self._embeddings.embed_documents(texts)
        aiLogger.info("embeddings: %d text(s)", len(texts))
        #------------------------------------------------------------------
        # Write the OpenAI answer
        #------------------------------------------------------------------
        as_base64 = body.get("encoding_format") == "base64"
        data      = [
            {
                "object": "embedding",
                "index": index,
                "embedding": _base64(vector) if as_base64 else vector,
            }
            for index, vector in enumerate(vectors)
        ]
        return {
            "object": "list",
            "model": self.embed_name,
            "data": data,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }


#==========================================================================
# Join text parts of a message
#==========================================================================
def _text_parts_joined(message: dict[str, Any]) -> dict[str, Any]:
    """
    Turn content given as a list of parts ([{"type": "text", "text": ...}])
    into one string; local chat templates expect text.

    Input:
        . message       : one OpenAI message
    Output:
        . message       : the same message with string content
    """
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return {**message, "content": content}


#==========================================================================
# Encode one vector as base64 float32
#==========================================================================
def _base64(vector: list[float]) -> str:
    """
    Encode a vector the way the OpenAI API does for encoding_format=base64.

    Input:
        . vector        : one embedding
    Output:
        . text          : base64 of the little-endian float32 bytes
    """
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode()
