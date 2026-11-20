"""Send and receive JSON over HTTP with the standard library only."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  gzip
import  json
import  urllib.error
import  urllib.request
from    typing                          import  Any


#==========================================================================
# HTTP error with the server's own words
#==========================================================================
class HttpJsonError(OSError):
    """
    Purpose: carry the status code and the start of the server's answer, so a
    500 or a 403 says why it failed.
    """


#==========================================================================
# One JSON request
#==========================================================================
def request_json(
    url: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    *,
    timeout: float = 300,
) -> Any:
    """
    GET (body None) or POST JSON and decode the answer. Langflow compresses
    large answers even unasked, so a gzip body is unpacked here.

    Input:
        . url           : full address
        . body          : JSON body to POST, or None for GET
        . headers       : extra headers, e.g. {"x-api-key": ...}
        . timeout       : seconds before the request gives up
    Output:
        . value         : the decoded JSON answer
    """
    data    = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data    = data,
        headers = {"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            raw = answer.read()
    except urllib.error.HTTPError as error:
        text = error.read().decode(errors="replace")[:400]
        raise HttpJsonError(f"HTTP {error.code} from {url}: {text}") from None
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
