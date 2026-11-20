"""Call a Langflow flow over its API: the same flow as the Playground."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  os
from    typing                          import  Any

from    factdesk.integrations.http_json  import  request_json


#==========================================================================
# Langflow API client
#==========================================================================
class LangflowClient:
    """
    Purpose: find, run and download flows. The key stays in the environment
    (LANGFLOW_API_KEY, from .env) and is read only when a request needs it.
    """

    #----------------------------------------------------------------------
    # Keep the address
    #----------------------------------------------------------------------
    def __init__(self, base_url: str, *, timeout: float = 300) -> None:
        """
        Remember where Langflow listens.

        Input:
            . base_url      : e.g. http://127.0.0.1:7860
            . timeout       : seconds per request; a flow run waits for the model
        """
        self._base_url = base_url.rstrip("/")
        self._timeout  = timeout

    #----------------------------------------------------------------------
    # Find a flow by name
    #----------------------------------------------------------------------
    def find_flow(self, name: str) -> str:
        """
        Return the id of the most recently updated flow with this name.

        Input:
            . name          : the flow name shown in Langflow
        Output:
            . flow_id       : its id, for run_flow and get_flow
        """
        url   = f"{self._base_url}/api/v1/flows/?get_all=true&header_flows=true"
        flows = request_json(url, headers=_headers(), timeout=self._timeout)
        mine  = [flow for flow in flows if flow.get("name") == name]
        if not mine:
            names = sorted({flow.get("name") for flow in flows if flow.get("name")})
            raise ValueError(f"no flow named {name!r} in Langflow; flows there: {names[:12]}")
        return sorted(mine, key=lambda flow: flow.get("updated_at") or "")[-1]["id"]

    #----------------------------------------------------------------------
    # Run a flow like the Playground
    #----------------------------------------------------------------------
    def run_flow(
        self,
        flow_id: str,
        question: str,
        *,
        tweaks: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        """
        Send one chat message through the flow and return the answer text.
        tweaks change component settings for this run only; session_id keeps
        conversations apart (Langflow's default session is the flow id).

        Input:
            . flow_id       : from find_flow
            . question      : the chat input
            . tweaks        : {component id: {field: value}}, or None
            . session_id    : a conversation id, or None
        Output:
            . answer        : the text of the Chat Output component
        """
        body: dict[str, Any] = {
            "input_value": question,
            "input_type": "chat",
            "output_type": "chat",
        }
        if tweaks:
            body["tweaks"] = tweaks
        if session_id:
            body["session_id"] = session_id
        url = f"{self._base_url}/api/v1/run/{flow_id}"
        out = request_json(url, body, _headers(), timeout=self._timeout)
        return out["outputs"][0]["outputs"][0]["results"]["message"]["text"].strip()

    #----------------------------------------------------------------------
    # Download a flow
    #----------------------------------------------------------------------
    def get_flow(self, flow_id: str) -> dict[str, Any]:
        """
        Return the flow JSON: the same content as Export in the interface.

        Input:
            . flow_id       : from find_flow
        Output:
            . flow          : the flow, with data.nodes and data.edges
        """
        url = f"{self._base_url}/api/v1/flows/{flow_id}"
        return request_json(url, headers=_headers(), timeout=self._timeout)


#==========================================================================
# API key header
#==========================================================================
def _headers() -> dict[str, str]:
    """
    Build the x-api-key header; a missing key is one clear sentence, not a 403.

    Output:
        . headers       : {"x-api-key": key}
    """
    key = os.environ.get("LANGFLOW_API_KEY")
    if not key:
        raise ValueError(
            "LANGFLOW_API_KEY is not set: create a key in Langflow (Settings -> Langflow API "
            "Keys) and put it in .env, next to main.py"
        )
    return {"x-api-key": key}
