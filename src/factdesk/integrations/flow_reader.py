"""Read an exported Langflow flow: components, their key settings, the edges.

A flow file is data.nodes (each with data.type and data.node.template, the
fields) and data.edges (sourceHandle.name -> targetHandle.fieldName). The
reader shows the fields a person sets on the canvas, plus a few advanced ones
that decide where requests go, and flags a password field holding a value.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    dataclasses                     import  dataclass
from    typing                          import  Any


#==========================================================================
# Reader settings
#==========================================================================
@dataclass(frozen=True)
class FlowFields:
    """
    Purpose: which fields the summary shows and which password values are
    safe placeholders (not a real key), and the search component's k field.
    """

    always_shown: tuple[str, ...]       = (
        "openai_api_base", "base_url", "max_tokens", "number_of_results",
        "chroma_server_host", "chroma_server_http_port", "persist_directory",
        "tiktoken_model_name", "tiktoken_enable",
    )
    hidden: tuple[str, ...]             = ("code",)
    safe_secret_values: tuple[str, ...] = ("", "local", "none", "dummy", "OPENAI_API_KEY")
    search_kind: str                    = "Chroma"
    k_field: str                        = "number_of_results"


#==========================================================================
# One component
#==========================================================================
@dataclass(frozen=True)
class FlowComponent:
    """
    Purpose: one node of the flow with the settings worth reading.
    """

    node_id: str
    kind: str
    name: str
    settings: dict[str, Any]


#==========================================================================
# The whole flow
#==========================================================================
@dataclass(frozen=True)
class FlowSummary:
    """
    Purpose: what a person reviews before sharing or changing a flow.
    """

    name: str
    components: list[FlowComponent]
    edges: list[str]
    secrets: list[str]


#==========================================================================
# Read a flow
#==========================================================================
def read_flow(flow: dict[str, Any], *, shown: FlowFields = FlowFields()) -> FlowSummary:
    """
    Walk data.nodes and data.edges of an exported flow.

    Input:
        . flow          : the flow JSON (Export, or LangflowClient.get_flow)
        . shown         : which fields to show and which secrets are safe
    Output:
        . summary       : components, "A.out -> B.in" edges, fields holding a secret
    """
    nodes = flow["data"]["nodes"]
    names = {node["id"]: _display_name(node) for node in nodes}
    return FlowSummary(
        name        = flow.get("name", "?"),
        components  = [_component(node, names[node["id"]], shown) for node in nodes],
        edges       = [_edge(edge, names) for edge in flow["data"]["edges"]],
        secrets     = [
            f"{names[node['id']]}.{key}"
            for node in nodes
            for key, field in _template(node).items()
            if _holds_secret(field, shown)
        ],
    )


#==========================================================================
# Find a component
#==========================================================================
def node_id(flow: dict[str, Any], kind: str) -> str:
    """
    Return the id of the first component of one type, e.g. "Chroma".

    Input:
        . flow          : the flow JSON
        . kind          : the component type (data.type)
    Output:
        . node_id       : its id, the key of a tweak
    """
    for node in flow["data"]["nodes"]:
        if node["data"]["type"] == kind:
            return node["id"]
    kinds = sorted({node["data"]["type"] for node in flow["data"]["nodes"]})
    raise ValueError(f"no component of type {kind!r} in this flow; types: {kinds}")


#==========================================================================
# Build a tweak
#==========================================================================
def tweak(flow: dict[str, Any], *, kind: str, field: str, value: Any) -> dict[str, dict[str, Any]]:
    """
    Build the tweak that changes one field for ONE run: {node id: {field: value}}.
    The flow saved in Langflow does not change.

    Input:
        . flow          : the flow JSON
        . kind          : the component type, e.g. "Chroma"
        . field         : the template field, e.g. "number_of_results"
        . value         : the value for this run
    Output:
        . tweaks        : the tweaks argument of LangflowClient.run_flow
    """
    return {node_id(flow, kind): {field: value}}


#==========================================================================
# Settings of one component
#==========================================================================
def _component(node: dict[str, Any], name: str, shown: FlowFields) -> FlowComponent:
    """
    Keep the scalar fields that have a value and are set on the canvas (not
    advanced), or are in always_shown; a password shows as "***".

    Input:
        . node          : one entry of data.nodes
        . name          : its display name
        . shown         : the reader settings
    Output:
        . component     : id, type, name and the kept settings
    """
    settings = {}
    for key, field in _template(node).items():
        if not isinstance(field, dict) or key in shown.hidden:
            continue
        value = field.get("value")
        if value in ("", None) or not isinstance(value, str | int | float | bool):
            continue
        if field.get("advanced") and key not in shown.always_shown:
            continue
        settings[key] = "***" if field.get("password") and _holds_secret(field, shown) else value
    return FlowComponent(node["id"], node["data"]["type"], name, settings)


#==========================================================================
# Does a field hold a real secret?
#==========================================================================
def _holds_secret(field: Any, shown: FlowFields) -> bool:
    """
    Flag a password field with a value that is not a placeholder; a value
    loaded from a Langflow global variable is only the variable's name.

    Input:
        . field         : one template field
        . shown         : the safe placeholder values
    Output:
        . secret        : True when sharing the file would share a key
    """
    if not isinstance(field, dict) or not field.get("password") or field.get("load_from_db"):
        return False
    return field.get("value") not in (None, *shown.safe_secret_values)


#==========================================================================
# One edge as text
#==========================================================================
def _edge(edge: dict[str, Any], names: dict[str, str]) -> str:
    """
    Write an edge as "Component.output -> Component.input".

    Input:
        . edge          : one entry of data.edges
        . names         : node id -> display name
    Output:
        . text          : the edge in one line
    """
    source = edge["data"]["sourceHandle"]["name"]
    target = edge["data"]["targetHandle"]["fieldName"]
    return f"{names[edge['source']]}.{source} -> {names[edge['target']]}.{target}"


#==========================================================================
# Fields of a node
#==========================================================================
def _template(node: dict[str, Any]) -> dict[str, Any]:
    """
    Return the template (the fields) of one node.

    Input:
        . node          : one entry of data.nodes
    Output:
        . template      : field name -> field dict
    """
    return node["data"]["node"]["template"]


#==========================================================================
# Name on the canvas
#==========================================================================
def _display_name(node: dict[str, Any]) -> str:
    """
    Use the name shown on the canvas, or the type when the file has none.

    Input:
        . node          : one entry of data.nodes
    Output:
        . name          : display name
    """
    return node["data"]["node"].get("display_name") or node["data"]["type"]
