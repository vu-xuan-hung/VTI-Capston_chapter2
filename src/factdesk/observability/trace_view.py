"""Save a run tree as JSON and print it on one screen, live or from a file."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
import  time
from    dataclasses                     import  dataclass
from    pathlib                         import  Path
from    typing                          import  Any

from    factdesk.observability.local_tracer  import  LocalTracer


#==========================================================================
# Printer settings
#==========================================================================
@dataclass(frozen=True)
class TreeStyle:
    """
    Purpose: how much of each run one line shows, and which LangGraph
    plumbing runs are hidden (their children stay visible).
    """

    brief_in: int                       = 34
    brief_out: int                      = 60
    plumbing: tuple[str, ...]           = ("ChannelWrite", "Branch", "Runnable")


#==========================================================================
# Tracer -> rows
#==========================================================================
def trace_rows(tracer: LocalTracer) -> list[dict[str, Any]]:
    """
    Turn every run into a JSON-ready row, nothing cut: the same idea as a
    LangSmith export.

    Input:
        . tracer        : a tracer that was passed as a callback
    Output:
        . rows          : id, parent, type, name, start, seconds, tokens, tags,
                          metadata, inputs, outputs, error; in time order
    """

    #----------------------------------------------------------------------
    # One row per run, in time order
    #----------------------------------------------------------------------
    return [
        {
            "id": run.run_id,
            "parent": run.parent,
            "type": run.kind,
            "name": run.name,
            "start": run.started,
            "seconds": round((run.ended or run.started) - run.started, 3),
            "tokens": run.tokens,
            "tags": run.tags,
            "metadata": _full(run.metadata),
            "inputs": _full(run.inputs),
            "outputs": _full(run.outputs),
            "error": run.error,
        }
        for run in sorted(tracer.runs.values(), key=lambda item: item.started)
    ]


#==========================================================================
# Save and load
#==========================================================================
def save_trace(tracer: LocalTracer, path: Path) -> Path:
    """
    Write the run tree to a JSON file.

    Input:
        . tracer        : the tracer of one call
        . path          : target file; parent folders are created
    Output:
        . path          : the file written
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(trace_rows(tracer), indent=1, ensure_ascii=False, default=str)
    path.write_text(text, encoding="utf-8")
    return path


#==========================================================================
# Read a saved trace
#==========================================================================
def load_trace(path: Path) -> list[dict[str, Any]]:
    """
    Read the rows of a file written by save_trace.

    Input:
        . path          : a trace JSON file
    Output:
        . rows          : the saved rows
    """
    return json.loads(path.read_text(encoding="utf-8"))


#==========================================================================
# Default trace file name
#==========================================================================
def trace_path(runtime_dir: Path, label: str) -> Path:
    """
    Name a new trace file runtime/traces/<date>-<time>-<label>.json.

    Input:
        . runtime_dir   : the project's runtime folder
        . label         : a short word, e.g. the command name
    Output:
        . path          : a path that does not overwrite older traces
    """
    return runtime_dir / "traces" / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.json"


#==========================================================================
# Rows -> one line per run
#==========================================================================
def tree_lines(rows: list[dict[str, Any]], *, style: TreeStyle = TreeStyle()) -> list[str]:
    """
    Indent each run under its parent; plumbing runs are hidden and their
    children move up one level.

    Input:
        . rows          : from trace_rows or load_trace
        . style         : line widths and hidden run names
    Output:
        . lines         : "type name seconds tokens  input -> output" per run
    """

    #----------------------------------------------------------------------
    # Index the children of every run
    #----------------------------------------------------------------------
    ids      = {row["id"] for row in rows}
    children: dict[str | None, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: item["start"]):
        children.setdefault(row["parent"] if row["parent"] in ids else None, []).append(row)
    #----------------------------------------------------------------------
    # Walk from the roots
    #----------------------------------------------------------------------
    lines: list[str] = []

    def walk(row: dict[str, Any], depth: int) -> None:
        """
        Add one run and its children.

        Input:
            . row           : one run
            . depth         : indent level
        """
        hidden = row["type"] == "chain" and row["name"].startswith(style.plumbing)
        if not hidden:
            lines.append(_line(row, depth, style))
        for child in children.get(row["id"], []):
            walk(child, depth if hidden else depth + 1)

    for root in children.get(None, []):
        walk(root, 0)
    return lines


#==========================================================================
# One printed line
#==========================================================================
def _line(row: dict[str, Any], depth: int, style: TreeStyle) -> str:
    """
    Format one run: type, name, time, tokens, a short input and output.

    Input:
        . row           : one run
        . depth         : indent level
        . style         : line widths
    Output:
        . line          : one line of text
    """
    seconds = f"{row['seconds']:.1f}s"
    tokens  = f" {row['tokens']['total_tokens']} tok" if row.get("tokens") else ""
    output  = f"ERROR {row['error']}" if row.get("error") else _brief(row["outputs"], style.brief_out)
    return (
        f"{'  ' * depth}{row['type']:<9} {row['name'][:26]:<26} {seconds:>6}{tokens}  "
        f"{_brief(row['inputs'], style.brief_in)} -> {output}"
    )


#==========================================================================
# Short text of any saved value
#==========================================================================
def _brief(value: Any, size: int) -> str:
    """
    Make a short one-line text: a message shows its tool calls and content,
    a document its chunk id.

    Input:
        . value         : a saved input or output (JSON-ready)
        . size          : maximum characters
    Output:
        . text          : at most size characters
    """
    if isinstance(value, dict) and "role" in value:
        calls = value.get("tool_calls") or []
        value = (f"tool_calls={json.dumps(calls, ensure_ascii=False)} " if calls else "") + str(value["content"])
    elif isinstance(value, dict) and "page_content" in value:
        value = f"[{value['metadata'].get('chunk_id', '?')}] {value['page_content']}"
    elif isinstance(value, list) and value and isinstance(value[0], dict) and "page_content" in value[0]:
        value = " | ".join(f"[{doc['metadata'].get('chunk_id', '?')}]" for doc in value)
    elif isinstance(value, list) and value and isinstance(value[0], dict) and "role" in value[0]:
        value = _brief(value[-1], size)
    elif isinstance(value, dict):
        value = json.dumps({key: _brief(item, 40) for key, item in value.items()}, ensure_ascii=False)
    flat = " ".join(str(value).split())
    return flat if len(flat) <= size else f"{flat[: size - 3]}..."


#==========================================================================
# JSON-ready copy of any value
#==========================================================================
def _full(value: Any) -> Any:
    """
    Convert messages, Documents, Pydantic objects and containers into plain
    JSON values, keeping all of their text.

    Input:
        . value         : anything a callback received
    Output:
        . value         : dicts, lists, strings and numbers only
    """
    if hasattr(value, "page_content"):
        return {"page_content": value.page_content, "metadata": _full(value.metadata)}
    if hasattr(value, "content") and hasattr(value, "type"):
        message: dict[str, Any] = {"role": value.type, "content": _full(value.content)}
        if getattr(value, "tool_calls", None):
            message["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in value.tool_calls]
        return message
    if isinstance(value, dict):
        return {str(key): _full(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_full(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return _full(value.model_dump()) if hasattr(value, "model_dump") else str(value)
