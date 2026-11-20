"""Record every run of a LangChain or LangGraph call as a tree, on this laptop.

The same mechanism LangSmith's tracer uses: a callback handler that
LangChain calls at the start and the end of each run (chain, chat model,
retriever, tool). No account, no network; trace_view.py saves and prints it.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  time
from    dataclasses                     import  dataclass, field
from    typing                          import  Any
from    uuid                            import  UUID

from    langchain_core.callbacks        import  BaseCallbackHandler


#==========================================================================
# One run
#==========================================================================
@dataclass
class Run:
    """
    Purpose: one unit of work: type, name, parent, input, output, time,
    tokens. Mutable: the end callback fills the output.
    """

    run_id: str
    parent: str | None
    kind: str
    name: str
    inputs: Any
    started: float
    tags: list[str]                     = field(default_factory=list)
    metadata: dict[str, Any]            = field(default_factory=dict)
    outputs: Any                        = None
    error: str | None                   = None
    ended: float | None                 = None
    tokens: dict[str, int] | None       = None


#==========================================================================
# Callback handler that keeps the run tree
#==========================================================================
class LocalTracer(BaseCallbackHandler):
    """
    Purpose: keep every run of the calls it is passed to (config callbacks),
    with parent links, so the whole tree can be saved and printed.
    """

    #----------------------------------------------------------------------
    # Start empty
    #----------------------------------------------------------------------
    def __init__(self) -> None:
        """
        Create an empty record.
        """
        self.runs: dict[str, Run] = {}

    #----------------------------------------------------------------------
    # Runs without a recorded parent
    #----------------------------------------------------------------------
    def roots(self) -> list[Run]:
        """
        Return the top-level runs, in time order.

        Output:
            . roots         : runs whose parent is not recorded
        """
        roots = [run for run in self.runs.values() if run.parent not in self.runs]
        return sorted(roots, key=lambda run: run.started)

    #----------------------------------------------------------------------
    # Find runs
    #----------------------------------------------------------------------
    def find(self, *, kind: str | None = None, name: str | None = None) -> list[Run]:
        """
        Select runs by type and/or name, in time order.

        Input:
            . kind          : "chain", "llm", "retriever", "tool", or None
            . name          : run name, or None
        Output:
            . runs          : the matching runs
        """
        found = [
            run
            for run in self.runs.values()
            if (kind is None or run.kind == kind) and (name is None or run.name == name)
        ]
        return sorted(found, key=lambda run: run.started)

    #----------------------------------------------------------------------
    # Chain runs (graphs, nodes, prompts, parsers)
    #----------------------------------------------------------------------
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Open a chain run.

        Input:
            . serialized    : LangChain's description of the runnable
            . inputs        : the chain input
            . run_id        : id of this run
            . parent_run_id : id of the parent run, or None
            . kwargs        : name, tags, metadata
        """
        self._start(run_id, parent_run_id, "chain", _name(serialized, kwargs, "chain"), inputs, kwargs)

    #----------------------------------------------------------------------
    # Chain run ended
    #----------------------------------------------------------------------
    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """
        Close a chain run with its output.

        Input:
            . outputs       : the chain output
            . run_id        : id of this run
            . kwargs        : unused callback extras
        """
        self._end(run_id, outputs=outputs)

    #----------------------------------------------------------------------
    # Chain run failed
    #----------------------------------------------------------------------
    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """
        Close a chain run with its error (an interrupt shows up here too).

        Input:
            . error         : the exception
            . run_id        : id of this run
            . kwargs        : unused callback extras
        """
        self._end(run_id, error=f"{type(error).__name__}: {error}")

    #----------------------------------------------------------------------
    # Chat model call
    #----------------------------------------------------------------------
    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Open a chat model run with the messages it receives.

        Input:
            . serialized    : LangChain's description of the model
            . messages      : one list of messages per prompt (one prompt here)
            . run_id        : id of this run
            . parent_run_id : id of the parent run, or None
            . kwargs        : name, tags, metadata
        """
        name = _name(serialized, kwargs, "chat model")
        self._start(run_id, parent_run_id, "llm", name, messages[0], kwargs)

    #----------------------------------------------------------------------
    # Model call ended
    #----------------------------------------------------------------------
    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """
        Close a model run with its message and token counts.

        Input:
            . response      : LLMResult with one generation
            . run_id        : id of this run
            . kwargs        : unused callback extras
        """
        generation = response.generations[0][0]
        message    = getattr(generation, "message", None)
        output     = message if message is not None else generation.text
        self._end(run_id, outputs=output, tokens=getattr(message, "usage_metadata", None))

    #----------------------------------------------------------------------
    # Model call failed
    #----------------------------------------------------------------------
    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """
        Close a model run with its error.

        Input:
            . error         : the exception
            . run_id        : id of this run
            . kwargs        : unused callback extras
        """
        self._end(run_id, error=f"{type(error).__name__}: {error}")

    #----------------------------------------------------------------------
    # Retriever call
    #----------------------------------------------------------------------
    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Open a retriever run with its query.

        Input:
            . serialized    : LangChain's description of the retriever
            . query         : the search text
            . run_id        : id of this run
            . parent_run_id : id of the parent run, or None
            . kwargs        : name, tags, metadata
        """
        name = _name(serialized, kwargs, "retriever")
        self._start(run_id, parent_run_id, "retriever", name, query, kwargs)

    #----------------------------------------------------------------------
    # Retriever ended
    #----------------------------------------------------------------------
    def on_retriever_end(self, documents: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """
        Close a retriever run with the documents found.

        Input:
            . documents     : the retrieved Documents
            . run_id        : id of this run
            . kwargs        : unused callback extras
        """
        self._end(run_id, outputs=documents)

    #----------------------------------------------------------------------
    # Retriever failed
    #----------------------------------------------------------------------
    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """
        Close a retriever run with its error.

        Input:
            . error         : the exception
            . run_id        : id of this run
            . kwargs        : unused callback extras
        """
        self._end(run_id, error=f"{type(error).__name__}: {error}")

    #----------------------------------------------------------------------
    # Tool call
    #----------------------------------------------------------------------
    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Open a tool run with its input.

        Input:
            . serialized    : LangChain's description of the tool
            . input_str     : the tool input as text
            . run_id        : id of this run
            . parent_run_id : id of the parent run, or None
            . kwargs        : name, tags, metadata, inputs
        """
        name = _name(serialized, kwargs, "tool")
        self._start(run_id, parent_run_id, "tool", name, kwargs.get("inputs") or input_str, kwargs)

    #----------------------------------------------------------------------
    # Tool ended
    #----------------------------------------------------------------------
    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """
        Close a tool run with its result (a ToolMessage's content).

        Input:
            . output        : the tool result
            . run_id        : id of this run
            . kwargs        : unused callback extras
        """
        self._end(run_id, outputs=getattr(output, "content", output))

    #----------------------------------------------------------------------
    # Tool failed
    #----------------------------------------------------------------------
    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """
        Close a tool run with its error.

        Input:
            . error         : the exception
            . run_id        : id of this run
            . kwargs        : unused callback extras
        """
        self._end(run_id, error=f"{type(error).__name__}: {error}")

    #----------------------------------------------------------------------
    # Open one run
    #----------------------------------------------------------------------
    def _start(
        self,
        run_id: UUID,
        parent: UUID | None,
        kind: str,
        name: str,
        inputs: Any,
        extra: dict[str, Any],
    ) -> None:
        """
        Record the start of a run.

        Input:
            . run_id        : id of this run
            . parent        : id of the parent run, or None
            . kind          : chain, llm, retriever or tool
            . name          : run name
            . inputs        : what the run received
            . extra         : callback kwargs with tags and metadata
        """
        self.runs[str(run_id)] = Run(
            run_id      = str(run_id),
            parent      = str(parent) if parent else None,
            kind        = kind,
            name        = name,
            inputs      = inputs,
            started     = time.time(),
            tags        = list(extra.get("tags") or []),
            metadata    = dict(extra.get("metadata") or {}),
        )

    #----------------------------------------------------------------------
    # Close one run
    #----------------------------------------------------------------------
    def _end(
        self,
        run_id: UUID,
        *,
        outputs: Any = None,
        error: str | None = None,
        tokens: Any = None,
    ) -> None:
        """
        Record the end of a run; an unknown id (started before this tracer
        was attached) is ignored.

        Input:
            . run_id        : id of this run
            . outputs       : what the run returned
            . error         : "Type: message" when it failed
            . tokens        : usage_metadata of a model call
        """
        run = self.runs.get(str(run_id))
        if run is None:
            return
        run.outputs = outputs
        run.error   = error
        run.tokens  = dict(tokens) if tokens else None
        run.ended   = time.time()


#==========================================================================
# Best run name
#==========================================================================
def _name(serialized: dict[str, Any] | None, extra: dict[str, Any], default: str) -> str:
    """
    Choose the most useful name LangChain gives for a run.

    Input:
        . serialized    : LangChain's description, may be None
        . extra         : callback kwargs, may hold "name"
        . default       : used when neither has a name
    Output:
        . name          : run name
    """
    return extra.get("name") or (serialized or {}).get("name") or default
