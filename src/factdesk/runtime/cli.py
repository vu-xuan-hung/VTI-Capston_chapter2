"""The kit's commands: servers, the knowledge base, traces, flows, keys.

Each command is a function (config, args) -> exit code; main.py declares
the arguments and kit_commands() maps names to functions. Project commands
live in runtime/project_commands.py, the same way.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  argparse
import  json
from    collections.abc                 import  Callable

from    factdesk.ai_logger                     import  set_up_ai_logger
from    factdesk.config                        import  Config
from    factdesk.evaluation.compare            import  compare_experiments
from    factdesk.evaluation.experiment         import  load_experiment
from    factdesk.integrations.flow_reader      import  FlowFields, FlowSummary, read_flow, tweak
from    factdesk.integrations.langflow_client  import  LangflowClient
from    factdesk.integrations.server_check     import  check_servers
from    factdesk.knowledge.chunking            import  chunk_documents, load_documents
from    factdesk.knowledge.vector_store        import  open_vector_store
from    factdesk.observability.cloud           import  load_env_file
from    factdesk.observability.local_tracer    import  LocalTracer
from    factdesk.observability.trace_config    import  run_config
from    factdesk.observability.trace_view      import  load_trace, save_trace, trace_path, tree_lines
from    factdesk.runtime.key_scan              import  find_keys


type Command = Callable[[Config, argparse.Namespace], int]


#==========================================================================
# Command table
#==========================================================================
def kit_commands() -> dict[str, Command]:
    """
    Map each kit command name to its function.

    Output:
        . commands      : name -> function(config, args) -> exit code
    """
    return {
        "serve-model": serve_model,
        "servers": servers,
        "index": index,
        "search": search,
        "trace-show": trace_show,
        "scan-keys": scan_keys,
        "flow-show": flow_show,
        "flow-ask": flow_ask,
        "compare": compare,
    }


#==========================================================================
# serve-model
#==========================================================================
def serve_model(config: Config, args: argparse.Namespace) -> int:
    """
    Run the OpenAI-compatible model server until Ctrl+C: the local models,
    the small chat model with --small, or the fake backend with --fake.

    Input:
        . config        : model names, host and port
        . args          : small, serve_fake (--fake), port
    Output:
        . code          : 0 after Ctrl+C
    """
    from    factdesk.integrations.local_models  import  fake_backend, load_local_backend
    from    factdesk.integrations.model_server  import  make_server

    #----------------------------------------------------------------------
    # Choose the backend
    #----------------------------------------------------------------------
    set_up_ai_logger("INFO")
    if args.serve_fake or config.model.backend == "fake":
        backend = fake_backend()
    else:
        backend = load_local_backend(
            chat_name       = config.model.small_chat_model if args.small else config.model.chat_model,
            embed_name      = config.model.embedding_model,
            max_new_tokens  = config.model.max_new_tokens,
        )

    #----------------------------------------------------------------------
    # Serve until Ctrl+C
    #----------------------------------------------------------------------
    port   = args.port or config.servers.model_port
    server = make_server(backend, host=config.servers.model_host, port=port)
    print(f"Server ready: http://{config.servers.model_host}:{port}/v1  model {backend.chat_name}"
          "   (stop with Ctrl+C)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


#==========================================================================
# servers
#==========================================================================
def servers(config: Config, args: argparse.Namespace) -> int:
    """
    Print [OK] or [NO] for the model server, Chroma and Langflow; a missing
    one shows how to start it.

    Input:
        . config        : the addresses
        . args          : unused
    Output:
        . code          : 0 when all answer, else 1
    """
    statuses = check_servers(config.servers, kb_dir=config.paths.kb_dir)
    for status in statuses:
        print(f"[{'OK' if status.alive else 'NO'}] {status.name:<14} {status.url}")
        if not status.alive:
            print(f"     start it: {status.start}")
    return 0 if all(status.alive for status in statuses) else 1


#==========================================================================
# index
#==========================================================================
def index(config: Config, args: argparse.Namespace) -> int:
    """
    Build or refresh the collection from the documents folder; running it
    twice changes nothing the second time.

    Input:
        . config        : docs folder, chunking, collection, backend
        . args          : unused
    Output:
        . code          : 0
    """
    docs   = load_documents(config.paths.docs_dir)
    chunks = chunk_documents(
        docs,
        size    = config.knowledge.chunk_size,
        overlap = config.knowledge.chunk_overlap,
    )
    report = open_vector_store(config).index(chunks)
    print(
        f"[OK] {config.knowledge.collection} ({config.knowledge.store}): {len(docs)} documents, "
        f"{report.total} chunks; added {report.added}, changed {report.changed}, "
        f"removed {report.removed}, unchanged {report.unchanged}"
    )
    return 0


#==========================================================================
# search
#==========================================================================
def search(config: Config, args: argparse.Namespace) -> int:
    """
    Print the k nearest chunks with their scores; "*" marks the ones the
    retriever keeps (score >= threshold). --trace also saves a run tree.

    Input:
        . config        : collection, top_k, threshold
        . args          : query, k, where (key=value, repeatable), trace
    Output:
        . code          : 0
    """
    where = _where(args.where)
    store = open_vector_store(config)
    hits  = store.search(args.query, k=args.k, where=where)
    for hit in hits:
        mark = "*" if hit.score >= config.knowledge.threshold else " "
        print(f"{mark} {hit.score:.3f}  [{hit.chunk_id}] {hit.title}: {' '.join(hit.text.split())[:110]}")
    print(f"({len(hits)} hits; * = score >= threshold {config.knowledge.threshold})")
    if args.trace:
        tracer = LocalTracer()
        store.retriever(k=args.k, where=where).invoke(
            args.query, config=run_config(config.tracing, tracer=tracer, run_name="search")
        )
        print(f"[OK] trace saved: {save_trace(tracer, trace_path(config.paths.runtime_dir, 'search'))}")
    return 0


#==========================================================================
# trace-show
#==========================================================================
def trace_show(config: Config, args: argparse.Namespace) -> int:
    """
    Print a saved trace as a tree, one line per run.

    Input:
        . config        : unused
        . args          : path of the trace file
    Output:
        . code          : 0
    """
    print("\n".join(tree_lines(load_trace(args.path))))
    return 0


#==========================================================================
# scan-keys
#==========================================================================
def scan_keys(config: Config, args: argparse.Namespace) -> int:
    """
    Refuse to pass when a file that would be submitted holds a key.

    Input:
        . config        : the project root
        . args          : unused
    Output:
        . code          : 0 when clean, 1 when something looks like a key
    """
    found = find_keys(config.paths.root)
    for item in found:
        print(f"[NO] {item}")
    print("[OK] no API key in the project files" if not found else
          f"[NO] {len(found)} key-like text(s): remove them, keep keys in .env only")
    return 1 if found else 0


#==========================================================================
# flow-show
#==========================================================================
def flow_show(config: Config, args: argparse.Namespace) -> int:
    """
    Read an exported flow file: components, settings, edges, secrets. No
    server needed.

    Input:
        . config        : unused
        . args          : path of the flow JSON
    Output:
        . code          : 0, or 1 when a password field holds a value
    """
    summary = read_flow(json.loads(args.path.read_text(encoding="utf-8")))
    print("\n".join(_flow_lines(summary)))
    return 1 if summary.secrets else 0


#==========================================================================
# flow-ask
#==========================================================================
def flow_ask(config: Config, args: argparse.Namespace) -> int:
    """
    Ask the Langflow flow one question over its API; --k changes the search
    component's k for this run only (a tweak).

    Input:
        . config        : Langflow address and flow name
        . args          : question, k, session
    Output:
        . code          : 0
    """
    load_env_file(config.paths.root / ".env")
    client  = LangflowClient(config.servers.langflow_url, timeout=config.servers.request_timeout)
    flow_id = client.find_flow(config.servers.flow_name)
    tweaks  = None
    if args.k:
        fields = FlowFields()
        tweaks = tweak(client.get_flow(flow_id), kind=fields.search_kind, field=fields.k_field, value=args.k)
    print(client.run_flow(flow_id, args.question, tweaks=tweaks, session_id=args.session))
    return 0


#==========================================================================
# compare
#==========================================================================
def compare(config: Config, args: argparse.Namespace) -> int:
    """
    Compare two saved experiments case by case.

    Input:
        . config        : the results folder
        . args          : before, after (experiment names)
    Output:
        . code          : 0
    """
    before = load_experiment(config.paths.results_dir, args.before)
    after  = load_experiment(config.paths.results_dir, args.after)
    result = compare_experiments(before, after)
    print(f"{before.name} ({before.backend}) -> {after.name} ({after.backend})")
    print("\n".join(result.lines))
    if result.only_before or result.only_after:
        print(f"not compared: only in {before.name} {result.only_before}, only in {after.name} {result.only_after}")
    return 0


#==========================================================================
# Parse --where key=value
#==========================================================================
def _where(items: list[str] | None) -> dict[str, object] | None:
    """
    Turn key=value pairs into a Chroma filter; two or more are joined with
    $and. Numbers and true/false are converted.

    Input:
        . items         : ["lang=vi", "year=2026"], or None
    Output:
        . where         : a Chroma where filter, or None
    """
    if not items:
        return None
    pairs = []
    for item in items:
        key, sep, text = item.partition("=")
        if not sep:
            raise ValueError(f"--where needs key=value, got {item!r}")
        pairs.append({key: json.loads(text) if text in ("true", "false") or text.lstrip("-").isdigit() else text})
    return pairs[0] if len(pairs) == 1 else {"$and": pairs}


#==========================================================================
# Lines of a flow summary
#==========================================================================
def _flow_lines(summary: FlowSummary) -> list[str]:
    """
    Write a flow summary for reading in the terminal.

    Input:
        . summary       : from read_flow
    Output:
        . lines         : components with settings, edges, the secrets check
    """
    lines = [f"flow {summary.name!r}: {len(summary.components)} components, {len(summary.edges)} edges", ""]
    for component in summary.components:
        lines.append(f"[{component.name}]  {component.kind}  id {component.node_id}")
        for key, value in component.settings.items():
            flat = " ".join(str(value).split())
            lines.append(f"    {key:<24} {flat if len(flat) <= 70 else flat[:67] + '...'}")
    lines += ["", "edges (output -> input):", *(f"    {edge}" for edge in summary.edges), ""]
    if summary.secrets:
        lines.append(f"[NO] a password field holds a value: {', '.join(summary.secrets)} - do NOT share this file")
    else:
        lines.append("[OK] no password field holds a value")
    return lines
