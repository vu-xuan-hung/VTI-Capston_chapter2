"""PROJECT COMMANDS of FactDesk: ask, resume, define, link, eval, flow-compare, submit-check.

main.py merges these with the kit commands (index, search, trace-show, compare, ...).
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  argparse
import  json
import  sys
import  uuid
from    collections.abc                 import  Callable
from    dataclasses                     import  asdict

from    factdesk.config                        import  Config
from    factdesk.evaluation.dataset            import  load_dataset
from    factdesk.evaluation.evaluators         import  all_evaluators
from    factdesk.evaluation.experiment         import  Experiment, run_experiment, save_experiment
from    factdesk.evaluation.targets            import  make_target
from    factdesk.integrations.flow_compare     import  compare_answers
from    factdesk.integrations.langflow_client  import  LangflowClient
from    factdesk.knowledge.rag_slice           import  answer_from_notes
from    factdesk.observability.cloud           import  enable_tracing, load_env_file
from    factdesk.observability.local_tracer    import  LocalTracer
from    factdesk.observability.trace_config    import  case_config
from    factdesk.observability.trace_view      import  save_trace, trace_path
from    factdesk.runtime.desk                  import  Desk, open_checkpointer
from    factdesk.runtime.submit_check          import  submit_checks


type Command = Callable[[Config, argparse.Namespace], int]


#==========================================================================
# Declare the project's arguments
#==========================================================================
def add_project_parsers(commands: argparse._SubParsersAction) -> None:
    """
    Add the project's subcommands to main.py's parser.

    Input:
        . commands      : the subparsers object of main.py
    """

    #----------------------------------------------------------------------
    # Ask and resume
    #----------------------------------------------------------------------
    ask = commands.add_parser("ask", help="one question, the JSON contract (v2 graph by default)")
    ask.add_argument("question")
    ask.add_argument("--version", choices=["v1", "v2"], default="v2", help="v1 = free agent, v2 = graph")
    ask.add_argument("--thread", help="checkpoint thread id (v2); default: a new one")
    resume = commands.add_parser("resume", help="answer a clarification: continue an interrupted v2 run")
    resume.add_argument("--thread", required=True, help="the thread id printed by ask")
    resume.add_argument("--choice", required=True, help="the country code you mean, e.g. KOR")

    #----------------------------------------------------------------------
    # Definitions and links
    #----------------------------------------------------------------------
    define = commands.add_parser("define", help="what does a number mean? (the notes chain, the Langflow slice)")
    define.add_argument("question")
    link = commands.add_parser("link", help="link one country name to a code (stop T03)")
    link.add_argument("text")

    #----------------------------------------------------------------------
    # Evaluation, Langflow comparison and the submission check
    #----------------------------------------------------------------------
    run_eval = commands.add_parser("eval", help="run a version on the cases and save results/<version>.json")
    run_eval.add_argument("version", choices=["v1", "v2"])
    run_eval.add_argument("--split", choices=["dev", "holdout"], default="dev")
    run_eval.add_argument("--cloud", action="store_true", help="run on LangSmith (needs LANGSMITH_API_KEY)")
    commands.add_parser("flow-compare", help="ask the code and the Langflow flow the same questions")
    commands.add_parser("submit-check", help="is the submission complete? [OK] or [NO] per item")


#==========================================================================
# ask
#==========================================================================
def ask(config: Config, args: argparse.Namespace) -> int:
    """
    Answer one question with v1 or v2, print the JSON contract and save a trace.

    Input:
        . config        : the whole configuration
        . args          : question, version, thread
    Output:
        . code          : 0
    """
    load_env_file(config.paths.root / ".env")
    desk   = Desk(config, checkpointer=open_checkpointer(config.paths.runtime_dir))
    tracer = LocalTracer()
    thread = args.thread or f"ask-{uuid.uuid4().hex[:8]}"
    run    = case_config(config.tracing, tracer=tracer, version=args.version, case_id="ask", backend=config.model.backend)
    answer = desk.ask(args.question, thread_id=thread, run_config=run) if args.version == "v2" \
        else desk.ask_agent(args.question, run_config=run)
    print(json.dumps(answer.to_contract(), indent=2, ensure_ascii=False))
    print(f"trace: {save_trace(tracer, trace_path(config.paths.runtime_dir, 'ask'))}", file=sys.stderr)
    if answer.status == "needs_clarification":
        print(f"[?] resume with: python main.py resume --thread {thread} --choice <CODE>", file=sys.stderr)
    return 0


#==========================================================================
# resume
#==========================================================================
def resume(config: Config, args: argparse.Namespace) -> int:
    """
    Continue an interrupted v2 run in this NEW process: the checkpoint file
    holds the state, the choice goes into the clarify node.

    Input:
        . config        : the whole configuration
        . args          : thread, choice
    Output:
        . code          : 0
    """
    desk   = Desk(config, checkpointer=open_checkpointer(config.paths.runtime_dir))
    tracer = LocalTracer()
    run    = case_config(config.tracing, tracer=tracer, version="v2", case_id="resume", backend=config.model.backend)
    answer = desk.resume(thread_id=args.thread, choice=args.choice, run_config=run)
    print(json.dumps(answer.to_contract(), indent=2, ensure_ascii=False))
    print(f"trace: {save_trace(tracer, trace_path(config.paths.runtime_dir, 'resume'))}", file=sys.stderr)
    return 0


#==========================================================================
# define
#==========================================================================
def define(config: Config, args: argparse.Namespace) -> int:
    """
    Answer a "what does this number mean" question from the note chunks.

    Input:
        . config        : the whole configuration
        . args          : question
    Output:
        . code          : 0
    """
    desk         = Desk(config)
    text, chunks = answer_from_notes(args.question, store=desk.store, chat=desk.chat.get(), rag=config.rag)
    print(text)
    print(f"(notes used: {', '.join(chunks) or 'none above the threshold'})")
    return 0


#==========================================================================
# link
#==========================================================================
def link(config: Config, args: argparse.Namespace) -> int:
    """
    Show how one country name is linked: status, method and candidates with scores.

    Input:
        . config        : the whole configuration
        . args          : text
    Output:
        . code          : 0
    """
    found = Desk(config).link(args.text)
    print(f"{found.text!r}: {found.status} {found.code or ''} (method {found.method})")
    for item in found.candidates:
        print(f"    {item.score:.3f}  {item.code}  {item.name}")
    return 0


#==========================================================================
# eval
#==========================================================================
def run_eval(config: Config, args: argparse.Namespace) -> int:
    """
    Run one version on the dev cases (or the hold-out cases once), score it with
    the four code evaluators and save results/<version>[_holdout].json and .csv.
    --cloud runs the same target and evaluators on LangSmith instead.

    Input:
        . config        : the whole configuration
        . args          : version, split, cloud
    Output:
        . code          : 0
    """

    #----------------------------------------------------------------------
    # Cases, desk and the index must exist
    #----------------------------------------------------------------------
    load_env_file(config.paths.root / ".env")
    path = config.paths.cases_path if args.split == "dev" else config.paths.holdout_path
    if not path.exists():
        raise ValueError(f"{path} not found: the hold-out cases belong to the instructor")
    desk = Desk(config)
    if desk.store.count() == 0:
        raise ValueError("the entity collection is empty: run `python main.py index` first")
    cases  = load_dataset(path)
    target = make_target(args.version, desk)
    name   = args.version if args.split == "dev" else f"{args.version}_holdout"

    #----------------------------------------------------------------------
    # Run locally or on LangSmith
    #----------------------------------------------------------------------
    if args.cloud:
        from    factdesk.evaluation.cloud_experiment  import  run_cloud_experiment

        enable_tracing(config.tracing.project)
        online = run_cloud_experiment(
            name, target, cases, all_evaluators(),
            dataset_name    = f"{config.tracing.dataset}-{args.split}",
            metadata        = {"version": args.version, "backend": config.model.backend},
        )
        print(f"[OK] LangSmith experiment: {online}")
        return 0
    backend    = config.model.backend if config.model.backend == "fake" else f"{config.model.backend}:{config.model.chat_model}"
    experiment = run_experiment(name, target, cases, all_evaluators(), backend=backend)

    json_path, _ = save_experiment(experiment, config.paths.results_dir)
    print("\n".join(summary_lines(experiment)))
    print(f"[OK] saved {json_path}")
    return 0


#==========================================================================
# Lines of an experiment
#==========================================================================
def summary_lines(experiment: Experiment) -> list[str]:
    """
    Write one line per case (its failed metrics) and the mean of each metric.

    Input:
        . experiment    : a finished experiment
    Output:
        . lines         : text for the terminal
    """
    metrics = list(experiment.summary["scores"])
    lines   = [f"{experiment.name} on {experiment.summary['cases']} cases, backend {experiment.backend}", ""]
    for row in experiment.rows:
        failed = [metric for metric in metrics if row[metric] < 1] + (["ERROR " + row["error"]] if row["error"] else [])
        lines.append(f"{row['id']:<5}{row['kind']:<12}{row['seconds']:>7.1f}s {row['model_calls']} calls  "
                     f"{'ok' if not failed else 'FAILED: ' + ', '.join(failed)}")
    lines += ["", *(f"{metric:<26}{score:.3f}" for metric, score in experiment.summary["scores"].items())]
    lines.append(f"seconds {experiment.summary['seconds']}  model calls {experiment.summary['model_calls']}  errors {experiment.summary['errors']}")
    return lines


#==========================================================================
# flow-compare
#==========================================================================
def flow_compare(config: Config, args: argparse.Namespace) -> int:
    """
    Ask the code and the Langflow flow the questions of data/flow_questions.json.
    Definition questions: the code is the notes chain. The numeric question:
    the code is the v2 graph, and the flow (no tool, no code step) is checked
    against the facts the graph computed. Saves results/flow_compare.json.

    Input:
        . config        : the whole configuration
        . args          : unused
    Output:
        . code          : 0
    """

    #----------------------------------------------------------------------
    # Connect to the code and the flow
    #----------------------------------------------------------------------
    load_env_file(config.paths.root / ".env")
    desk      = Desk(config)
    client    = LangflowClient(config.servers.langflow_url, timeout=config.servers.request_timeout)
    flow_id   = client.find_flow(config.servers.flow_name)
    questions = json.loads((config.paths.data_dir / "flow_questions.json").read_text(encoding="utf-8"))
    rows      = []

    #----------------------------------------------------------------------
    # Ask both sides each question
    #----------------------------------------------------------------------
    for item in questions:
        facts = result = None
        years: list[int] = []
        if item["numeric"]:
            answer = desk.ask(item["question"], thread_id=f"flow-{item['id']}")

            code_text, facts, result, years = answer.answer, answer.to_contract()["facts"], answer.to_contract()["result"], answer.years
        else:
            code_text, _ = answer_from_notes(item["question"], store=desk.store, chat=desk.chat.get(), rag=config.rag)
        flow_text  = client.run_flow(flow_id, item["question"])
        comparison = compare_answers(
            code_text, flow_text,
            expected_terms  = item["expected_terms"],
            facts           = facts,
            result          = result,
            years           = years,
            numbers         = config.numbers,
        )
        rows.append({"id": item["id"], "question": item["question"], "code": code_text, "flow": flow_text, **asdict(comparison)})
        print(f"{item['id']}  {item['question']}\n    code: {' '.join(code_text.split())[:150]}\n    flow: {' '.join(flow_text.split())[:150]}\n    {asdict(comparison)}")

    #----------------------------------------------------------------------
    # Save the comparison
    #----------------------------------------------------------------------
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.results_dir / "flow_compare.json").write_text(
        json.dumps({"backend": config.model.backend, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print("[OK] saved results/flow_compare.json")
    return 0


#==========================================================================
# submit-check
#==========================================================================
def submit_check(config: Config, args: argparse.Namespace) -> int:
    """
    Print one [OK] or [NO] line per item of a submission and a final verdict.

    Input:
        . config        : the whole configuration
        . args          : unused
    Output:
        . code          : 0 when everything is [OK], else 1
    """
    checks = submit_checks(config)
    for passed, line in checks:
        print(f"[{'OK' if passed else 'NO'}] {line}")
    done = all(passed for passed, _ in checks)
    print(f"submit-check: [{'OK' if done else 'NO'}]")
    return 0 if done else 1


#==========================================================================
# Map the project's command names to functions
#==========================================================================
def project_commands() -> dict[str, Command]:
    """
    Return the project's command table, merged into main.py's.

    Output:
        . commands      : name -> function(config, args) -> exit code
    """
    return {
        "ask": ask,
        "resume": resume,
        "define": define,
        "link": link,
        "eval": run_eval,
        "flow-compare": flow_compare,
        "submit-check": submit_check,
    }
