"""Run a target over a dataset with evaluators: the local langsmith.evaluate.

An evaluator is a plain function with the signature langsmith 0.14 accepts
(checked in langsmith/evaluation/evaluator.py, _normalize_evaluator_func):
its parameters are NAMED from (inputs, outputs, reference_outputs) - also
run, example, attachments on LangSmith - and it returns a bool, a number, or
{"score": ..., "comment": ...}. The metric name is the function name. The
same function therefore runs here and in cloud_experiment.py unchanged:

    def ev_has_source(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        return reference_outputs["source_id"] in outputs["sources"]
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  csv
import  inspect
import  json
import  time
from    collections.abc                 import  Callable
from    dataclasses                     import  asdict, dataclass
from    pathlib                         import  Path
from    typing                          import  Any

from    langchain_core.tracers.context  import  collect_runs

from    factdesk.evaluation.dataset     import  Example


#==========================================================================
# One experiment
#==========================================================================
@dataclass(frozen=True)
class Experiment:
    """
    Purpose: one version run on one dataset: a row per case (outputs, error,
    seconds, model calls, one score per evaluator) and the summary.
    """

    name: str
    backend: str
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


#==========================================================================
# Run an experiment
#==========================================================================
def run_experiment(
    name: str,
    target: Callable[[dict[str, Any]], dict[str, Any]],
    dataset: list[Example],
    evaluators: list[Callable[..., Any]],
    *,
    backend: str,
) -> Experiment:
    """
    Call target(inputs) on every case and score it. A crash is a result too:
    its error is kept and it scores 0 on every evaluator.

    Input:
        . name          : version name, e.g. "v1"; also the results file name
        . target        : inputs dict -> outputs dict (the app under test)
        . dataset       : the cases
        . evaluators    : LangSmith-style evaluator functions
        . backend       : what answered ("local", "server", "fake"), recorded
    Output:
        . experiment    : rows and summary; save it with save_experiment
    """
    started = time.perf_counter()
    rows    = [_run_case(example, target, evaluators) for example in dataset]
    metrics = [evaluator.__name__ for evaluator in evaluators]
    summary = {
        "cases": len(rows),
        "errors": sum(row["error"] is not None for row in rows),
        "seconds": round(time.perf_counter() - started, 2),
        "mean_seconds": round(sum(row["seconds"] for row in rows) / len(rows), 2),
        "model_calls": sum(row["model_calls"] for row in rows),
        "scores": {metric: round(sum(row[metric] for row in rows) / len(rows), 3) for metric in metrics},
    }
    return Experiment(name=name, backend=backend, rows=rows, summary=summary)


#==========================================================================
# One case
#==========================================================================
def _run_case(
    example: Example,
    target: Callable[[dict[str, Any]], dict[str, Any]],
    evaluators: list[Callable[..., Any]],
) -> dict[str, Any]:
    """
    Run one case, time it, count its chat model calls and score it.

    Input:
        . example       : one case
        . target        : the app under test
        . evaluators    : evaluator functions
    Output:
        . row           : id, split, seconds, model_calls, error, inputs,
                          outputs, reference_outputs, one score per evaluator
    """

    #----------------------------------------------------------------------
    # Run the target; collect_runs sees every LangChain run inside it
    #----------------------------------------------------------------------
    started = time.perf_counter()
    error   = None
    with collect_runs() as collector:
        try:
            outputs = target(example.inputs)
        except NotImplementedError:
            raise
        except (RuntimeError, ValueError, KeyError, OSError) as failure:
            outputs = {}
            error   = f"{type(failure).__name__}: {failure}"
    seconds = round(time.perf_counter() - started, 2)

    #----------------------------------------------------------------------
    # Score it
    #----------------------------------------------------------------------
    row: dict[str, Any] = {
        "id": example.id,
        "split": example.split,
        "kind": example.kind,
        "seconds": seconds,
        "model_calls": sum(_llm_runs(run) for run in collector.traced_runs),
        "error": error,
        "inputs": example.inputs,
        "outputs": outputs,
        "reference_outputs": example.outputs,
    }
    for evaluator in evaluators:
        score = 0.0 if error else _score(_call(evaluator, example, outputs), evaluator.__name__)
        row[evaluator.__name__] = score
    return row


#==========================================================================
# Call an evaluator by its parameter names
#==========================================================================
def _call(evaluator: Callable[..., Any], example: Example, outputs: dict[str, Any]) -> Any:
    """
    Pass each parameter by name, as langsmith does; run/example objects
    exist only on LangSmith, so asking for them here stops with a message.

    Input:
        . evaluator     : the evaluator function
        . example       : the case (inputs and reference outputs)
        . outputs       : what the target returned
    Output:
        . result        : the evaluator's raw result
    """
    values = {"inputs": example.inputs, "outputs": outputs, "reference_outputs": example.outputs}
    params = list(inspect.signature(evaluator).parameters)
    other  = [param for param in params if param not in values]
    if other:
        raise ValueError(
            f"evaluator {evaluator.__name__}: parameters {other} are not available locally; "
            "use inputs, outputs, reference_outputs"
        )
    return evaluator(**{param: values[param] for param in params})


#==========================================================================
# Normalise an evaluator result
#==========================================================================
def _score(result: Any, name: str) -> float:
    """
    Turn a bool, a number or {"score": ...} into a float, like langsmith.

    Input:
        . result        : the evaluator's raw result
        . name          : evaluator name, for the error message
    Output:
        . score         : 1.0 / 0.0 for a bool, else the number
    """
    if isinstance(result, dict) and "score" in result:
        result = result["score"]
    if isinstance(result, bool | int | float):
        return float(result)
    raise ValueError(f"evaluator {name} returned {result!r}: use a bool, a number or {{'score': ...}}")


#==========================================================================
# Count chat model runs in a run tree
#==========================================================================
def _llm_runs(run: Any) -> int:
    """
    Count the runs of type "llm" in one collected root run and its children.

    Input:
        . run           : a langsmith Run from collect_runs
    Output:
        . count         : number of model calls
    """
    return int(run.run_type == "llm") + sum(_llm_runs(child) for child in run.child_runs)


#==========================================================================
# Save an experiment
#==========================================================================
def save_experiment(experiment: Experiment, folder: Path) -> tuple[Path, Path]:
    """
    Write results/<name>.json (everything) and results/<name>.csv (one line
    per case; inputs and outputs as JSON text). Same name = overwritten.

    Input:
        . experiment    : from run_experiment
        . folder        : the results folder
    Output:
        . json_path     : the full file, read by load_experiment
        . csv_path      : the table, for a spreadsheet
    """
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f"{experiment.name}.json"
    csv_path  = folder / f"{experiment.name}.csv"
    json_path.write_text(json.dumps(asdict(experiment), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(experiment.rows[0]))
        writer.writeheader()
        for row in experiment.rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
                for key, value in row.items()
            })
    return json_path, csv_path


#==========================================================================
# Read a saved experiment
#==========================================================================
def load_experiment(folder: Path, name: str) -> Experiment:
    """
    Read results/<name>.json, for compare in a later process.

    Input:
        . folder        : the results folder
        . name          : the experiment name
    Output:
        . experiment    : as it was saved
    """
    path = folder / f"{name}.json"
    if not path.exists():
        saved = sorted(item.stem for item in folder.glob("*.json"))
        raise ValueError(f"no experiment {name!r} in {folder}; saved: {saved}")
    return Experiment(**json.loads(path.read_text(encoding="utf-8")))
