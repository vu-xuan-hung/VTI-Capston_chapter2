"""submit-check: the list of things a submission must have, one [OK] or [NO] line each."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  ast
import  json
import  re
import  tomllib
from    pathlib                         import  Path

from    factdesk.config                    import  Config
from    factdesk.evaluation.experiment     import  load_experiment
from    factdesk.integrations.flow_reader  import  read_flow
from    factdesk.runtime.key_scan          import  find_keys


#==========================================================================
# All checks
#==========================================================================
def submit_checks(config: Config) -> list[tuple[bool, str]]:
    """
    Run every check of a submission: stops done, notes filled, results from a
    real model, a flow file without secrets, a filled report, no key in a file.

    Input:
        . config        : paths
    Output:
        . checks        : (passed, one line) per check, in the order of the README
    """

    #----------------------------------------------------------------------
    # Stops: no TODO raise left in the code
    #----------------------------------------------------------------------
    root   = config.paths.root
    left   = sorted(_unfinished_stops(root / "src"))
    checks = [(not left, f"stops T01-T08: {'all done' if not left else 'not done: ' + ', '.join(left)}")]

    #----------------------------------------------------------------------
    # Notes of T07 and the report of T09
    #----------------------------------------------------------------------
    notes   = config.paths.my_work_dir / "notes.toml"
    empty   = _empty_values(tomllib.loads(notes.read_text(encoding="utf-8"))) if notes.exists() else ["(file missing)"]
    checks.append((not empty, f"my_work/notes.toml: {'filled' if not empty else 'empty: ' + ', '.join(empty)}"))
    report  = root / "REPORT.md"
    todo    = report.read_text(encoding="utf-8").count("[TODO") if report.exists() else -1
    checks.append((todo == 0, f"REPORT.md: {'filled' if todo == 0 else ('missing' if todo < 0 else f'{todo} [TODO] left')}"))

    #----------------------------------------------------------------------
    # Results from a real model, and the flow
    #----------------------------------------------------------------------
    for name in ("v1", "v2"):
        path = config.paths.results_dir / f"{name}.json"
        if not path.exists():
            checks.append((False, f"results/{name}.json: missing (run: python main.py eval {name})"))
            continue
        backend = load_experiment(config.paths.results_dir, name).backend
        checks.append((not backend.startswith("fake"), f"results/{name}.json: backend {backend}"))
    flows = sorted(path for path in config.paths.flows_dir.glob("*.json") if path.name != "FLOW_SPEC.json")
    if flows:
        summary = read_flow(json.loads(flows[0].read_text(encoding="utf-8")))
        checks.append((not summary.secrets, f"flow {flows[0].name}: {len(summary.components)} components, secrets {summary.secrets or 'none'}"))
    else:
        checks.append((False, "flows/: no exported flow (export it from Langflow into flows/)"))
    checks.append(((config.paths.results_dir / "flow_compare.json").exists(), "results/flow_compare.json"))

    #----------------------------------------------------------------------
    # Keys
    #----------------------------------------------------------------------
    found = find_keys(root)
    checks.append((not found, f"scan-keys: {'clean' if not found else str(len(found)) + ' key-like text(s)'}"))
    return checks


#==========================================================================
# Unfinished stops in source code
#==========================================================================
def _unfinished_stops(source_dir: Path) -> set[str]:
    """
    Collect stop IDs from raise statements. Python's parser ignores comments
    and represents quoted examples as text instead of statements.

    Input:
        . source_dir    : the source tree to scan
    Output:
        . stops         : IDs of the remaining exercise placeholders
    """
    stops = set()
    for path in source_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            match node:
                case ast.Raise(exc=ast.Call(
                    func=ast.Name(id="NotImplementedError"),
                    args=[ast.Constant(value=str(message)), *_],
                )):
                    if stop := re.match(r"T0[1-8]\b", message):
                        stops.add(stop[0])
    return stops


#==========================================================================
# Empty values of a TOML table
#==========================================================================
def _empty_values(table: dict[str, object], *, prefix: str = "") -> list[str]:
    """
    List the keys whose value is an empty text, at any depth.

    Input:
        . table         : a parsed TOML table
        . prefix        : the dotted path so far
    Output:
        . keys          : e.g. ["agent_error.case"]
    """
    empty: list[str] = []
    for key, value in table.items():
        if isinstance(value, dict):
            empty += _empty_values(value, prefix=f"{prefix}{key}.")
        elif not str(value).strip():
            empty.append(f"{prefix}{key}")
    return empty
