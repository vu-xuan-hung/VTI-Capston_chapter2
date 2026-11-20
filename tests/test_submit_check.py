"""submit-check: the list of what a submission must have."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  shutil
from    pathlib                         import  Path

from    factdesk.config                 import  Config, PathConfig
from    factdesk.evaluation.experiment  import  Experiment, save_experiment
from    factdesk.runtime.submit_check   import  submit_checks
from    helpers                         import  FIXTURES


#==========================================================================
# A small project folder
#==========================================================================
def make_project(tmp_path: Path, *, backend: str, report: str, stop: str) -> Config:
    """
    Build a folder with every item of a submission, then spoil what the test asks.

    Input:
        . tmp_path      : temporary folder
        . backend       : the backend recorded in the saved results
        . report        : the text of REPORT.md
        . stop          : a source line, e.g. a leftover NotImplementedError("T03") or a comment
    Output:
        . config        : paths pointing into the folder
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "code.py").write_text(f"{stop}\n", encoding="utf-8")
    (tmp_path / "my_work").mkdir()
    (tmp_path / "my_work" / "notes.toml").write_text('[a]\nb = "filled"\n', encoding="utf-8")
    (tmp_path / "REPORT.md").write_text(report, encoding="utf-8")
    (tmp_path / "flows").mkdir()
    shutil.copy(FIXTURES / "flows" / "company_x_rag.json", tmp_path / "flows" / "factdesk.json")
    config = Config(paths=PathConfig(
        root=tmp_path, results_dir=tmp_path / "results", flows_dir=tmp_path / "flows", my_work_dir=tmp_path / "my_work",
    ))
    for name in ("v1", "v2"):
        row = {"id": "D01", "split": "dev", "kind": "value", "seconds": 1.0, "model_calls": 1, "error": None, "x": 1.0}
        save_experiment(Experiment(name, backend, [row], {"scores": {"x": 1.0}}), config.paths.results_dir)
    (config.paths.results_dir / "flow_compare.json").write_text("{}", encoding="utf-8")
    return config


#==========================================================================
# All good, and each way to fail
#==========================================================================
def test_a_complete_submission_passes_and_each_gap_is_named(tmp_path: Path) -> None:
    """
    Pin submit_checks: everything present passes; a leftover stop, an unfilled
    report, fake results and a missing flow each produce one [NO] line.

    Input:
        . tmp_path      : temporary folder
    """
    good = make_project(tmp_path, backend="local:Qwen", report="# Report\nfilled\n", stop="x = 1")
    assert all(passed for passed, _ in submit_checks(good))
    other = tmp_path / "bad"
    other.mkdir()
    bad   = make_project(other, backend="fake", report="# Report\n[TODO T09: write]\n", stop='raise NotImplementedError("T03: link")')
    lines = {line.split(":")[0]: passed for passed, line in submit_checks(bad)}
    assert lines["stops T01-T08"] is False and lines["REPORT.md"] is False and lines["results/v1.json"] is False
    shutil.rmtree(bad.paths.flows_dir)
    bad.paths.flows_dir.mkdir()
    assert any(not passed and line.startswith("flows/") for passed, line in submit_checks(bad))
