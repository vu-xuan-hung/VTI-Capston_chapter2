"""main.py with --fake: index twice, search with a filter and a trace, checks."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    pathlib                         import  Path

import  pytest
from    fake_langflow                   import  FLOW_NAME, FakeLangflow
from    helpers                         import  FIXTURES

from    main                            import  main


#==========================================================================
# A project folder with its own config
#==========================================================================
@pytest.fixture
def project(tmp_path: Path) -> Path:
    """
    Write two documents and a config whose paths all point into tmp_path.

    Input:
        . tmp_path      : temporary project folder
    Output:
        . config_path   : the TOML file to pass with --config
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "hours.md").write_text("# Opening hours\nThe office opens at eight and closes at five.\n", encoding="utf-8")
    (docs / "refund.md").write_text("# Refunds\nA refund is paid within fourteen days.\n", encoding="utf-8")

    #----------------------------------------------------------------------
    # Keep every path inside the fixture on Windows and POSIX
    #----------------------------------------------------------------------
    root = tmp_path.as_posix()
    config = tmp_path / "app.toml"
    config.write_text(
        "[knowledge]\ncollection = \"cli-kb\"\nchunk_size = 8\nchunk_overlap = 2\nthreshold = 0.1\n"
        "[paths]\n"
        f'root = "{root}"\n'
        f'docs_dir = "{root}/docs"\n'
        f'runtime_dir = "{root}/runtime"\n'
        f'results_dir = "{root}/results"\n'
        f'kb_dir = "{root}/kb_store"\n',
        encoding="utf-8",
    )
    return config


#==========================================================================
# index twice
#==========================================================================
def test_index_twice_reports_no_change(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Pin the index command and its report on a second run.

    Input:
        . project       : the config path
        . capsys        : captured output
    """
    assert main(["--config", str(project), "--fake", "index"]) == 0
    assert "added 4" in capsys.readouterr().out
    assert main(["--config", str(project), "--fake", "index"]) == 0
    assert "added 0, changed 0, removed 0, unchanged 4" in capsys.readouterr().out


#==========================================================================
# search with a filter, saved trace, trace-show
#==========================================================================
def test_search_filters_and_saves_a_trace_that_prints(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Pin search --where and --trace, then trace-show on the saved file.

    Input:
        . project       : the config path
        . capsys        : captured output
    """
    main(["--config", str(project), "--fake", "index"])
    capsys.readouterr()
    code = main(["--config", str(project), "--fake", "search", "refund days", "--where", "file=refund.md", "--trace"])
    out  = capsys.readouterr().out
    assert code == 0
    assert "[refund#001]" in out and "[hours#" not in out
    trace = out.split("trace saved: ")[1].strip()
    assert main(["trace-show", trace]) == 0
    assert capsys.readouterr().out.startswith("retriever search")


#==========================================================================
# scan-keys and flow-show
#==========================================================================
def test_checks_report_ok_and_no(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Pin the exit codes of scan-keys and flow-show.

    Input:
        . project       : the config path
        . capsys        : captured output
    """
    assert main(["--config", str(project), "scan-keys"]) == 0
    (project.parent / "notes.md").write_text("token " + "hf_" + "x" * 30, encoding="utf-8")
    assert main(["--config", str(project), "scan-keys"]) == 1
    assert main(["flow-show", str(FIXTURES / "flows" / "company_x_rag.json")]) == 0
    assert "[OK] no password field holds a value" in capsys.readouterr().out


#==========================================================================
# A user error is one line, exit code 2
#==========================================================================
def test_a_missing_experiment_is_one_line(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Pin that main turns a ValueError into "Cannot do this" and exit code 2.

    Input:
        . project       : the config path
        . capsys        : captured output
    """
    assert main(["--config", str(project), "compare", "v1", "v2"]) == 2
    assert "Cannot do this" in capsys.readouterr().err


#==========================================================================
# flow-ask with a tweak, against the fake Langflow
#==========================================================================
def test_flow_ask_finds_the_flow_and_sends_the_k_tweak(
    project: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Pin flow-ask end to end: find by name, --k as a Chroma tweak, the answer printed.

    Input:
        . project       : the config path
        . capsys        : captured output
        . monkeypatch   : sets LANGFLOW_API_KEY for this test only
    """
    monkeypatch.setenv("LANGFLOW_API_KEY", "test-value")
    langflow = FakeLangflow()
    with project.open("a", encoding="utf-8") as file:
        file.write(f'[servers]\nlangflow_url = "{langflow.url}"\nflow_name = "{FLOW_NAME}"\n')
    code = main(["--config", str(project), "flow-ask", "When does the office open?", "--k", "2"])
    langflow.close()
    assert code == 0
    assert capsys.readouterr().out.strip() == langflow.answer
    assert langflow.requests[-1]["tweaks"] == {"Chroma-b": {"number_of_results": 2}}
