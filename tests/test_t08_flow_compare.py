"""T08: compare_answers, and flow-compare against the fake Langflow (run: pytest -k t08)."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
import  re
from    pathlib                         import  Path

import  pytest
from    fake_langflow                   import  FakeLangflow

from    factdesk.config                     import  FormatConfig
from    factdesk.integrations.flow_compare  import  compare_answers
from    helpers                             import  project_toml
from    main                                import  main


FACTS = [{"code": "VNM", "year": 2015, "indicator": "population", "value": 92823254}]


#==========================================================================
# The comparison
#==========================================================================
def test_t08_compare_answers_finds_missing_terms_and_unverified_numbers() -> None:
    """
    Pin T08: missing expected terms (ignoring case) for both answers; numbers are checked
    against the facts only when facts are given.
    """
    both = compare_answers(
        "GDP is the value of goods and services.", "It is a number.",
        expected_terms=["goods and services"], facts=None, result=None, years=[], numbers=FormatConfig(),
    )
    assert both.code_missing_terms == [] and both.flow_missing_terms == ["goods and services"]
    assert both.code_unverified == [] and both.flow_unverified == []
    numeric = compare_answers(
        "Viet Nam had 92,823,254 people in 2015.", "Viet Nam had about 90 million people in 2015.",
        expected_terms=[], facts=FACTS, result=None, years=[2015], numbers=FormatConfig(),
    )
    assert numeric.code_unverified == [] and numeric.flow_unverified == ["90 million"]


#==========================================================================
# The command, with definition questions only
#==========================================================================
def test_t08_flow_compare_command_saves_one_row_per_question(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Pin T08 through the command: the code chain (notes) and the fake Langflow answer the same
    definition questions and the comparison is saved in results/flow_compare.json.

    Input:
        . tmp_path      : temporary folder
        . capsys        : captured output
        . monkeypatch   : sets LANGFLOW_API_KEY for this test only
    """
    #----------------------------------------------------------------------
    # A fake Langflow and a question file with definitions only
    #----------------------------------------------------------------------
    monkeypatch.setenv("LANGFLOW_API_KEY", "test-value")
    langflow = FakeLangflow()
    data     = tmp_path / "data"
    data.mkdir()
    (data / "flow_questions.json").write_text(json.dumps([
        {"id": "F1", "question": "What does GDP mean?", "expected_terms": ["flow"], "numeric": False},
    ]), encoding="utf-8")
    config = project_toml(tmp_path)
    text   = config.read_text(encoding="utf-8").replace('data_dir = "data"', f'data_dir = {json.dumps(data.as_posix(), ensure_ascii=False)}')
    text   = re.sub(r'^langflow_url = ".*?"', f'langflow_url = "{langflow.url}"', text, flags=re.MULTILINE)
    config.write_text(text, encoding="utf-8")
    #----------------------------------------------------------------------
    # Index, compare, read the saved rows
    #----------------------------------------------------------------------
    assert main(["--config", str(config), "--fake", "index"]) == 0
    capsys.readouterr()
    code = main(["--config", str(config), "--fake", "flow-compare"])
    langflow.close()
    saved = json.loads((tmp_path / "results" / "flow_compare.json").read_text(encoding="utf-8"))
    assert code == 0 and saved["backend"] == "fake" and saved["rows"][0]["id"] == "F1"
    assert saved["rows"][0]["flow"] == langflow.answer and saved["rows"][0]["flow_missing_terms"] == []
    assert saved["rows"][0]["code_missing_terms"] == ["flow"]
