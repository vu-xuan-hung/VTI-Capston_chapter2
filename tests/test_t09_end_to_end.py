"""T09: everything together on the fake backend (run: pytest -k t09). In a starter repo this fails at the first unfinished stop it reaches."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
import  re
from    dataclasses                     import  replace
from    pathlib                         import  Path

import  pytest
from    fake_langflow                   import  FakeLangflow

from    factdesk.llm.fake_chat_model    import  FakeRules
from    factdesk.llm.fake_rules         import  project_fake_rules
from    factdesk.runtime.desk           import  Desk, open_checkpointer
from    helpers                         import  indexed_desk, project_config, project_toml
from    main                            import  main


#==========================================================================
# Run the whole dev set
#==========================================================================
def test_t09_every_dev_case_ends_in_a_valid_contract_and_v2_passes_on_the_fake_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    Pin the whole path: `eval v2` on the 12 dev cases, saved results with the backend, every
    evaluator at 1.0 (the fake model reads questions by rules), no crashed case.

    Input:
        . tmp_path      : temporary folder
        . capsys        : captured output
    """
    config = str(project_toml(tmp_path))
    assert main(["--config", config, "--fake", "index"]) == 0
    assert main(["--config", config, "--fake", "eval", "v2"]) == 0
    saved = json.loads((tmp_path / "results" / "v2.json").read_text(encoding="utf-8"))
    assert saved["backend"] == "fake" and saved["summary"]["cases"] == 12 and saved["summary"]["errors"] == 0
    assert set(saved["summary"]["scores"].values()) == {1.0}
    assert len(list((tmp_path / "runtime" / "traces").glob("*-v2-D09.json"))) == 1


#==========================================================================
# V1 and v2 compare shows cases that got better
#==========================================================================
def test_t09_v1_and_v2_compare_shows_cases_that_got_better(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Pin `compare v1 v2`: both saved by `eval`, the table lists better cases (the fake agent mis-reads a few questions).

    Input:
        . tmp_path      : temporary folder
        . capsys        : captured output
    """
    config = str(project_toml(tmp_path))
    main(["--config", config, "--fake", "index"])
    assert main(["--config", config, "--fake", "eval", "v1"]) == 0
    assert main(["--config", config, "--fake", "eval", "v2"]) == 0
    capsys.readouterr()
    assert main(["--config", config, "compare", "v1", "v2"]) == 0
    out = capsys.readouterr().out
    assert "v1 (fake) -> v2 (fake)" in out and re.search(r"better: \['D", out)


#==========================================================================
# Interrupt and resume in two processes
#==========================================================================
def test_t09_resume_in_a_new_desk_finishes_the_run_and_writes_the_outbox_once(tmp_path: Path) -> None:
    """
    Pin the human-in-the-loop path: ask stops with needs_clarification, a NEW Desk on the same
    SQLite file resumes the same thread, and the outbox has exactly one line.

    Input:
        . tmp_path      : temporary folder
    """
    first  = indexed_desk(tmp_path, checkpointer=open_checkpointer(tmp_path / "runtime"))
    asked  = first.ask("What was the population of Korea in 2015?", thread_id="t1")
    assert asked.status == "needs_clarification" and "KOR" in asked.reason and asked.facts == []
    config = project_config(tmp_path)
    second = Desk(config, checkpointer=open_checkpointer(tmp_path / "runtime"))
    done   = second.resume(thread_id="t1", choice="KOR")
    assert done.status == "answered" and done.countries == ["KOR"] and done.result.value == 51014947 and done.verified
    lines  = (tmp_path / "runtime" / "outbox.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    with pytest.raises(ValueError, match="nothing to resume"):
        second.resume(thread_id="t1", choice="KOR")


#==========================================================================
# Ask and resume commands use the checkpoint file
#==========================================================================
def test_t09_ask_and_resume_commands_use_the_checkpoint_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Pin the commands: `ask` prints needs_clarification and the thread, `resume` (a separate main call) answers.

    Input:
        . tmp_path      : temporary folder
        . capsys        : captured output
    """
    config = str(project_toml(tmp_path))
    main(["--config", config, "--fake", "index"])
    capsys.readouterr()
    assert main(["--config", config, "--fake", "ask", "Population of Korea in 2015?", "--thread", "T9"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "needs_clarification"
    assert main(["--config", config, "--fake", "resume", "--thread", "T9", "--choice", "PRK"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "answered" and out["countries"] == ["PRK"]


#==========================================================================
# Model writes, code decides
#==========================================================================
def rules_with_text(write) -> FakeRules:
    """
    Take the project's fake rules and replace only the text rule.

    Input:
        . write         : prompt -> text
    Output:
        . rules         : FakeRules
    """
    return replace(project_fake_rules(), text=write)


#==========================================================================
# A wrong number is retried once with the list then accepted
#==========================================================================
def test_t09_a_wrong_number_is_retried_once_with_the_list_then_accepted(tmp_path: Path) -> None:
    """
    Pin verify: the first answer has a number that is not in the table; the retry prompt names it
    and shows the table; the second answer copies the numbers and is accepted.

    Input:
        . tmp_path      : temporary folder
    """
    prompts: list[str] = []

    def write(prompt: str) -> str:
        """
        Write a wrong population first, then the table lines.

        Input:
            . prompt        : the messages of one explain call
        Output:
            . text          : the answer
        """
        prompts.append(prompt)
        return "Viet Nam had 97,338,579 people." if "NOT in the FACTS" not in prompt else project_fake_rules().text(prompt)

    answer = indexed_desk(tmp_path, rules=rules_with_text(write)).ask("What was Vietnam's population in 2015?", thread_id="t")
    assert len(prompts) == 2 and "97,338,579" in prompts[1] and "92,823,254" in prompts[1]
    assert answer.status == "answered" and answer.verified and answer.reason == "model answer verified after a retry"
    assert answer.facts[0].value == 92823254


#==========================================================================
# A model that never copies the numbers gets the template answer
#==========================================================================
def test_t09_a_model_that_never_copies_the_numbers_gets_the_template_answer(tmp_path: Path) -> None:
    """
    Pin the last line of defence: two wrong answers, then the answer written by code; the
    contract numbers are the table's, reason says "template".

    Input:
        . tmp_path      : temporary folder
    """
    calls: list[str] = []

    def write(prompt: str) -> str:
        """
        Always write a number that is not in the table.

        Input:
            . prompt        : the messages of one explain call
        Output:
            . text          : the wrong answer
        """
        calls.append(prompt)
        return "Viet Nam had 1,000 people."

    answer = indexed_desk(tmp_path, rules=rules_with_text(write)).ask("What was Vietnam's population in 2015?", thread_id="t")
    assert len(calls) == 2 and answer.status == "answered" and answer.verified
    assert answer.reason.startswith("template") and "92,823,254" in answer.answer and answer.facts[0].value == 92823254


#==========================================================================
# Declined questions never reach the explain step
#==========================================================================
def test_t09_declined_questions_never_reach_the_explain_step(tmp_path: Path) -> None:
    """
    Pin the hand-over: no data, an indicator that is not in the data, and a non-data question end
    with a status, a fixed sentence from code and no model-written answer.

    Input:
        . tmp_path      : temporary folder
    """
    #----------------------------------------------------------------------
    # A model that records every time it is asked to write
    #----------------------------------------------------------------------
    calls: list[str] = []

    def write(prompt: str) -> str:
        """
        Record that the model was asked to write.

        Input:
            . prompt        : the messages
        Output:
            . text          : an obviously fake answer
        """
        calls.append(prompt)
        return "[FAKE MODEL ANSWER]"

    #----------------------------------------------------------------------
    # Three questions the table cannot or must not answer
    #----------------------------------------------------------------------
    desk = indexed_desk(tmp_path, rules=rules_with_text(write))
    gap  = desk.ask("What was the GDP of Vietnam in 2024?", thread_id="a")
    kind = desk.ask("What was the unemployment rate of Spain in 2019?", thread_id="b")
    chat = desk.ask("What is the capital city of Japan?", thread_id="c")
    assert (gap.status, kind.status, chat.status) == ("no_data", "out_of_scope", "out_of_scope")
    assert "no GDP for Viet Nam in 2024" in gap.answer and "latest gdp year for VNM is 2023" in gap.reason
    assert gap.verified and kind.verified and chat.verified and calls == []


#==========================================================================
# A model that cannot fill the query still ends in a contract
#==========================================================================
def test_t09_a_model_that_cannot_fill_the_query_still_ends_in_a_contract(tmp_path: Path) -> None:
    """
    Pin that an invalid structured output (an operation that does not exist) becomes
    out_of_scope with the reason, not a crash.

    Input:
        . tmp_path      : temporary folder
    """
    bad    = replace(project_fake_rules(), structured=lambda schema, prompt: {
        "operation": "sum", "indicator_text": "gdp", "country_texts": [], "years": []})
    answer = indexed_desk(tmp_path, rules=bad).ask("anything", thread_id="t")
    assert answer.status == "out_of_scope" and "not a valid Query" in answer.reason


#==========================================================================
# No tool call at all
#==========================================================================
def test_t09_a_model_that_makes_no_tool_call_ends_in_a_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Pin that structured output returning None (the model answered in prose) is handled like an invalid Query.

    Input:
        . monkeypatch   : replaces the parse chain by one that returns None
    """
    from    langchain_core.runnables        import  RunnableLambda

    from    factdesk.graph.nodes            import  parse as parse_module

    monkeypatch.setattr(parse_module, "build_parse_chain", lambda chat: RunnableLambda(lambda inputs: None))
    chat = type("Source", (), {"get": lambda self: None})()
    out  = parse_module.parse_node({"question": "anything"}, chat=chat)
    assert out["operation"] == "other" and "made no tool call" in out["reason"]


#==========================================================================
# flow-compare with the numeric question
#==========================================================================
def test_t09_flow_compare_marks_the_flow_answer_that_has_no_table_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Pin flow-compare on the real question file: the numeric question F4 is answered by the graph and
    a flow answer with an invented number is reported as unverified.

    Input:
        . tmp_path      : temporary folder
        . monkeypatch   : sets LANGFLOW_API_KEY for this test only
    """
    monkeypatch.setenv("LANGFLOW_API_KEY", "test-value")
    langflow        = FakeLangflow()
    langflow.answer = "About 90 million people lived there in 2015."
    config          = project_toml(tmp_path)
    text            = re.sub(r'^langflow_url = ".*?"', f'langflow_url = "{langflow.url}"', config.read_text(encoding="utf-8"), flags=re.MULTILINE)
    config.write_text(text, encoding="utf-8")
    main(["--config", str(config), "--fake", "index"])
    assert main(["--config", str(config), "--fake", "flow-compare"]) == 0
    langflow.close()
    rows = {row["id"]: row for row in json.loads((tmp_path / "results" / "flow_compare.json").read_text(encoding="utf-8"))["rows"]}
    assert rows["F4"]["flow_unverified"] == ["90 million"] and rows["F4"]["code_unverified"] == []
