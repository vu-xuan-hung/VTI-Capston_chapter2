"""Experiments: local run + save + compare; evaluators also fit langsmith."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  datetime
import  uuid
from    pathlib                         import  Path
from    typing                          import  Any

import  pytest
from    langsmith.evaluation.evaluator  import  run_evaluator
from    langsmith.schemas               import  Example as LangSmithExample
from    langsmith.schemas               import  Run
from    pydantic                        import  BaseModel

from    factdesk.evaluation.compare     import  case_change, compare_experiments
from    factdesk.evaluation.dataset     import  Example, load_dataset
from    factdesk.evaluation.experiment  import  load_experiment, run_experiment, save_experiment
from    factdesk.llm.fake_chat_model    import  FakeChatModel, FakeRules


#==========================================================================
# Toy app
#==========================================================================
class Reply(BaseModel):
    """
    Purpose: the structured answer of the toy target.
    """

    answer: str


#==========================================================================
# Toy target factory
#==========================================================================
def _target(prefix: str) -> Any:
    """
    Build a target that asks the fake model for a Reply: "<prefix> <question>".
    The question "boom" raises, to test a crash row.

    Input:
        . prefix        : what the fake model puts before the question
    Output:
        . target        : inputs dict -> outputs dict
    """
    rules = FakeRules(structured=lambda name, prompt: {"answer": f"{prefix} {prompt}"})
    chat  = FakeChatModel(rules=rules).with_structured_output(Reply)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Answer one case.

        Input:
            . inputs        : {"question": ...}
        Output:
            . outputs       : {"answer": ...}
        """
        if inputs["question"] == "boom":
            raise ValueError("the app crashed")
        return {"answer": chat.invoke(inputs["question"]).answer}

    return target


#==========================================================================
# Evaluators (LangSmith signature)
#==========================================================================
def ev_has_word(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """
    Check that the expected word is in the answer.

    Input:
        . inputs        : the case inputs
        . outputs       : the target outputs
        . reference_outputs: {"word": ...}
    Output:
        . passed        : True when the word appears
    """
    return reference_outputs["word"] in outputs["answer"]


#==========================================================================
# Second evaluator: a dict score
#==========================================================================
def ev_short(outputs: dict) -> dict:
    """
    Score the answer length: 1.0 under 30 characters, else 0.5.

    Input:
        . outputs       : the target outputs
    Output:
        . result        : {"score": ..., "comment": ...}
    """
    return {"score": 1.0 if len(outputs["answer"]) < 30 else 0.5, "comment": "length"}


#==========================================================================
# Dataset
#==========================================================================
def _dataset() -> list[Example]:
    """
    Three cases, one of which crashes the target.

    Output:
        . cases         : E1, E2, E3
    """
    return [
        Example("E1", {"question": "alpha"}, {"word": "new"}),
        Example("E2", {"question": "beta"}, {"word": "beta"}),
        Example("E3", {"question": "boom"}, {"word": "x"}),
    ]


#==========================================================================
# Run, save, load
#==========================================================================
def test_experiment_scores_counts_calls_and_keeps_errors(tmp_path: Path) -> None:
    """
    Pin per-case rows (score, seconds, model calls, error) and the saved files.

    Input:
        . tmp_path      : temporary results folder
    """
    experiment = run_experiment("v1", _target("old"), _dataset(), [ev_has_word, ev_short], backend="fake")
    rows       = {row["id"]: row for row in experiment.rows}
    assert (rows["E1"]["ev_has_word"], rows["E2"]["ev_has_word"]) == (0.0, 1.0)
    assert rows["E1"]["model_calls"] == 1
    assert rows["E3"]["error"] == "ValueError: the app crashed"
    assert (rows["E3"]["ev_has_word"], rows["E3"]["ev_short"], rows["E3"]["model_calls"]) == (0.0, 0.0, 0)
    assert experiment.summary["scores"] == {"ev_has_word": 0.333, "ev_short": 0.667}
    assert experiment.summary["errors"] == 1
    json_path, csv_path = save_experiment(experiment, tmp_path)
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("id,split,kind,seconds,model_calls")
    assert load_experiment(tmp_path, "v1") == experiment
    assert json_path.name == "v1.json"


#==========================================================================
# Compare case by case
#==========================================================================
def test_compare_lists_better_worse_and_same_cases() -> None:
    """
    Pin case-by-case comparison: E1 gains, E2 loses, E3 (crash) stays.
    """
    before = run_experiment("v1", _target("old"), _dataset(), [ev_has_word], backend="fake")
    after  = run_experiment("v2", _target("new"), _dataset(), [ev_has_word], backend="fake")
    result = compare_experiments(before, after)
    assert result.changes == {"E1": "better", "E2": "same", "E3": "same"}
    flipped = compare_experiments(after, before)
    assert flipped.worse == ["E1"]
    assert case_change([(0.0, 1.0), (1.0, 0.0)]) == "mixed"
    assert result.lines[0].startswith("case")


#==========================================================================
# The same evaluator runs on LangSmith
#==========================================================================
def test_evaluators_are_accepted_by_langsmith_offline() -> None:
    """
    Pin that langsmith wraps our evaluator signature and maps its arguments
    by name (no network: evaluate_run on local Run/Example objects).
    """

    #----------------------------------------------------------------------
    # A LangSmith run and example, built offline
    #----------------------------------------------------------------------
    now     = datetime.datetime.now(datetime.timezone.utc)
    run     = Run(
        id          = uuid.uuid4(),
        name        = "target",
        run_type    = "chain",
        inputs      = {"question": "beta"},
        outputs     = {"answer": "old beta"},
        start_time  = now,
        trace_id    = uuid.uuid4(),
    )
    example = LangSmithExample(
        id          = uuid.uuid4(),
        dataset_id  = uuid.uuid4(),
        inputs      = {"question": "beta"},
        outputs     = {"word": "beta"},
        created_at  = now,
    )
    #----------------------------------------------------------------------
    # Wrap and call the evaluators the way langsmith.evaluate does
    #----------------------------------------------------------------------
    word  = run_evaluator(ev_has_word).evaluate_run(run, example)
    short = run_evaluator(ev_short).evaluate_run(run, example)
    assert (word.key, word.score) == ("ev_has_word", True)
    assert (short.key, short.score, short.comment) == ("ev_short", 1.0, "length")


#==========================================================================
# Evaluator parameters only LangSmith has
#==========================================================================
def test_a_run_parameter_is_refused_locally() -> None:
    """
    Pin the local runner's message for a LangSmith-only parameter.
    """

    def ev_uses_run(run: Any) -> bool:
        """
        Ask for the LangSmith run object.

        Input:
            . run           : not available locally
        Output:
            . passed        : never reached
        """
        return True

    with pytest.raises(ValueError, match="not available locally"):
        run_experiment("x", _target("a"), _dataset()[:1], [ev_uses_run], backend="fake")


#==========================================================================
# Dataset file
#==========================================================================
def test_load_dataset_filters_by_split_and_refuses_duplicates(tmp_path: Path) -> None:
    """
    Pin split filtering and the unique-id rule.

    Input:
        . tmp_path      : temporary data folder
    """
    path = tmp_path / "cases.json"
    path.write_text(
        '[{"id": "A", "inputs": {}, "outputs": {}}, {"id": "B", "inputs": {}, "outputs": {}, "split": "holdout"}]',
        encoding="utf-8",
    )
    assert [case.id for case in load_dataset(path, split="holdout")] == ["B"]
    path.write_text('[{"id": "A", "inputs": {}, "outputs": {}}, {"id": "A", "inputs": {}, "outputs": {}}]', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        load_dataset(path)
