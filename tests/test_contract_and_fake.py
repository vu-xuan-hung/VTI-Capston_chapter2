"""The contract keys, the notes chain through the CLI, and what the fake model reads from a question."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    pathlib                         import  Path

import  pytest

from    factdesk.contract               import  Answer
from    factdesk.llm.fake_rules         import  read_question
from    helpers                         import  project_toml
from    main                            import  main


#==========================================================================
# The keys of the contract
#==========================================================================
def test_contract_has_the_documented_keys_in_order() -> None:
    """
    Pin to_contract: the twelve keys of the README, in order, with null for what a hand-over leaves empty.
    """
    contract = Answer(question="q", status="out_of_scope", reason="r").to_contract()
    assert list(contract) == [
        "question", "operation", "indicator", "countries", "years", "facts", "result", "unit", "answer",
        "verified", "status", "reason",
    ]
    assert contract["operation"] is None and contract["result"] is None and contract["facts"] == []


#==========================================================================
# What the fake model reads
#==========================================================================
@pytest.mark.parametrize(("question", "operation", "indicator", "countries", "years"), [
    ("What was Vietnam's population in 2015?", "value", "population", ["Vietnam"], [2015]),
    ("Compare the GDP per person of Japan and South Korea in 2020.", "compare", "GDP per person", ["Japan", "South Korea"], [2020]),
    ("By how much did Thailand's economy grow between 2010 and 2020?", "growth", "GDP", ["Thailand"], [2010, 2020]),
    ("What is the capital city of Japan?", "other", "unknown", ["Japan"], []),
])
def test_fake_model_reads_the_question_by_rules(
    question: str, operation: str, indicator: str, countries: list[str], years: list[int]
) -> None:
    """
    Pin the regex reading of the fake model, which drives every fake test.

    Input:
        . question      : a user question
        . operation     : the expected operation
        . indicator     : the expected indicator text
        . countries     : the expected country texts
        . years         : the expected years
    """
    fields = read_question(question)
    assert (fields["operation"], fields["indicator_text"], fields["country_texts"], fields["years"]) == (
        operation, indicator, countries, years)


#==========================================================================
# The notes chain (the Langflow slice) through the CLI
#==========================================================================
def test_define_answers_from_note_chunks_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Pin `define`: it answers from the retrieved chunks and prints the chunk ids it used.

    Input:
        . tmp_path      : temporary folder
        . capsys        : captured output
    """
    config = str(project_toml(tmp_path))
    assert main(["--config", config, "--fake", "index"]) == 0
    capsys.readouterr()
    assert main(["--config", config, "--fake", "define", "What does GDP mean? Goods and services."]) == 0
    out = capsys.readouterr().out
    assert out.startswith("[FAKE MODEL ANSWER]") and "(notes used: " in out


#==========================================================================
# The hand-over sentences
#==========================================================================
def test_decline_node_writes_a_fixed_sentence_for_each_kind_of_hand_over() -> None:
    """
    Pin decline_node: a missing cell names the country, indicator and year; an unknown
    country text is named; anything else gets the scope sentence. All are verified (no data number).
    """
    from    factdesk.compute.table          import  load_facts
    from    factdesk.graph.nodes.decline    import  decline_node
    from    helpers                         import  ROOT

    table = load_facts(ROOT / "data" / "facts.csv", ROOT / "data" / "countries.json", ROOT / "data" / "indicators.json")
    cell  = decline_node({"status": "no_data", "missing": {"code": "VNM", "year": 2024, "indicator": "gdp"}}, table=table)
    assert cell["answer"] == "The table has no GDP for Viet Nam in 2024, so I cannot answer." and cell["verified"] is True
    name  = decline_node({"status": "no_data", "unknown": ["Atlantis"]}, table=table)
    assert name["answer"] == "I could not find Atlantis in the table, so I cannot answer."
    scope = decline_node({"status": "out_of_scope", "reason": "no country is named"}, table=table)
    assert scope["answer"].startswith("I cannot answer this question: no country is named. I answer questions that ask for")
