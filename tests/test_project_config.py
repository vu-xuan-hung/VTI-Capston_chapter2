"""configs/app.toml: every project section loads, and the values that code and tools share cannot drift."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations


from    factdesk.config                 import  EvalConfig, FormatConfig, GraphConfig, SliceConfig, load_config
from    helpers                         import  ROOT


#==========================================================================
# The TOML equals the defaults the evaluators use
#==========================================================================
def test_toml_numbers_and_graph_equal_the_class_defaults() -> None:
    """
    Pin that the evaluators (which cannot receive a config) see the same
    number rules, graph numbers and tolerance as the application.
    """
    config = load_config(ROOT / "configs" / "app.toml", root=ROOT)
    assert config.numbers == FormatConfig()
    assert config.graph == GraphConfig()
    assert config.evaluation == EvalConfig()


#==========================================================================
# The aggregate list and the alias table
#==========================================================================
def test_the_hand_checked_lists_have_the_expected_shape() -> None:
    """
    Pin the lists in [facts]: 48 aggregate codes without WLD or XKX, aliases keyed by real codes.
    """
    config = load_config(ROOT / "configs" / "app.toml", root=ROOT)
    assert len(config.facts.aggregate_codes) == 48 and len(set(config.facts.aggregate_codes)) == 48
    assert "WLD" not in config.facts.aggregate_codes and "XKX" not in config.facts.aggregate_codes
    assert config.facts.keep_aggregates == ["WLD"]
    assert config.facts.aliases["KOR"][0] == "South Korea"


#==========================================================================
# The notes chain settings
#==========================================================================
def test_the_rag_slice_has_the_values_a_flow_needs() -> None:
    """
    Pin [rag]: the template has {context} and {question} and the parser pattern
    uses chunk metadata keys.
    """
    rag = load_config(ROOT / "configs" / "app.toml", root=ROOT).rag
    assert isinstance(rag, SliceConfig)
    assert "{context}" in rag.template and "{question}" in rag.template
    assert rag.parser_pattern == "[{chunk_id}] {text}" and rag.k == 3


#==========================================================================
# No secrets in the example files
#==========================================================================
def test_env_examples_have_no_values_for_keys() -> None:
    """
    Pin that .env.example holds empty key values only.
    """
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    keys  = [line for line in lines if line.strip() and not line.startswith("#")]
    assert keys and all(line.endswith("=") for line in keys), keys
