"""The rules of the fake model: --fake runs the whole project with no download.

The fake reads the question with a few regular expressions, the way a very
simple person would: good enough to drive every code path in a test, never a
real answer. Its text always starts with "[FAKE MODEL ANSWER]".
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  re
from    typing                          import  Any

from    langchain_core.messages         import  BaseMessage, HumanMessage, ToolMessage

from    factdesk.llm.fake_chat_model    import  FakeRules


#==========================================================================
# Rules used by --fake and by the fake model server
#==========================================================================
def project_fake_rules() -> FakeRules:
    """
    Return the rules the fake backend answers with: a regex parse for Query,
    an echo of the FACTS table for explain, one tool call for the agent.

    Output:
        . rules         : FakeRules for ChatSource and the fake model server
    """
    return FakeRules(structured=_structured, text=_text, tool_calls=_tool_calls)


#==========================================================================
# Question -> Query fields
#==========================================================================
def read_question(question: str) -> dict[str, Any]:
    """
    Guess the Query fields of a question with regular expressions.

    Input:
        . question      : the user's question
    Output:
        . fields        : operation, indicator_text, country_texts, years
    """

    #----------------------------------------------------------------------
    # Years, indicator and operation by keywords
    #----------------------------------------------------------------------
    lower = question.lower()
    years = [int(year) for year in re.findall(r"\b(?:19|20)\d\d\b", question)]
    if "per person" in lower or "per capita" in lower:
        indicator = "GDP per person"
    elif "unemployment" in lower:
        indicator = "unemployment rate"
    elif "gdp" in lower or "econom" in lower:
        indicator = "GDP"
    elif "population" in lower or "people" in lower or "inhabitants" in lower:
        indicator = "population"
    else:
        indicator = "unknown"
    if indicator == "unknown" and not years:
        operation = "other"
    elif lower.startswith("rank") or "rank " in lower:
        operation = "rank"
    elif "compare" in lower or " or " in lower or "which had" in lower:
        operation = "compare"
    elif len(years) == 2 or "grow" in lower:
        operation = "growth"
    else:
        operation = "value"

    #----------------------------------------------------------------------
    # Country names: runs of capitalised words that are not question words
    #----------------------------------------------------------------------
    skip   = {
        "what", "how", "which", "compare", "rank", "by", "roughly", "the", "gdp", "did", "was", "is", "in",
        "from", "between", "and", "much", "big", "who", "will", "tell", "do", "don", "of", "population", "people",
    }
    names  = re.findall(r"(?:[A-Z][\w.'-]*|S\.)(?: (?:of )?[A-Z][\w.'-]*)*", question)
    texts  = []
    for name in names:
        words = [word for word in re.split(r"[ ,]+", name) if word.lower().strip(".'") not in skip]
        texts += [" ".join(words).removesuffix("'s")] if words else []
    return {"operation": operation, "indicator_text": indicator, "country_texts": texts, "years": years}


#==========================================================================
# Structured output: the Query
#==========================================================================
def _structured(schema_name: str, prompt: str) -> dict[str, Any]:
    """
    Fill the Query schema from the question, which is the last line of the prompt.

    Input:
        . schema_name   : the Pydantic class name asked for
        . prompt        : every message's text joined by newlines
    Output:
        . fields        : the fields of Query
    """
    if schema_name != "Query":
        raise ValueError(f"fake rules know only the Query schema, got {schema_name!r}")
    return read_question(prompt.split("\n")[-1])


#==========================================================================
# Plain text: explain and the notes answer
#==========================================================================
def _text(prompt: str) -> str:
    """
    Echo the FACTS table so every number is copied exactly (the verify step
    passes); any other prompt gets the end of itself.

    Input:
        . prompt        : every message's text, the last one last
    Output:
        . text          : an obviously fake answer
    """
    if "FACTS:" in prompt:
        facts = re.split(r"\nYour previous answer|\nAnswer:", prompt.split("FACTS:\n")[1])[0].replace("\n", " ; ")
        return f"[FAKE MODEL ANSWER] {facts}"
    return "[FAKE MODEL ANSWER] " + " ".join(prompt.split()[-24:])


#==========================================================================
# Tool calls of the agent (v1)
#==========================================================================
def _tool_calls(messages: list[BaseMessage], tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """
    Call one tool per question, then stop: get_value, compare or growth by
    the question's shape, with country codes from a tiny name table.

    Input:
        . messages      : the conversation so far
        . tools         : bound OpenAI tool schemas (unused: the three names are known)
    Output:
        . calls         : one call, or None once a tool answered
    """
    if any(isinstance(message, ToolMessage) for message in messages):
        return None
    question = next(message.text for message in messages if isinstance(message, HumanMessage))
    fields   = read_question(question)
    codes    = {"vietnam": "VNM", "japan": "JPN", "korea": "KOR", "south korea": "KOR", "thailand": "THA", "canada": "CAN"}
    country  = [codes.get(text.lower(), text.upper()[:3]) for text in fields["country_texts"]] or ["VNM"]
    kind     = {"GDP per person": "gdp_per_capita", "GDP": "gdp"}.get(fields["indicator_text"], "population")
    years    = fields["years"] or [2020]
    if fields["operation"] == "growth" and len(years) == 2:
        return [{"name": "growth", "args": {"country": country[0], "indicator": kind, "year_from": years[0], "year_to": years[1]}}]
    if len(country) >= 2:
        return [{"name": "compare", "args": {"country_a": country[0], "country_b": country[1], "indicator": kind, "year": years[0]}}]
    return [{"name": "get_value", "args": {"country": country[0], "indicator": kind, "year": years[0]}}]
