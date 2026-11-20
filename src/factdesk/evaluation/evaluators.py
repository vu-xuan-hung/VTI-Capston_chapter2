"""Four code evaluators: no model judges the answer. Each is a LangSmith-style function.

The parameters are NAMED (inputs, outputs, reference_outputs), so the same
function runs in `python main.py eval v2` and on LangSmith (`--cloud`). The
metric name is the function name. `outputs` is the contract of the run;
`reference_outputs` is the case's expected block (status, indicator,
countries, years, result_value).
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                          import  annotations

import  re
from    typing                              import  Any

from    factdesk.config                     import  EvalConfig, FormatConfig
from    factdesk.verify.numbers             import  find_unverified


#==========================================================================
# Were the right things looked up
#==========================================================================
def entities_correct(
    inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> bool:
    """
    Check the entities the run worked with against the case: indicator,
    countries and years, each compared only when the case lists it
    (countries and years as sets: the order of the question is not the point).

    Input:
        . inputs        : the case inputs (unused)
        . outputs       : the contract of the run
        . reference_outputs: the expected block of the case
    Output:
        . ok            : True when every listed entity matches
    """
    for key in ("indicator", "countries", "years"):
        if key not in reference_outputs:
            continue
        got, want = outputs.get(key), reference_outputs[key]
        if isinstance(want, list):
            got, want = sorted(got or []), sorted(want)
        if got != want:
            return False
    return True


#==========================================================================
# Is the number right
#==========================================================================
def result_exact(
    inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> bool:
    """
    Check result.value against the case's expected result_value, within the
    relative tolerance of EvalConfig. When the case expects no number
    (result_value is null), the run must not have produced one.

    Input:
        . inputs        : the case inputs (unused)
        . outputs       : the contract of the run
        . reference_outputs: the expected block; "result_value" is a number or None
    Output:
        . ok            : True when the number matches (or both are absent)
    """
    v1 = outputs.get("result")
    v2 = reference_outputs.get("result_value")
    if v2 is None:
        return v1 is None
    if v1 is None:
        return False

    actual = v1.get("value")
    if actual is None:
        return False
    allowed_error = abs(v2) * EvalConfig().result_rel_tol
    return abs(actual - v2) <= allowed_error
    # TODO T06 -- compare outputs["result"]["value"] with reference_outputs["result_value"] within EvalConfig().result_rel_tol
    # (relative). When the expected value is None,
    # return True only if the run produced no result (outputs["result"] is None).
    # CHECK: pytest -k t06 -- exact, within tolerance, wrong, missing, and the null-result cases.
    # raise NotImplementedError("T06: result_exact")


#==========================================================================
# Does the answer text invent numbers
#==========================================================================
def answer_numbers_verified(
    inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> bool:
    """
    Check that no number in the answer text is missing from the run's own
    facts and result (the years of the run, the years written in the question
    and counting words are allowed). It reads the OUTPUT, so it also judges
    v1, which has no verify step.

    Input:
        . inputs        : the case inputs; "question" gives the years the answer may repeat
        . outputs       : the contract of the run: answer, facts, result, years
        . reference_outputs: the expected block (unused)
    Output:
        . ok            : True when find_unverified returns nothing
    """
    question_years = [
        int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", inputs.get("question", ""))
    ]
    return not find_unverified(
        answer=outputs.get("answer", ""),
        facts=outputs.get("facts", []),
        result=outputs.get("result"),
        years=outputs.get("years", []) + question_years,
        numbers=FormatConfig(),
    )


    # TODO T06 -- return True when find_unverified(answer, facts=..., result=..., years=..., numbers=FormatConfig()) finds nothing;
    # read answer, facts, result and years from the OUTPUTS of the run (it must also judge v1); allow the years written in inputs["question"].
    # CHECK: pytest -k t06 -- an invented number fails, rounded table numbers pass, an answer with numbers but no facts fails.
    # raise NotImplementedError("T06: answer_numbers_verified")


#==========================================================================
# Did the run end the right way
#==========================================================================
def status_correct(
    inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> bool:
    """
    Check the final status (answered, no_data, out_of_scope) against the case.

    Input:
        . inputs        : the case inputs (unused)
        . outputs       : the contract of the run
        . reference_outputs: the expected block; "status"
    Output:
        . ok            : True when the statuses are equal
    """
    return outputs.get("status") == reference_outputs["status"]


#==========================================================================
# The set an experiment runs
#==========================================================================
def all_evaluators() -> list[Any]:
    """
    List the four evaluators in the order of the results table.

    Output:
        . evaluators    : entities_correct, result_exact, answer_numbers_verified, status_correct
    """
    return [entities_correct, result_exact, answer_numbers_verified, status_correct]
