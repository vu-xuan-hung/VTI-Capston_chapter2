"""Compare two saved experiments case by case: a mean hides what got worse."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    dataclasses                     import  dataclass

from    factdesk.evaluation.experiment  import  Experiment


#==========================================================================
# The comparison
#==========================================================================
@dataclass(frozen=True)
class Comparison:
    """
    Purpose: the change of every shared case. "mixed" (one score up, another
    down) is listed under worse too: something got worse.
    """

    metrics: list[str]
    changes: dict[str, str]             # case id -> better | worse | mixed | same
    better: list[str]
    worse: list[str]
    same: list[str]
    only_before: list[str]
    only_after: list[str]
    lines: list[str]


#==========================================================================
# Compare two experiments
#==========================================================================
def compare_experiments(before: Experiment, after: Experiment) -> Comparison:
    """
    Match cases by id and compare each shared metric's score.

    Input:
        . before        : the older version
        . after         : the newer version on the same cases
    Output:
        . comparison    : per-case change, the lists, and a printable table
    """

    #----------------------------------------------------------------------
    # Match cases and metrics
    #----------------------------------------------------------------------
    metrics = [name for name in before.summary["scores"] if name in after.summary["scores"]]
    if not metrics:
        raise ValueError(f"{before.name} and {after.name} share no evaluator")
    old   = {row["id"]: row for row in before.rows}
    new   = {row["id"]: row for row in after.rows}
    cases = [case_id for case_id in old if case_id in new]

    #----------------------------------------------------------------------
    # Label each case and write its line
    #----------------------------------------------------------------------
    width   = max(10, *(len(metric) for metric in metrics))
    lines   = [f"{'case':<8}" + "".join(f"{metric:>{width + 3}}" for metric in metrics) + "   change"]
    changes = {}
    for case_id in cases:
        pairs  = [(old[case_id][metric], new[case_id][metric]) for metric in metrics]
        change = case_change(pairs)
        cells  = "".join(f"{_arrow(a, b):>{width + 3}}" for a, b in pairs)
        changes[case_id] = change
        lines.append(f"{case_id:<8}{cells}   {'' if change == 'same' else change}".rstrip())

    #----------------------------------------------------------------------
    # Means and lists
    #----------------------------------------------------------------------
    better = [case_id for case_id in cases if changes[case_id] == "better"]
    worse  = [case_id for case_id in cases if changes[case_id] in ("worse", "mixed")]
    same   = [case_id for case_id in cases if changes[case_id] == "same"]
    scores = (before.summary["scores"], after.summary["scores"])
    means  = "".join(f"{_arrow(scores[0][m], scores[1][m]):>{width + 3}}" for m in metrics)
    lines += ["", f"{'mean':<8}{means}", f"better: {better}", f"worse:  {worse}", f"same:   {len(same)} cases"]
    return Comparison(
        metrics     = metrics,
        changes     = changes,
        better      = better,
        worse       = worse,
        same        = same,
        only_before = [case_id for case_id in old if case_id not in new],
        only_after  = [case_id for case_id in new if case_id not in old],
        lines       = lines,
    )


#==========================================================================
# Label one case
#==========================================================================
def case_change(pairs: list[tuple[float, float]]) -> str:
    """
    Say how one case changed over its metrics.

    Input:
        . pairs         : (score before, score after) per metric
    Output:
        . change        : "better" (only gains), "worse" (only losses),
                          "mixed" (both) or "same"
    """
    gained = any(after > before for before, after in pairs)
    lost   = any(after < before for before, after in pairs)
    if gained and lost:
        return "mixed"
    if gained:
        return "better"
    return "worse" if lost else "same"


#==========================================================================
# One score change as text
#==========================================================================
def _arrow(before: float, after: float) -> str:
    """
    Write a score change compactly for the table.

    Input:
        . before        : old score
        . after         : new score
    Output:
        . text          : e.g. "0.50->1.00"
    """
    return f"{before:.2f}->{after:.2f}"
