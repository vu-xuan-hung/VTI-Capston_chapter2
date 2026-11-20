"""Load evaluation cases: what the app receives and what a correct answer holds."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  json
from    dataclasses                     import  dataclass
from    pathlib                         import  Path
from    typing                          import  Any


#==========================================================================
# One case
#==========================================================================
@dataclass(frozen=True)
class Example:
    """
    Purpose: one case, shaped like a LangSmith example: inputs for the
    target, outputs as the reference the evaluators compare with, and the
    split it belongs to (e.g. "dev" while improving, "holdout" once, at the end).
    """

    id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    split: str                          = "dev"
    kind: str                           = ""        # what the case tests: value, compare, ambiguous, no_data ...
    note: str                           = ""        # one line: why the case exists


#==========================================================================
# Load a dataset file
#==========================================================================
def load_dataset(path: Path, *, split: str | None = None) -> list[Example]:
    """
    Read a JSON list of {"id", "inputs", "outputs", "split"?} in file order.
    Ids must be unique: compare matches two experiments by id.

    Input:
        . path          : the dataset file, e.g. data/dataset.json
        . split         : keep only this split, or None for all
    Output:
        . examples      : the cases
    """
    examples = [Example(**row) for row in json.loads(path.read_text(encoding="utf-8"))]
    ids      = [example.id for example in examples]
    repeated = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if repeated:
        raise ValueError(f"{path.name}: duplicated case ids {repeated}")
    chosen = [example for example in examples if split is None or example.split == split]
    if not chosen:
        splits = sorted({example.split for example in examples})
        raise ValueError(f"{path.name}: no case in split {split!r}; splits: {splits}")
    return chosen
