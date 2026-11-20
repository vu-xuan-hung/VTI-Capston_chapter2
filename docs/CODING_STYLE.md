# Project Coding Style

This is the shared coding-style rule for every contributor and agent. It
applies to `src/`, `main.py`, `tools/`, `scripts/` and `tests/`. More specific
rules in `AGENTS.md`, a plan, or a module's established contract take
precedence.

## Core rule

Write the simplest code that clearly expresses the task and its important
data, evaluation, and operational contracts. Prefer direct operations and
small helpers over layers of wrappers, configuration objects, and defensive
checks. Validate where a concrete risk enters, not at every internal step.
Do not silently repair a broken contract.

## Structure and names

- Keep configuration, immutable contracts, pure transformations, IO, and
  orchestration distinct. Put shared, importable logic in a module with no
  top-level execution; keep notebooks thin orchestration and presentation
  layers.
- Give one function one named responsibility. Use verb-led public names;
  reserve a leading underscore for a local helper. Name data objects for their
  role (`history`, `future_exog`, `scored`, `calibration`), not their temporary
  implementation.
- Put experiment/model settings in one early configuration section, held in
  config classes (see "Config classes" below); do not scatter unexplained
  literals through transformations.
- Use a typed record when related settings or provenance genuinely need to
  travel together. Ordinary arguments, values, and dictionaries are sufficient
  for simple local work; do not create a class for every intermediate result.
- Annotate function parameters and returns. Use concrete collection types and
  keyword-only arguments after `*` when values could be confused at a call
  site. Every function and method has a docstring in the Input/Output form
  (see "Formatting and comments"); keep it focused on the contract and an
  important non-obvious invariant.

## Validation and failure behavior

- Validate untrusted inputs and important invariants once at the relevant
  boundary. Select checks for plausible failures in the current task, such as
  incompatible artifacts, invalid join cardinality, or cutoff leakage; do not
  turn this list into boilerplate for every function. Internal helpers may rely
  on already validated inputs until an operation can break that invariant.
- Let clear library exceptions propagate. Avoid generic try/except wrappers,
  repeated existence/schema/null checks, fallback chains, or custom validators
  that merely duplicate an operation's own useful error.
- Fail loudly and locally. Raise the narrowest useful exception with the
  violated invariant plus diagnostic counts, identifiers, date range, or
  expected/actual values. Log the same actionable message where the component
  convention uses `aiLogger`; do not log a secret.
- Allow a fallback only when it is an intentional part of the contract. Make
  its trigger, value, and provenance observable in the returned data or run
  metadata. Never mask a missing source, incompatible schema, or failed gate
  with a generic default.
- Implement acceptance gates only where the task or an established contract
  requires them. Make those gates explicit and stop when they fail. Exploratory
  diagnostics should usually produce a concise observation, not another gate,
  gate registry, or pass/fail framework.

## Formatting and comments

The project uses the VMS house style: banner and section comments, column
alignment, and Input/Output docstrings. There is no auto-formatter;
black/ruff-format would strip the alignment, so do not run them. Authors apply
the style by hand and reviewers check it. Every module of this project is an
example; `src/factdesk/config.py` shows the config-class pattern.

```python
"""Module docstring stays the first statement."""

#==========================================================================
# Imports
#==========================================================================
import  calendar
import  os

import  polars                          as      pl
from    dotenv                          import  load_dotenv

from    src.common.metrics              import  wmape


#==========================================================================
# Snap stores onto the weather grid
#==========================================================================
def snap_stores(stores: pl.DataFrame, *, cell_deg: float) -> pl.DataFrame:
    """
    Assign every store to a cell_deg x cell_deg weather grid cell.

    Input:
        . stores        : store_code, lat, lon (one row per store)
        . cell_deg      : grid resolution in degrees
    Output:
        . cells         : stores plus cell_lat, cell_lon
    """

    #----------------------------------------------------------------------
    # Snap coordinates to the grid
    #----------------------------------------------------------------------
    lat_cell        = (pl.col("lat") / cell_deg).round() * cell_deg
    lon_cell        = (pl.col("lon") / cell_deg).round() * cell_deg  # same grid as vintage

    return stores.with_columns(cell_lat=lat_cell, cell_lon=lon_cell)
```

- **Width.** Every banner/section rule line ends exactly at column 75,
  whatever its indent: `#` plus `=` or `-` up to column 75. Nested sections
  therefore get shorter rules.
- **Banners (`#=`).** Use three lines: rule, `# Title`, rule. Put one before
  the import block, before each top-level section (configuration, constants,
  orchestration/`main`), and before each top-level function and class.
  Separate a banner from the previous code with two blank lines.
- **Sections (`#-`).** Use the same three-line form. Put one before each
  method inside a class, and one before each logical step inside a function
  body longer than about ten lines. Short bodies need no sections. A section
  title says what the step does, not how.
- **Alignment.** Inside one contiguous block (no blank line in between), align:
  the `=` of consecutive assignments; the module and `import`/`as` keyword
  columns of an import group (`import  x`, `from    x   import  y`); the
  `=`/`:` of class fields; and trailing `#` comments. Never align across a
  blank line or section. In a multi-line call, put one argument per line
  with a four-space hanging indent, and align keyword arguments as
  `name        = value` (padded name, one space after `=`).
- **Docstrings.** Every function and method has one, including private
  helpers and tests:

  ```python
  """
  One-line summary of what it does.

  Input:
      . name          : meaning, unit, grain
  Output:
      . name          : meaning, unit, grain
  """
  ```

  Leave out `Input:` when there are no parameters (besides `self`/`cls`) and
  `Output:` when the function returns `None`. A test docstring says which
  behavior it pins. Class docstrings start with `Purpose:`.
- **Imports.** Group them as standard library, third party, then project,
  each under its own `#-` sub-header when there is more than one group.
  Import order inside a group stays alphabetical.
- **General.** Use four-space indentation and parentheses rather than `\\`
  continuations. Keep snake_case names, type hints and keyword-only
  arguments. Do not copy these VMS habits: bare `except:`, swallowed
  exceptions in loops, commented-out code, camelCase, `__` name mangling,
  `obj   .method()` call alignment, or `raise RuntimeError` in place of
  `abc.abstractmethod`.
- **Comment content.** Write comments and docstrings in English. Explain
  decisions, units, source boundaries, leakage risks, or invariants, not a line
  that is already obvious from clear code. Keep comments next to the code they
  constrain. No plan, phase or finding references in code.
- Prefer direct expressions and named intermediate values to dense chains.

## Config classes

- Group settings into `@dataclass(frozen=True)` config classes: one class
  per config module, split only when a field group has a separate consumer. Fields are
  lower snake_case, type-annotated and aligned, with a `#-` section per field
  group. The class docstring starts with `Purpose:`.
- Callers instantiate the class and read fields: `cfg = ProductConfig()`,
  `cfg.horizon`. A field that derives from another uses
  `field(default_factory=...)` or a `@property`; do not mutate a config.

## Before handing off

Confirm the code reads directly, each added guard addresses a concrete failure,
and repeated checks or unnecessary abstractions have been removed. Preserve
required provenance, fallback behavior, appropriate tests, and checkpoint
safety. Apply the specific correctness requirements from `AGENTS.md`; simpler
code does not mean dropping required leakage, data-integrity, or security checks.

## Checking the style in this project

- `python tools/style_lint.py --allow-consts-in prompts.py src main.py tools tests scripts`
  must print `violations: 0` (`tests/test_style.py` runs it). It checks the rule
  width, docstrings (Input/Output, no placeholder text), banners and sections,
  aligned module-level imports, bare `except`, module-level UPPERCASE constants
  and shadowed `cfg` names. Prompt templates are module constants by design:
  keep them in a file named `prompts.py`.
- Import blocks in this project are separated by a blank line, as in the example
  above; a `#-` sub-header per group is also accepted.
- `python tools/rename_package.py --align-only FILE_OR_DIR ...` re-aligns import
  blocks (only import lines; it is not a formatter).
- `ruff check .` is for real errors only (`E`, `F`; alignment and line length off).
