"""The house style lint passes on the whole kit (docs/CODING_STYLE.md)."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  subprocess
import  sys

from    helpers                         import  ROOT


#==========================================================================
# Zero violations
#==========================================================================
def test_style_lint_reports_zero_violations() -> None:
    """
    Pin tools/style_lint.py on src, main.py, tools, tests and scripts;
    prompts.py may hold its templates as module constants.
    """
    targets = [name for name in ("src", "main.py", "tools", "tests", "scripts") if (ROOT / name).exists()]
    result  = subprocess.run(
        [sys.executable, "tools/style_lint.py", "--allow-consts-in", "prompts.py", *targets],
        cwd             = ROOT,
        capture_output  = True,
        text            = True,
    )
    assert result.stdout.strip().endswith("violations: 0"), result.stdout
    assert result.returncode == 0
