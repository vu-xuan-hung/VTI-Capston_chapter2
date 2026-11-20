"""Look for API keys in every file that would be submitted."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  os
import  re
from    dataclasses                     import  dataclass
from    pathlib                         import  Path


#==========================================================================
# What to scan
#==========================================================================
@dataclass(frozen=True)
class KeyScanConfig:
    """
    Purpose: key patterns (LangSmith, OpenAI-style incl. Langflow, Hugging
    Face, GitHub, AWS), the folders and files that are never submitted, and
    the text suffixes read.
    """

    patterns: tuple[str, ...]           = (
        r"lsv2_[A-Za-z0-9_]{10,}",
        r"sk-[A-Za-z0-9_-]{16,}",
        r"hf_[A-Za-z0-9]{20,}",
        r"gh[pousr]_[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{16}",
    )
    skip_dirs: tuple[str, ...]          = (
        ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "runtime", "kb_store",
    )
    skip_files: tuple[str, ...]         = (".env", ".env.langflow")
    suffixes: tuple[str, ...]           = (
        ".py", ".md", ".toml", ".txt", ".json", ".csv", ".yml", ".yaml", ".ipynb", ".example", ".cfg", ".ini",
    )


#==========================================================================
# Scan a project
#==========================================================================
def find_keys(root: Path, *, scan: KeyScanConfig = KeyScanConfig()) -> list[str]:
    """
    Report every text that looks like a key, by file and line; only the
    first characters are printed, so the report itself leaks nothing.

    Input:
        . root          : the project folder
        . scan          : patterns and skipped folders/files
    Output:
        . found         : "path:line: first 8 characters..." per match
    """
    pattern = re.compile("|".join(f"(?:{item})" for item in scan.patterns))
    found   = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in scan.skip_dirs and not name.endswith(".egg-info"))
        for name in sorted(files):
            path = Path(folder) / name
            if name in scan.skip_files or path.suffix not in scan.suffixes:
                continue
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for number, line in enumerate(lines, 1):
                relative_path = path.relative_to(root).as_posix()
                found += [f"{relative_path}:{number}: {key[:8]}..." for key in pattern.findall(line)]
    return found
