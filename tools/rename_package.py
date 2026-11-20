"""Rename the kit package and re-align the import blocks.

Usage:
    python tools/rename_package.py NEW_NAME          # factdesk -> NEW_NAME everywhere
    python tools/rename_package.py --align-only PATH ...

A rename changes module path lengths, which breaks the aligned `import`
column of the house style; every import block is re-aligned afterwards.
Only import lines are touched: this is not a formatter.
"""

#==========================================================================
# Imports
#==========================================================================
import  argparse
import  os
import  re
import  sys
from    dataclasses                     import  dataclass
from    pathlib                         import  Path


#==========================================================================
# Rename settings
#==========================================================================
@dataclass(frozen=True)
class RenameConfig:
    """
    Purpose: the placeholder name, the text files rewritten, the folders
    skipped, and the import column of the house style.
    """

    old_name: str                       = "factdesk"
    suffixes: tuple[str, ...]           = (".py", ".toml", ".md", ".txt", ".yml", ".yaml", ".example", ".json")
    skip_dirs: tuple[str, ...]          = (".git", ".venv", "__pycache__", "runtime", "results", "kb_store", "fixtures")
    keyword_column: int                 = 41    # 1-based column of `import` / `as`
    import_pattern: str                 = (
        r"^(?P<indent> *)(?P<kw>from|import)\s+(?P<module>\S+)"
        r"(?:\s+(?P<kw2>import|as)\s+(?P<rest>.+))?$"
    )


#==========================================================================
# Align the import blocks of one text
#==========================================================================
def align_imports(text: str, *, cfg: RenameConfig) -> str:
    """
    Re-align every block of consecutive import lines with the same indent:
    module at indent + 8, `import`/`as` at the keyword column, pushed right
    for the whole block when a module path is too long.

    Input:
        . text          : a Python file's text
        . cfg           : the keyword column and the import pattern
    Output:
        . text          : the same text with aligned import lines
    """

    #----------------------------------------------------------------------
    # Group consecutive import lines by indent
    #----------------------------------------------------------------------
    lines  = text.split("\n")
    blocks: list[list[int]] = []
    for number, line in enumerate(lines):
        match = re.match(cfg.import_pattern, line)
        if not match or (match.group("kw") == "from" and not match.group("kw2")):
            continue
        same = blocks and blocks[-1][-1] == number - 1
        if same and _indent(lines[blocks[-1][-1]]) == len(match.group("indent")):
            blocks[-1].append(number)
        else:
            blocks.append([number])

    #----------------------------------------------------------------------
    # Rewrite each block
    #----------------------------------------------------------------------
    for block in blocks:
        matches = [re.match(cfg.import_pattern, lines[number]) for number in block]
        longest = max(len(match.group("module")) for match in matches)
        width   = max(cfg.keyword_column - 1 - 8, longest + 2)
        for number, match in zip(block, matches):
            head = f"{match.group('indent')}{match.group('kw'):<8}"
            if match.group("kw2"):
                lines[number] = f"{head}{match.group('module'):<{width}}{match.group('kw2'):<8}{match.group('rest')}"
            else:
                lines[number] = f"{head}{match.group('module')}"
    return "\n".join(lines)


#==========================================================================
# Rename the package everywhere
#==========================================================================
def rename(root: Path, new_name: str, *, cfg: RenameConfig) -> list[Path]:
    """
    Move src/<old> to src/<new>, replace the old name as a whole word in
    every text file, and re-align the imports of every Python file.

    Input:
        . root          : the project folder
        . new_name      : the new package name (a Python identifier)
        . cfg           : names, suffixes and skipped folders
    Output:
        . changed       : the files whose text changed
    """
    if not new_name.isidentifier() or new_name != new_name.lower():
        raise ValueError(f"package name must be a lower-case Python identifier, got {new_name!r}")
    source = root / "src" / cfg.old_name
    if not source.is_dir():
        raise ValueError(f"{source} not found: already renamed?")
    source.rename(root / "src" / new_name)
    changed = []
    for path in _text_files(root, cfg=cfg):
        old  = path.read_text(encoding="utf-8")
        text = re.sub(rf"\b{cfg.old_name}\b", new_name, old)
        text = align_imports(text, cfg=cfg) if path.suffix == ".py" else text
        if text != old:
            path.write_text(text, encoding="utf-8")
            changed.append(path)
    return changed


#==========================================================================
# Text files of the project
#==========================================================================
def _text_files(root: Path, *, cfg: RenameConfig) -> list[Path]:
    """
    List the files whose suffix is rewritten, skipping generated folders.

    Input:
        . root          : the project folder
        . cfg           : suffixes and skipped folders
    Output:
        . files         : sorted paths
    """
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in cfg.skip_dirs and not name.endswith(".egg-info"))
        found  += [Path(folder) / name for name in files if Path(name).suffix in cfg.suffixes]
    return sorted(found)


#==========================================================================
# Indent of a line
#==========================================================================
def _indent(line: str) -> int:
    """
    Count the leading spaces of a line.

    Input:
        . line          : one line of text
    Output:
        . spaces        : number of leading spaces
    """
    return len(line) - len(line.lstrip(" "))


#==========================================================================
# Command line
#==========================================================================
def main(argv: list[str]) -> int:
    """
    Rename the package, or only re-align the imports of some files.

    Input:
        . argv          : command-line arguments without the program name
    Output:
        . code          : 0
    """

    #----------------------------------------------------------------------
    # Arguments
    #----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Rename the kit package / align imports")
    parser.add_argument("--align-only", action="store_true", help="only re-align imports of PATH...")
    parser.add_argument("names", nargs="+", help="NEW_NAME, or files/folders with --align-only")
    args = parser.parse_args(argv)
    cfg  = RenameConfig()
    root = Path(__file__).resolve().parents[1]
    if not args.align_only:
        changed = rename(root, args.names[0], cfg=cfg)
        print(f"[OK] src/{args.names[0]}; {len(changed)} files updated. Next: pip install -e . --no-deps")
        return 0
    #----------------------------------------------------------------------
    # Only re-align the given files
    #----------------------------------------------------------------------
    paths = [Path(name) for name in args.names]
    files = [item for path in paths for item in (sorted(path.rglob("*.py")) if path.is_dir() else [path])]
    for path in files:
        old  = path.read_text(encoding="utf-8")
        text = align_imports(old, cfg=cfg)
        if text != old:
            path.write_text(text, encoding="utf-8")
            print(f"aligned {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
