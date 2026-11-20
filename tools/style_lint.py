"""House-style lint (docs/CODING_STYLE.md), adapted from the VMS rollout lint.

Usage: python tools/style_lint.py [--allow-consts] [--allow-consts-in NAME]... FILE_OR_DIR ...
Folders are walked on disk (no git needed); .git, __pycache__, runtime,
results, kb_store and .venv are skipped. Exit 1 when any violation is found.
  R1  banner/section rule lines end exactly at column 75
  R2  every def has a docstring
  R3  docstring has Input: when the def has parameters (besides self/cls)
  R4  docstring has Output: when the def returns a value
  R5  no bare `except:`
  R6  top-level def/class preceded by a `#=` banner; methods by a `#-` section
  R7  module-level import lines use the aligned `import  ` / `from    ` form
  R8  module-level UPPERCASE constant (should live in a config class);
      skipped for tests, with --allow-consts, and for files named by
      --allow-consts-in (e.g. prompts.py, whose templates are constants)
  R9  placeholder docstring (generated filler, name repeated as meaning)
  R10 function body over 15 code lines without a #- section
  R11 function assigns/accepts a name that shadows a module-level cfg instance
"""

#==========================================================================
# Imports
#==========================================================================
import  argparse
import  ast
import  os
import  re
import  sys
from    collections.abc                 import  Iterator
from    dataclasses                     import  dataclass


#==========================================================================
# Lint settings
#==========================================================================
@dataclass(frozen=True)
class LintConfig:
    """
    Purpose: the rule-line pattern and width, the section threshold and the
    folders that are never linted.
    """

    rule_pattern: str                   = r"^( *)#([=-])\2{9,}\s*$"
    width: int                          = 75
    max_unsectioned: int                = 15
    skip_dirs: tuple[str, ...]          = ("__pycache__", ".git", "runtime", "results", "kb_store", ".venv")


#==========================================================================
# Files to lint
#==========================================================================
def iter_files(paths: list[str], *, cfg: LintConfig) -> Iterator[str]:
    """
    Yield the .py files of the given files and folders, walking folders on
    disk and skipping the configured folder names (never a Python package: a
    package named runtime is source code, not run output).

    Input:
        . paths         : files and folders from the command line
        . cfg           : lint settings (skipped folders)
    Output:
        . files         : paths of Python files, folders walked in sorted order
    """
    for path in paths:
        if not os.path.isdir(path):
            yield path
            continue
        for folder, dirs, files in os.walk(path):
            dirs[:] = sorted(
                name for name in dirs
                if name not in cfg.skip_dirs or os.path.exists(os.path.join(folder, name, "__init__.py"))
            )
            for name in sorted(files):
                if name.endswith(".py"):
                    yield os.path.join(folder, name)


#==========================================================================
# Walk a function without nested definitions
#==========================================================================
def own_nodes(fn: ast.AST) -> Iterator[ast.AST]:
    """
    Walk a function's body without descending into nested defs or classes.

    Input:
        . fn            : a FunctionDef or AsyncFunctionDef
    Output:
        . nodes         : every AST node that belongs to fn itself
    """
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))


#==========================================================================
# Does a function return a value?
#==========================================================================
def fn_returns(fn: ast.AST) -> bool:
    """
    Tell whether a function returns something other than None, or yields.

    Input:
        . fn            : a FunctionDef or AsyncFunctionDef
    Output:
        . returns       : True when Output: is required in its docstring
    """
    for node in own_nodes(fn):
        if isinstance(node, ast.Return) and node.value is not None:
            if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                return True
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
    return False


#==========================================================================
# Is a rule line directly above a definition?
#==========================================================================
def banner_above(lines: list[str], node: ast.AST, ch: str, *, cfg: LintConfig) -> bool:
    """
    Check that the line right above a def/class (or its first decorator)
    is a rule line of the given character; a blank line in between fails.

    Input:
        . lines         : the file's lines
        . node          : a FunctionDef, AsyncFunctionDef or ClassDef
        . ch            : "=" for a banner, "-" for a section
        . cfg           : lint settings (rule pattern)
    Output:
        . found         : True when the banner/section is in place
    """
    first = min([node.lineno] + [item.lineno for item in node.decorator_list])
    above = first - 2
    if above < 0 or lines[above].strip() == "":
        return False
    match = re.match(cfg.rule_pattern, lines[above])
    return bool(match and match.group(2) == ch)


#==========================================================================
# Lint one definition
#==========================================================================
def lint_def(
    node: ast.AST,
    lines: list[str],
    *,
    depth: int,
    in_class: bool,
    cfg: LintConfig,
) -> list[tuple[int, str, str]]:
    """
    Apply R2, R3, R4, R9, R6 and R10 to one function or method.

    Input:
        . node          : a FunctionDef or AsyncFunctionDef
        . lines         : the file's lines
        . depth         : 0 for a top-level def, 1 for a method, more if nested
        . in_class      : True when node is directly inside a class
        . cfg           : lint settings
    Output:
        . violations    : (line, rule, message) tuples
    """

    #----------------------------------------------------------------------
    # Docstring rules
    #----------------------------------------------------------------------
    out    = []
    doc    = ast.get_docstring(node)
    params = [item.arg for item in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    if node.args.vararg:
        params.append(node.args.vararg.arg)
    if node.args.kwarg:
        params.append(node.args.kwarg.arg)
    params = [name for name in params if name not in ("self", "cls")]
    if doc is None:
        out.append((node.lineno, "R2", f"def {node.name} has no docstring"))
    else:
        if params and "Input:" not in doc:
            out.append((node.lineno, "R3", f"def {node.name} docstring lacks Input:"))
        if fn_returns(node) and "Output:" not in doc:
            out.append((node.lineno, "R4", f"def {node.name} docstring lacks Output:"))
        summary    = doc.strip().split("\n")[0].lower()
        stop_words = {"pins", "pin", "that", "the", "a", "an", "test", "tests"}
        words      = set(re.findall(r"[a-z0-9]+", summary)) - stop_words
        name_words = set(node.name.lower().split("_")) - {"test", ""}
        echo_name  = bool(words) and words <= name_words
        if echo_name or re.search(r"^Pins?: |: returned value$|^\. (\w+) +: \1$", doc, re.M):
            out.append((node.lineno, "R9", f"def {node.name} has a placeholder docstring"))

    #----------------------------------------------------------------------
    # Banner, section and length rules
    #----------------------------------------------------------------------
    if depth == 0 and not banner_above(lines, node, "=", cfg=cfg):
        out.append((node.lineno, "R6", f"top-level def {node.name} lacks #= banner"))
    body_start  = node.body[0].end_lineno + 1 if doc is not None else node.body[0].lineno
    body        = lines[body_start - 1 : node.end_lineno]
    code        = [line for line in body if line.strip() and not line.strip().startswith("#")]
    has_section = any(
        (match := re.match(cfg.rule_pattern, line)) and match.group(2) == "-" for line in body
    )
    if len(code) > cfg.max_unsectioned and not has_section:
        out.append((node.lineno, "R10", f"def {node.name} body has {len(code)} lines but no #- section"))
    if in_class and depth == 1 and not banner_above(lines, node, "-", cfg=cfg):
        out.append((node.lineno, "R6", f"method {node.name} lacks #- section"))
    return out


#==========================================================================
# Lint one file
#==========================================================================
def lint(path: str, *, allow_consts: bool, cfg: LintConfig) -> list[tuple[int, str, str]]:
    """
    Apply every rule to one Python file.

    Input:
        . path          : the file
        . allow_consts  : True to skip R8 for this file
        . cfg           : lint settings
    Output:
        . violations    : (line, rule, message) tuples, file order not guaranteed
    """

    #----------------------------------------------------------------------
    # R1 rule-line width, then parse
    #----------------------------------------------------------------------
    with open(path, encoding="utf-8") as file:
        src = file.read()
    lines = src.split("\n")
    out   = [
        (number, "R1", f"rule line length {len(line.rstrip())} != {cfg.width}")
        for number, line in enumerate(lines, 1)
        if re.match(cfg.rule_pattern, line) and len(line.rstrip()) != cfg.width
    ]
    try:
        tree = ast.parse(src)
    except SyntaxError as error:
        return [(error.lineno or 0, "SYNTAX", str(error))]
    parts   = os.path.normpath(path).split(os.sep)
    is_test = "tests" in parts[:-1] or parts[-1].startswith(("test_", "conftest"))

    #----------------------------------------------------------------------
    # Definitions, recursively (R2-R4, R6, R9, R10)
    #----------------------------------------------------------------------
    def visit(body: list[ast.stmt], depth: int, in_class: bool) -> None:
        """
        Lint every def and class in a body, then their own bodies.

        Input:
            . body          : statements of a module, class or function
            . depth         : nesting depth of these statements
            . in_class      : True when body is a class body
        """
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.extend(lint_def(node, lines, depth=depth, in_class=in_class, cfg=cfg))
                visit(node.body, depth + 1, False)
            elif isinstance(node, ast.ClassDef):
                if depth == 0 and not banner_above(lines, node, "=", cfg=cfg):
                    out.append((node.lineno, "R6", f"top-level class {node.name} lacks #= banner"))
                visit(node.body, depth + 1, True)
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        visit([child], depth, in_class)

    visit(tree.body, 0, False)

    #----------------------------------------------------------------------
    # R11 shadowed module-level cfg
    #----------------------------------------------------------------------
    module_cfgs = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name) and (target.id == "cfg" or target.id.endswith("_cfg"))
    }
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local  = {n.id for n in own_nodes(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            local |= {item.arg for item in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
            for name in sorted(local & module_cfgs):
                out.append((fn.lineno, "R11", f"def {fn.name} shadows module-level {name}"))

    #----------------------------------------------------------------------
    # R7 aligned imports, R5 bare except, R8 module constants
    #----------------------------------------------------------------------
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = lines[node.lineno - 1]
            if isinstance(node, ast.Import) and not text.startswith("import  "):
                out.append((node.lineno, "R7", "import not in aligned `import  x` form"))
            if isinstance(node, ast.ImportFrom) and not text.startswith("from    "):
                out.append((node.lineno, "R7", "from-import not in aligned `from    x` form"))
        for handler in ast.walk(node):
            if isinstance(handler, ast.ExceptHandler) and handler.type is None:
                out.append((handler.lineno, "R5", "bare except"))
        if not (allow_consts or is_test) and isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and re.fullmatch(r"[A-Z][A-Z0-9_]*", target.id) and target.id != "__all__":
                    out.append((node.lineno, "R8", f"module constant {target.id}"))
    return out


#==========================================================================
# Command line
#==========================================================================
def main(argv: list[str]) -> int:
    """
    Lint the given files and folders and print one line per violation.

    Input:
        . argv          : command-line arguments without the program name
    Output:
        . code          : 0 when clean, 1 when any violation is found
    """
    parser = argparse.ArgumentParser(description="House-style lint (docs/CODING_STYLE.md)")
    parser.add_argument("--allow-consts", action="store_true", help="skip R8 everywhere")
    parser.add_argument("--allow-consts-in", action="append", default=[], metavar="NAME",
                        help="skip R8 for files with this basename (repeatable)")
    parser.add_argument("paths", nargs="+")
    args  = parser.parse_args(argv)
    cfg   = LintConfig()
    total = 0
    for path in sorted(set(iter_files(args.paths, cfg=cfg))):
        allow = args.allow_consts or os.path.basename(path) in args.allow_consts_in
        for number, code, message in sorted(lint(path, allow_consts=allow, cfg=cfg)):
            print(f"{path}:{number}: {code} {message}")
            total += 1
    print(f"violations: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
