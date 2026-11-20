"""Key scan: a key in a submitted file is found; .env and runtime are skipped."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    pathlib                         import  Path

from    helpers                         import  ROOT

from    factdesk.runtime.key_scan       import  find_keys


#==========================================================================
# A key is found, by file and line
#==========================================================================
def test_a_key_in_a_submitted_file_is_reported_without_leaking(tmp_path: Path) -> None:
    """
    Pin file:line reporting with only the first characters shown; the fake
    key is built at run time so this file itself stays clean.

    Input:
        . tmp_path      : temporary project folder
    """
    fake_key = "lsv2_" + "pt_" + "a1b2c3d4e5" * 3
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(f'print("hi")\nKEY = "{fake_key}"\n', encoding="utf-8")
    found = find_keys(tmp_path)
    assert found == ["src/app.py:2: lsv2_pt_..."]
    assert fake_key not in found[0]


#==========================================================================
# Files that are never submitted are skipped
#==========================================================================
def test_env_files_and_runtime_folders_are_skipped(tmp_path: Path) -> None:
    """
    Pin the skip list: .env and runtime/ may hold keys, they are not submitted.

    Input:
        . tmp_path      : temporary project folder
    """
    fake_key = "sk-" + "Z9" * 12
    (tmp_path / ".env").write_text(f"LANGFLOW_API_KEY={fake_key}\n", encoding="utf-8")
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "trace.json").write_text(f'"{fake_key}"', encoding="utf-8")
    (tmp_path / ".env.example").write_text("LANGFLOW_API_KEY=\n", encoding="utf-8")
    assert find_keys(tmp_path) == []


#==========================================================================
# The kit itself is clean
#==========================================================================
def test_the_kit_has_no_key() -> None:
    """
    Pin that nothing in the kit looks like a key.
    """
    assert find_keys(ROOT) == []
