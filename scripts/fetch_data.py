"""Fetch the two public datasets from GitHub (datasets organisation) at pinned commits.

Usage: python scripts/fetch_data.py [--out data/raw]
Stdlib only. Downloaded files are data: they are written to disk and never executed.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  argparse
import  sys
import  urllib.request
from    dataclasses                     import  dataclass
from    pathlib                         import  Path


#==========================================================================
# What to fetch
#==========================================================================
@dataclass(frozen=True)
class FetchConfig:
    """
    Purpose: the pinned commits and the file list of each dataset repository.
    """

    raw_base: str                       = "https://raw.githubusercontent.com/datasets"
    user_agent: str                     = "course-dataset-fetch/1.0"
    timeout: int                        = 60        # seconds per file
    shas: tuple[tuple[str, str], ...]   = (
        ("population", "d4b1a4a9a17e2123baf0ef119de05bceced1ab1b"),
        ("gdp", "36d761dc5617982d3e50b213e02b9a1b47c4e958"),
    )
    files: tuple[str, ...]              = ("datapackage.json", "README.md")
    csv_files: tuple[tuple[str, str], ...] = (
        ("population", "data/population.csv"),
        ("gdp", "data/gdp.csv"),
    )


#==========================================================================
# Download one file
#==========================================================================
def fetch_file(repo: str, sha: str, path: str, out: Path, *, fetch: FetchConfig) -> int:
    """
    Download one file at a pinned commit and write it under out/<repo>/<path>.

    Input:
        . repo          : repository name in the datasets organisation
        . sha           : the pinned commit
        . path          : the file path inside the repository
        . out           : the raw data folder
        . fetch         : base URL, user agent, timeout
    Output:
        . size          : bytes written
    """
    url     = f"{fetch.raw_base}/{repo}/{sha}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": fetch.user_agent})
    with urllib.request.urlopen(request, timeout=fetch.timeout) as response:
        body = response.read()
    target = out / repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    return len(body)


#==========================================================================
# Command line
#==========================================================================
def main(argv: list[str] | None = None) -> int:
    """
    Download every file of every dataset and print its size.

    Input:
        . argv          : arguments without the program name
    Output:
        . code          : 0 when all files were written
    """
    fetch  = FetchConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args   = parser.parse_args(argv)
    total  = 0
    for repo, sha in fetch.shas:
        paths = [path for name, path in fetch.csv_files if name == repo] + list(fetch.files)
        for path in paths:
            size   = fetch_file(repo, sha, path, args.out, fetch=fetch)
            total += size
            print(f"{repo}/{path}  {size} bytes  (sha {sha[:12]})")
    print(f"[OK] {total} bytes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
