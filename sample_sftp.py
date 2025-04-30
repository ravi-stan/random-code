"""repo_search_tool.py
A ripgrep‑powered search Tool for smolagents.
Fixed: mark optional args as *nullable* per smolagents schema to avoid
``AssertionError: Nullable argument 'max_results' ... should have key 'nullable' set to True``.
"""

from smolagents import Tool
import subprocess, shlex, textwrap, pathlib, os

class RepoSearchTool(Tool):
    name = "repo_search"
    description = (
        "Search the local repository (recursively) for a regex or literal string via ripgrep "
        "and return matching lines with file paths + line numbers."
    )

    inputs = {
        "query": {
            "type": "string",
            "description": "Regex or literal to search for",
            "nullable": False,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of matches to return (default = 20)",
            "nullable": True,  # ← important for optional arg
        },
        "repo_path": {
            "type": "string",
            "description": "Root folder of the repository (default = '.')",
            "nullable": True,  # ← important for optional arg
        },
    }

    output_type = "string"

    # ----------------------------------------------------------------------------
    # core logic
    # ----------------------------------------------------------------------------
    def forward(
        self,
        query: str,
        max_results: int | None = 20,
        repo_path: str | os.PathLike | None = ".",
    ) -> str:
        repo_path = pathlib.Path(repo_path or ".").expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"{repo_path} does not exist")

        cmd = (
            f"rg --json -n --max-count {max_results or 20} "
            f"{shlex.quote(query)} {repo_path}"
        )
        proc = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc.returncode not in (0, 1):  # 1 = no matches
            raise RuntimeError(proc.stderr.strip())

        matches = []
        for line in proc.stdout.splitlines():
            if '"type":"match"' in line:
                path = line.split('"path":{"text":"', 1)[-1].split('"')[0]
                lno = line.split('"line_number":', 1)[-1].split(',')[0]
                text = line.split('"lines":{"text":"', 1)[-1].split('"')[0]
                matches.append(f"{path}:{lno}: {text}")

        return "No matches found." if not matches else "\n".join(matches)

