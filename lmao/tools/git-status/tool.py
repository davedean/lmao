from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.plugin_helpers import find_repo_root

PLUGIN = {
    "name": "git_status",
    "description": "Show git status (porcelain + branch) for the current repo.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "target ignored; args ignored",
    "usage": "{'tool':'git_status','target':'','args':''}",
}


def _success(data: dict) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": False, "error": message}, ensure_ascii=False)


def run(
    target: str,
    args: str,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    task_manager=None,
    debug_logger: Optional[object] = None,
) -> str:
    repo_root = find_repo_root(base)
    if not (repo_root / ".git").exists():
        return _error("not inside a git repository")

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-b"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return _error(f"git status exception: {exc}")

    if result.returncode != 0:
        return _error(f"git status failed: {result.stderr.strip() or result.stdout.strip()}")

    content = (result.stdout or "").rstrip()
    if len(content) > 200_000:
        content = content[:200_000]
        truncated = True
    else:
        truncated = False
    data = {"repo_root": str(repo_root), "output": content}
    if truncated:
        data["truncated"] = True
    return _success(data)

