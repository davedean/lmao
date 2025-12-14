from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from lmao.context import find_repo_root
from lmao.debug_log import DebugLogger
from lmao.plugins import PLUGIN_API_VERSION

PLUGIN = {
    "name": "git_commit",
    "description": "Create a git commit with the provided message.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": True,
    "allow_in_read_only": False,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "input_schema": "commit message in args",
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
    debug_logger: Optional[DebugLogger] = None,
) -> str:
    repo_root = find_repo_root(base)
    if not (repo_root / ".git").exists():
        return _error("not inside a git repository")

    message = str(args or "").strip()
    if not message:
        return _error("commit message is required")

    try:
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return _error(f"git commit exception: {exc}")

    if result.returncode != 0:
        return _error(f"git commit failed: {result.stderr.strip() or result.stdout.strip()}")

    data = {"message": result.stdout.strip() or "ok", "stderr": result.stderr.strip()}
    return _success(data)
