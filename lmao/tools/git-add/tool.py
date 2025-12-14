from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.debug_log import DebugLogger
from lmao.plugin_helpers import find_repo_root, safe_target_path

PLUGIN = {
    "name": "git_add",
    "description": "Stage a file or directory with git add.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": True,
    "allow_in_read_only": False,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "input_schema": "path to add",
    "usage": "{'tool':'git_add','target':'./path','args':''}",
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

    try:
        target_path = safe_target_path(target or ".", base, extra_roots)
    except Exception as exc:
        return _error(f"invalid target '{target}': {exc}")

    try:
        rel = str(target_path.relative_to(repo_root))
    except Exception:
        rel = str(target_path)

    try:
        result = subprocess.run(
            ["git", "add", rel],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return _error(f"git add exception: {exc}")

    if result.returncode != 0:
        return _error(f"git add failed: {result.stderr.strip() or result.stdout.strip()}")

    data = {"path": rel, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    return _success(data)
