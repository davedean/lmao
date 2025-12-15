from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.plugin_helpers import find_repo_root, safe_target_path

PLUGIN = {
    "name": "git_diff",
    "description": "Show git diff; supports args: '--staged' and/or '--stat'.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "target optional path; args: '--staged' and/or '--stat'",
    "usage": [
        "{'tool':'git_diff','target':'','args':''}",
        "{'tool':'git_diff','target':'','args':'--staged'}",
        "{'tool':'git_diff','target':'','args':'--stat'}",
        "{'tool':'git_diff','target':'path/to/file','args':'--staged'}",
    ],
}


def _success(data: dict) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": False, "error": message}, ensure_ascii=False)


def _parse_args(arg_str: str) -> tuple[bool, bool] | tuple[None, None]:
    tokens = [t for t in (arg_str or "").strip().split() if t]
    staged = False
    stat = False
    for token in tokens:
        if token in ("--staged", "staged"):
            staged = True
            continue
        if token in ("--stat", "stat"):
            stat = True
            continue
        return None, None
    return staged, stat


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

    staged, stat = _parse_args(str(args))
    if staged is None:
        return _error("unsupported args; allowed: '--staged' and/or '--stat'")

    try:
        target_path = safe_target_path(target or ".", base, extra_roots)
    except Exception as exc:
        return _error(f"invalid target '{target}': {exc}")

    rel_path: Optional[str]
    if target_path == base and target in ("", ".", "/", "\\"):
        rel_path = None
    else:
        try:
            rel_path = str(target_path.relative_to(repo_root))
        except Exception:
            rel_path = str(target_path)

    cmd = ["git", "diff"]
    if staged:
        cmd.append("--staged")
    if stat:
        cmd.append("--stat")
    if rel_path and rel_path not in (".", ""):
        cmd.extend(["--", rel_path])

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return _error(f"git diff exception: {exc}")

    if result.returncode != 0:
        return _error(f"git diff failed: {result.stderr.strip() or result.stdout.strip()}")

    content = (result.stdout or "").rstrip()
    if len(content) > 200_000:
        content = content[:200_000]
        truncated = True
    else:
        truncated = False

    data = {"repo_root": str(repo_root), "staged": staged, "stat": stat, "output": content}
    if rel_path:
        data["path"] = rel_path
    if truncated:
        data["truncated"] = True
    return _success(data)

