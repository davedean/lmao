from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.debug_log import DebugLogger
from lmao.plugin_helpers import normalize_path_for_output, safe_target_path

PLUGIN = {
    "name": "grep",
    "description": "Search for a substring in files (skips dotfiles, truncates after 200 matches).",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "target file/dir; args is pattern string",
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
    pattern = str(args)
    try:
        target_path = safe_target_path(target or ".", base, extra_roots)
    except Exception:
        return _error(f"target path '{target}' escapes working directory")

    if not target_path.exists():
        return _error(f"path '{target}' not found")

    if target_path.is_dir():
        candidates = [p for p in target_path.rglob("*") if p.is_file() and not p.name.startswith(".")]
    else:
        candidates = [target_path]

    matches = []
    truncated = False
    for candidate in candidates:
        try:
            content = candidate.read_text(encoding="utf-8")
        except Exception:
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            if pattern in line:
                matches.append({
                    "path": normalize_path_for_output(candidate, base),
                    "line": lineno,
                    "text": line,
                })
            if len(matches) >= 200:
                truncated = True
                break
        if truncated:
            break

    data = {"path": normalize_path_for_output(target_path, base), "pattern": pattern, "matches": matches}
    if truncated:
        data["truncated"] = True
    return _success(data)
