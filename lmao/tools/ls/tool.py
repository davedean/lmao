from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.plugin_helpers import normalize_path_for_output, safe_target_path

PLUGIN = {
    "name": "ls",
    "description": "List a file or directory (non-recursive).",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "target path (file or directory)",
    "usage": "{'tool':'ls','target':'.','args':''}",
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
    try:
        target_path = safe_target_path(target or ".", base, extra_roots)
    except Exception:
        return _error(f"target path '{target}' escapes working directory")

    if not target_path.exists():
        return _error(f"path '{target}' not found")
    try:
        if target_path.is_file():
            entries = [{
                "name": target_path.name,
                "path": normalize_path_for_output(target_path, base),
                "type": "file",
            }]
            data = {"path": normalize_path_for_output(target_path, base), "entries": entries}
            return _success(data)
        entries_list = []
        for p in sorted(target_path.iterdir(), key=lambda x: x.name):
            entries_list.append({
                "name": p.name,
                "path": normalize_path_for_output(p, base),
                "type": "dir" if p.is_dir() else "file",
            })
        data = {"path": normalize_path_for_output(target_path, base), "entries": entries_list}
        return _success(data)
    except Exception as exc:
        return _error(f"unable to list '{target}': {exc}")
