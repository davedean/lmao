from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from lmao.debug_log import DebugLogger
from lmao.plugins import PLUGIN_API_VERSION
from lmao.tools import normalize_path_for_output, safe_target_path

PLUGIN = {
    "name": "mkdir",
    "description": "Create a directory (recursively).",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": True,
    "allow_in_read_only": False,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "target directory path",
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
    try:
        target_path = safe_target_path(target or ".", base, extra_roots)
    except Exception:
        return _error(f"target path '{target}' escapes working directory")

    try:
        target_path.mkdir(parents=True, exist_ok=True)
        data = {"path": normalize_path_for_output(target_path, base), "created": True}
        return _success(data)
    except Exception as exc:
        return _error(f"unable to create directory '{target}': {exc}")
