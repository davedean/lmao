from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.debug_log import DebugLogger
from lmao.plugin_helpers import (
    normalize_path_for_output,
    safe_target_path,
    validate_skill_write_target,
)

PLUGIN = {
    "name": "write",
    "description": "Write text to a file (creates parents).",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": True,
    "allow_in_read_only": False,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "target path; args is file content",
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
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and target_path.is_dir():
            return _error(f"'{target}' is a directory; use mkdir instead")
        if str(target).rstrip().endswith(("/", "\\")):
            return _error(f"'{target}' looks like a directory path; use mkdir instead")
        skill_error = validate_skill_write_target(target_path, skill_roots)
        if skill_error:
            return _error(skill_error)
        with target_path.open("w", encoding="utf-8") as fh:
            fh.write(args or "")
        data = {
            "path": normalize_path_for_output(target_path, base),
            "bytes": len(args or ""),
        }
        return _success(data)
    except Exception as exc:
        return _error(f"unable to write '{target}': {exc}")
