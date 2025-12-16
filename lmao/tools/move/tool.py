from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.plugin_helpers import normalize_path_for_output, safe_target_path, validate_skill_write_target

PLUGIN = {
    "name": "move",
    "description": "Move/rename a file or directory (no overwrite).",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": True,
    "allow_in_read_only": False,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "v2 args: {to:'./new_path'}; v1 args: destination string; target: source",
    "usage": [
        "{\"tool\":\"move\",\"target\":\"./old_path\",\"args\":\"./new_path\"}",
        "{\"tool\":\"move\",\"target\":\"./old_path\",\"args\":{\"to\":\"./new_path\"}}",
    ],
}


def _success(data: dict) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": False, "error": message}, ensure_ascii=False)


def run(
    target: str,
    args: Any,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    task_manager=None,
    debug_logger: Optional[object] = None,
    meta: Optional[dict] = None,
) -> str:
    try:
        source_path = safe_target_path(target or ".", base, extra_roots)
    except Exception:
        return _error(f"target path '{target}' escapes working directory")
    if isinstance(args, dict):
        args = args.get("to") or args.get("dest") or args.get("destination") or ""
    try:
        dest_path = safe_target_path(str(args), base, extra_roots)
    except Exception:
        return _error(f"destination path '{args}' escapes working directory")

    try:
        if not source_path.exists():
            return _error(f"source '{target}' not found")
        if source_path.is_file():
            skill_error = validate_skill_write_target(dest_path, skill_roots)
            if skill_error:
                return _error(skill_error)
        if dest_path.exists():
            return _error(f"destination '{args}' already exists")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(dest_path)
        data = {
            "from": normalize_path_for_output(source_path, base),
            "to": normalize_path_for_output(dest_path, base),
        }
        return _success(data)
    except Exception as exc:
        return _error(f"unable to move '{target}' to '{args}': {exc}")
