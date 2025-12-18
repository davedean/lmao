from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
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
    "input_schema": "v2 args: {content:'...'}; v1 args: content string; target: path",
    "usage": [
        "{\"tool\":\"write\",\"target\":\"./filename\",\"args\":\"new content\"}",
        "{\"tool\":\"write\",\"target\":\"./filename\",\"args\":{\"content\":\"new content\"}}",
    ],
}


ESCAPE_HINTS = ("\\n", "\\r", "\\t", "\\\"", "\\'", "\\b", "\\f", "\\u")


def _should_decode_escaped_text(text: str) -> bool:
    if "\n" in text:
        return False
    return any(hint in text for hint in ESCAPE_HINTS)


def _decode_escaped_text(text: str) -> str:
    try:
        decoded = text.encode("utf-8").decode("unicode_escape")
    except Exception:
        return text
    return decoded


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
    if not str(target or "").strip():
        return _error(
            "missing target file path; set target to a relative file path like 'notes.txt' (mkdir is for directories)"
        )
    try:
        target_path = safe_target_path(target, base, extra_roots)
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
        if isinstance(args, dict):
            content = str(args.get("content") or args.get("text") or "")
        else:
            content = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        if isinstance(content, str) and _should_decode_escaped_text(content):
            content = _decode_escaped_text(content)
        with target_path.open("w", encoding="utf-8") as fh:
            fh.write(content or "")
        data = {
            "path": normalize_path_for_output(target_path, base),
            "bytes": len(content or ""),
        }
        return _success(data)
    except Exception as exc:
        return _error(f"unable to write '{target}': {exc}")
