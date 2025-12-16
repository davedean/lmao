from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.plugin_helpers import (
    normalize_path_for_output,
    parse_line_range,
    safe_target_path,
    validate_skill_write_target,
)

PLUGIN = {
    "name": "patch",
    "description": "Patch an existing file by replacing a line range in-place.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": True,
    "allow_in_read_only": False,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "v2 args: {range:'lines:10-20',content:'...'} (or {start:int,end:int,content:'...'}); v1 args: JSON string or 'lines:..\\n...'",
    "usage": [
        "{\"tool\":\"patch\",\"target\":\"file.py\",\"args\":\"{\\\"range\\\":\\\"lines:10-12\\\",\\\"content\\\":\\\"new text\\\"}\"}",
        "{\"tool\":\"patch\",\"target\":\"file.py\",\"args\":{\"range\":\"lines:10-12\",\"content\":\"new text\"}}",
        "{\"tool\":\"patch\",\"target\":\"file.py\",\"args\":\"lines:10-12\\nnew text\"}",
    ],
}


def _success(data: dict) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": False, "error": message}, ensure_ascii=False)


@dataclass(frozen=True)
class _PatchSpec:
    start: int
    end: int
    content: str


def _load_spec(raw_args: str) -> Optional[_PatchSpec]:
    text = str(raw_args or "")
    # Preferred: JSON object string.
    try:
        obj: Any = json.loads(text)
    except Exception:
        obj = None

    if isinstance(obj, dict):
        content = obj.get("content", "")
        range_str = obj.get("range", "")
        if isinstance(obj.get("start"), int) and isinstance(obj.get("end"), int):
            start = int(obj["start"])
            end = int(obj["end"])
            if start < 1:
                start = 1
            if end < start:
                end = start
            return _PatchSpec(start=start, end=end, content=str(content or ""))
        line_range = parse_line_range(str(range_str or ""))
        if not line_range:
            return None
        start, end = line_range
        return _PatchSpec(start=start, end=end, content=str(content or ""))

    # Fallback: "lines:X-Y\n<content...>"
    if "\n" not in text:
        return None
    header, body = text.split("\n", 1)
    line_range = parse_line_range(header)
    if not line_range:
        return None
    start, end = line_range
    return _PatchSpec(start=start, end=end, content=body)


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
        target_path = safe_target_path(target or ".", base, extra_roots)
    except Exception:
        return _error(f"target path '{target}' escapes working directory")

    if not target_path.exists():
        return _error(f"file '{target}' not found")
    if target_path.is_dir() or str(target).rstrip().endswith(("/", "\\")):
        return _error(f"'{target}' is a directory; patch requires a file path")

    if isinstance(args, dict):
        raw_args = json.dumps(args, ensure_ascii=False)
    else:
        raw_args = str(args)
    spec = _load_spec(raw_args)
    if not spec:
        return _error("invalid patch args; expected JSON {'range':'lines:10-20','content':'...'} or 'lines:10-20\\n...'")

    skill_error = validate_skill_write_target(target_path, skill_roots)
    if skill_error:
        return _error(skill_error)

    try:
        original = target_path.read_text(encoding="utf-8")
    except Exception as exc:
        return _error(f"unable to read '{target}': {exc}")

    original_lines = original.splitlines(keepends=True)
    start_idx = spec.start - 1
    end_idx = spec.end
    if start_idx > len(original_lines):
        start_idx = len(original_lines)
    if end_idx > len(original_lines):
        end_idx = len(original_lines)

    replacement_lines = spec.content.splitlines(keepends=True)
    if spec.content and not spec.content.endswith(("\n", "\r")):
        if replacement_lines:
            replacement_lines[-1] = replacement_lines[-1].rstrip("\n")

    patched_lines = original_lines[:start_idx] + replacement_lines + original_lines[end_idx:]
    patched = "".join(patched_lines)

    try:
        target_path.write_text(patched, encoding="utf-8")
    except Exception as exc:
        return _error(f"unable to write '{target}': {exc}")

    data = {
        "path": normalize_path_for_output(target_path, base),
        "range": {"start": spec.start, "end": spec.end},
        "bytes": len(patched.encode("utf-8")),
    }
    return _success(data)
