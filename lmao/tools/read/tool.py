from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

from lmao.plugins import PLUGIN_API_VERSION
from lmao.plugin_helpers import normalize_path_for_output, parse_line_range, safe_target_path

PLUGIN = {
    "name": "read",
    "description": "Read a file; supports optional line ranges like lines:10-20.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "target path; args optional line range string",
    "usage": [
        "{'tool':'read','target':'./filename','args':''}",
        "{'tool':'read','target':'./filename','args':'lines:10-40'}",
    ],
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
        return _error(f"file '{target}' not found")
    try:
        content = target_path.read_text(encoding="utf-8")
    except Exception as exc:
        return _error(f"unable to read '{target}': {exc}")

    line_range = parse_line_range(str(args))
    truncated = False
    if line_range:
        start, end = line_range
        lines = content.splitlines()
        selected = lines[start - 1 : end]
        content = "\n".join(selected)
        range_info = {"start": start, "end": end}
    else:
        range_info = None
    if len(content) > 200_000:
        content = content[:200_000]
        truncated = True

    data: dict = {
        "path": normalize_path_for_output(target_path, base),
        "content": content,
    }
    if range_info:
        data["lines"] = range_info
    if truncated:
        data["truncated"] = True
    return _success(data)
