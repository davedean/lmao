from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION, get_discovered_tool_registry

PLUGIN = {
    "name": "tools_list",
    "description": "List available tools (compact: name + description).",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "v2 args optional: {pattern?:string} (filters by substring on tool name; empty list if no matches)",
    "usage": [
        "{\"tool\":\"tools_list\",\"target\":\"\",\"args\":\"\"}",
        "{\"tool\":\"tools_list\",\"target\":\"\",\"args\":{}}",
        "{\"tool\":\"tools_list\",\"target\":\"\",\"args\":{\"pattern\":\"git\"}}",
        "{\"tool\":\"tools_list\",\"target\":\"\",\"args\":{\"pattern\":\"nope\"}}",
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
    debug_logger: Optional[object] = None,
    meta: Optional[dict] = None,
) -> str:
    pattern = ""
    if isinstance(args, dict):
        pattern = str(args.get("pattern") or "").strip()
    elif isinstance(args, str):
        pattern = str(args).strip()
    if not pattern and target:
        pattern = str(target).strip()

    registry = get_discovered_tool_registry()
    tools = sorted(registry.values(), key=lambda tool: tool.name)
    if pattern:
        needle = pattern.lower()
        tools = [tool for tool in tools if needle in tool.name.lower()]

    items = [{"name": tool.name, "description": tool.description} for tool in tools]
    return _success({"tools": items, "pattern": pattern})
