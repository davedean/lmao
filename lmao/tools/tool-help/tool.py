from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION, get_discovered_tool_registry

PLUGIN = {
    "name": "tool_help",
    "description": "Show usage and detailed help for an available tool by name.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "args: tool name (e.g., 'async_tail'); target ignored",
    "usage": "{'tool':'tool_help','target':'','args':'async_tail'}",
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
    tool_name = str(args or target or "").strip()
    if not tool_name:
        return _error("tool_help args must be a tool name")

    registry = get_discovered_tool_registry()
    tool = registry.get(tool_name)
    if tool is None:
        known = sorted(registry.keys())
        return _error(f"unknown tool '{tool_name}'. known: {', '.join(known[:50])}" + (" ..." if len(known) > 50 else ""))

    data = {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "usage": tool.usage_examples,
        "details": tool.details,
        "safety": {
            "is_destructive": tool.is_destructive,
            "always_confirm": tool.always_confirm,
            "allow_in_read_only": tool.allow_in_read_only,
            "allow_in_normal": tool.allow_in_normal,
            "allow_in_yolo": tool.allow_in_yolo,
        },
        "source": {"path": str(tool.path)},
    }
    return _success(data)
