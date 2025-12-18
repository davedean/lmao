from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION, get_discovered_tool_registry

PLUGIN = {
    "name": "tools_guide",
    "description": "Show usage and detailed help for an available tool by name.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": (
        "v2 args: {tool?:string, tools?:string[], list_only?:bool, max_known?:int}; "
        "v1 args: tool name string"
    ),
    "usage": [
        "{\"tool\":\"tools_guide\",\"target\":\"\",\"args\":\"async_tail\"}",
        "{\"tool\":\"tools_guide\",\"target\":\"\",\"args\":{\"tool\":\"async_tail\"}}",
        "{\"tool\":\"tools_guide\",\"target\":\"\",\"args\":{\"tools\":[\"read\",\"grep\"]}}",
        "{\"tool\":\"tools_guide\",\"target\":\"\",\"args\":{\"tool\":\"not_a_tool\",\"list_only\":true}}",
    ],
}


def _success(data: dict) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": False, "error": message}, ensure_ascii=False)


def _coerce_tool_names(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        name = raw.strip()
        return [name] if name else []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(raw).strip()
    return [text] if text else []


def _tools_guide_data(tool) -> dict:  # type: ignore[no-untyped-def]
    return {
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


def _unknown_tool_error(
    tool_name: str,
    *,
    known: list[str],
    list_only: bool,
    max_known: int,
) -> str:
    if list_only:
        return f"unknown tool '{tool_name}'"
    if max_known <= 0:
        return f"unknown tool '{tool_name}' (known tools available; set max_known>0 to list)"
    preview = ", ".join(known[:max_known])
    suffix = " ..." if len(known) > max_known else ""
    return f"unknown tool '{tool_name}'. known: {preview}{suffix}"


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
    list_only = False
    max_known = 20
    tool_names: list[str] = []

    if isinstance(args, dict):
        list_only = bool(args.get("list_only", False))
        raw_max_known = args.get("max_known", max_known)
        try:
            max_known = int(raw_max_known)
        except Exception:
            max_known = 20
        tool_names = _coerce_tool_names(args.get("tools"))
        if not tool_names:
            tool_names = _coerce_tool_names(args.get("tool") or args.get("name") or target or "")
    else:
        tool_names = _coerce_tool_names(args or target or "")
    if not tool_names:
        return _error("tools_guide args must be a tool name (or args.tools=[...])")

    registry = get_discovered_tool_registry()
    known = sorted(registry.keys())

    if len(tool_names) > 1:
        results = []
        for name in tool_names:
            tool = registry.get(name)
            if tool is None:
                results.append({"name": name, "success": False, "error": _unknown_tool_error(name, known=known, list_only=True, max_known=0)})
                continue
            results.append({"name": name, "success": True, "data": _tools_guide_data(tool)})
        return _success({"tools": results})

    tool_name = tool_names[0]
    tool = registry.get(tool_name)
    if tool is None:
        return _error(_unknown_tool_error(tool_name, known=known, list_only=list_only, max_known=max_known))

    return _success(_tools_guide_data(tool))
