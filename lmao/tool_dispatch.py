from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .debug_log import DebugLogger
from .path_safety import safe_target_path
from .plugins import PluginTool
from .task_list import TaskListManager
from .tool_parsing import ToolCall


def json_success(tool: str, data: Any, note: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {"tool": tool, "success": True, "data": data}
    if note:
        payload["note"] = note
    return json.dumps(payload, ensure_ascii=False)


def json_error(tool: str, message: str) -> str:
    return json.dumps({"tool": tool, "success": False, "error": message}, ensure_ascii=False)


def get_allowed_tools(
    read_only: bool, yolo_enabled: bool, plugins: Optional[Sequence[PluginTool]] = None
) -> List[str]:
    allowed: List[str] = []
    if plugins:
        for plugin in plugins:
            if read_only:
                if plugin.allow_in_read_only:
                    allowed.append(plugin.name)
                continue
            if yolo_enabled and (plugin.allow_in_yolo or plugin.allow_in_normal):
                allowed.append(plugin.name)
                continue
            if not yolo_enabled and plugin.allow_in_normal:
                allowed.append(plugin.name)
    return allowed


def plugin_allowed(plugin: PluginTool, read_only: bool, yolo_enabled: bool) -> bool:
    if read_only:
        return plugin.allow_in_read_only
    if yolo_enabled:
        return plugin.allow_in_yolo or plugin.allow_in_normal
    return plugin.allow_in_normal


def confirm_plugin_run(plugin: PluginTool, target: str, args: str) -> bool:
    prompt = f"[{plugin.name}] allow run? target={target!r} args={args!r} [y/N]: "
    try:
        approval = input(prompt).strip().lower()
    except EOFError:
        approval = ""
    return approval.startswith("y")


def run_tool(
    tool_call: ToolCall,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    yolo_enabled: bool,
    read_only: bool = False,
    plugin_tools: Optional[Dict[str, PluginTool]] = None,
    task_manager: Optional[TaskListManager] = None,
    debug_logger: Optional[DebugLogger] = None,
) -> str:
    tool = tool_call.tool
    target = tool_call.target
    args = tool_call.args

    if debug_logger:
        debug_logger.log(
            "tool.dispatch",
            f"tool={tool} target={target!r} args={args!r} base={base} extra_roots={[str(r) for r in extra_roots]}",
        )

    plugins = plugin_tools or {}
    if tool not in plugins:
        return json_error(tool, f"unsupported tool '{tool}'")

    try:
        _target_path = safe_target_path(target or ".", base, extra_roots)
    except Exception as exc:
        if debug_logger:
            debug_logger.log("tool.error", f"tool={tool} target={target!r} path_escape_error={exc}")
        return json_error(tool, f"target path '{target}' escapes working directory")

    plugin = plugins[tool]
    if not plugin_allowed(plugin, read_only, yolo_enabled):
        mode = "read-only" if read_only else ("yolo" if yolo_enabled else "normal")
        return json_error(tool, f"plugin '{tool}' is not allowed in {mode} mode")
    if plugin.always_confirm:
        if not confirm_plugin_run(plugin, target, args):
            return json_error(tool, f"plugin '{tool}' not approved by user")
    try:
        result = plugin.handler(
            target,
            args,
            base,
            extra_roots,
            skill_roots,
            task_manager,
            debug_logger,
        )
        if not isinstance(result, str):
            return json_error(tool, "plugin handlers must return a JSON string")
        return result
    except Exception as exc:
        if debug_logger:
            debug_logger.log("plugin.error", f"tool={tool} path={plugin.path} error={exc}")
        return json_error(tool, f"plugin '{tool}' failed: {exc}")

