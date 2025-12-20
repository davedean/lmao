from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .debug_log import DebugLogger
from .path_safety import safe_target_path
from .plugins import PluginTool
from .tool_parsing import ToolCall
from .runtime_tools import RuntimeContext, RuntimeTool, runtime_tool_allowed


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
    if not plugins:
        return []
    if yolo_enabled and not read_only:
        return [plugin.name for plugin in plugins]
    return [plugin.name for plugin in plugins if plugin_allowed(plugin, read_only, yolo_enabled)]


def plugin_allowed(plugin: PluginTool, read_only: bool, yolo_enabled: bool) -> bool:
    if read_only:
        return plugin.allow_in_read_only
    if yolo_enabled:
        return True
    return plugin.allow_in_normal


def _format_args_for_prompt(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False)
    except Exception:
        return str(args)


def confirm_plugin_run(plugin: PluginTool, target: str, args: Any) -> bool:
    prompt = f"[{plugin.name}] allow run? target={target!r} args={_format_args_for_prompt(args)!r} [y/N]: "
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
    runtime_tools: Optional[Dict[str, RuntimeTool]] = None,
    runtime_context: Optional[RuntimeContext] = None,
    task_manager: Optional[Any] = None,
    debug_logger: Optional[DebugLogger] = None,
) -> str:
    tool = tool_call.tool
    target = tool_call.target
    args = tool_call.args
    meta = getattr(tool_call, "meta", None)

    if debug_logger:
        debug_logger.log(
            "tool.dispatch",
            f"tool={tool} target={target!r} args={_format_args_for_prompt(args)!r} meta={meta!r} base={base} extra_roots={[str(r) for r in extra_roots]}",
        )

    plugins = plugin_tools or {}
    rt_tools = runtime_tools or {}
    if tool in rt_tools:
        rt = rt_tools[tool]
        if not runtime_tool_allowed(rt, read_only=read_only, yolo_enabled=yolo_enabled):
            mode = "read-only" if read_only else ("yolo" if yolo_enabled else "normal")
            return json_error(tool, f"runtime tool '{tool}' is not allowed in {mode} mode")
        if rt.is_destructive and read_only:
            return json_error(tool, "runtime tool is destructive and not allowed in read-only mode")
        if runtime_context is None:
            return json_error(tool, f"runtime context missing for '{tool}'")
        try:
            return rt.handler(runtime_context, target, args, meta)
        except Exception as exc:
            if debug_logger:
                debug_logger.log("runtime_tool.error", f"tool={tool} error={exc}")
            return json_error(tool, f"runtime tool '{tool}' failed: {exc}")

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
        if not yolo_enabled:
            if runtime_context and runtime_context.headless:
                return json_error(
                    tool,
                    f"plugin '{tool}' requires confirmation but headless mode is active",
                )
            if not confirm_plugin_run(plugin, target, args):
                return json_error(tool, f"plugin '{tool}' not approved by user")
    try:
        handler = plugin.handler
        try:
            sig = inspect.signature(handler)
            params = list(sig.parameters.values())
            accepts_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            accepts_meta = accepts_varargs or len(params) >= 8
        except Exception:
            accepts_meta = False

        if accepts_meta:
            result = handler(
                target,
                args,
                base,
                extra_roots,
                skill_roots,
                task_manager,
                debug_logger,
                meta,
            )
        else:
            result = handler(
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
