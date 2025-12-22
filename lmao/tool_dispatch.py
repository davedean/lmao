from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .debug_log import DebugLogger
from .hooks import (
    ErrorHookContext,
    ErrorHookTypes,
    LoggingHookContext,
    LoggingHookTypes,
    ToolHookContext,
    ToolHookTypes,
    get_global_hook_registry,
)
from .path_safety import safe_target_path
from .plugin_helpers import set_yolo_path_mode
from .plugins import PluginTool
from .tool_parsing import ToolCall
from .runtime_tools import RuntimeContext, RuntimeTool, runtime_tool_allowed


def json_success(tool: str, data: Any, note: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {"tool": tool, "success": True, "data": data}
    if note:
        payload["note"] = note
    return json.dumps(payload, ensure_ascii=False)


def json_error(tool: str, message: str) -> str:
    return json.dumps(
        {"tool": tool, "success": False, "error": message}, ensure_ascii=False
    )


def get_allowed_tools(
    read_only: bool, yolo_enabled: bool, plugins: Optional[Sequence[PluginTool]] = None
) -> List[str]:
    if not plugins:
        return []
    if yolo_enabled and not read_only:
        return [plugin.name for plugin in plugins if plugin.visible_to_agent]
    return [
        plugin.name
        for plugin in plugins
        if plugin_allowed(plugin, read_only, yolo_enabled) and plugin.visible_to_agent
    ]


def plugin_allowed(plugin: PluginTool, read_only: bool, yolo_enabled: bool) -> bool:
    if read_only:
        return plugin.allow_in_read_only
    if yolo_enabled:
        return True
    return plugin.allow_in_normal


def runtime_tool_allowed_visibility(
    runtime_tool: RuntimeTool, read_only: bool, yolo_enabled: bool
) -> bool:
    """Check if a runtime tool is allowed and visible to the agent."""
    if not runtime_tool_visible_to_agent(runtime_tool):
        return False
    return runtime_tool_allowed(
        runtime_tool, read_only=read_only, yolo_enabled=yolo_enabled
    )


def runtime_tool_visible_to_agent(runtime_tool: RuntimeTool) -> bool:
    """Check if a runtime tool is visible to the agent."""
    return runtime_tool.visible_to_agent


def get_allowed_runtime_tools(
    runtime_tools: Dict[str, RuntimeTool], read_only: bool, yolo_enabled: bool
) -> List[str]:
    """Get list of runtime tool names that are allowed and visible to the agent."""
    return [
        name
        for name, tool in runtime_tools.items()
        if runtime_tool_allowed_visibility(
            tool, read_only=read_only, yolo_enabled=yolo_enabled
        )
    ]


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
    hook_registry = (
        runtime_context.hook_registry
        if runtime_context and runtime_context.hook_registry
        else get_global_hook_registry()
    )

    if debug_logger:
        debug_logger.log(
            "tool.dispatch",
            f"tool={tool} target={target!r} args={_format_args_for_prompt(args)!r} meta={meta!r} base={base} extra_roots={[str(r) for r in extra_roots]}",
        )

    tool_context = ToolHookContext(
        hook_type=ToolHookTypes.PRE_TOOL_VALIDATION,
        runtime_state={
            "tool_call": tool_call,
            "base": base,
            "extra_roots": extra_roots,
            "skill_roots": skill_roots,
            "yolo_enabled": yolo_enabled,
            "read_only": read_only,
        },
        tool_name=tool,
        tool_target=target,
        tool_args=args,
        tool_call=tool_call,
    )
    pre_validation = hook_registry.execute_hooks(
        ToolHookTypes.PRE_TOOL_VALIDATION, tool_context
    )
    if pre_validation.modified_context:
        tool_context = pre_validation.modified_context
    tool_call, tool, target, args = _apply_tool_context_updates(tool_call, tool_context)
    if pre_validation.should_cancel:
        return json_error(tool, "tool execution cancelled by hook")
    if pre_validation.should_skip:
        return _hook_skip_result(tool, pre_validation)

    plugins = plugin_tools or {}
    rt_tools = runtime_tools or {}
    if tool in rt_tools:
        rt = rt_tools[tool]
        permission_context = tool_context.with_hook_type(
            ToolHookTypes.PRE_PERMISSION_CHECK
        )
        perm_result = hook_registry.execute_hooks(
            ToolHookTypes.PRE_PERMISSION_CHECK, permission_context
        )
        if perm_result.modified_context:
            tool_context = perm_result.modified_context
        tool_call, tool, target, args = _apply_tool_context_updates(
            tool_call, tool_context
        )
        if perm_result.should_cancel:
            return json_error(tool, "tool execution cancelled by hook")
        if perm_result.should_skip:
            return _hook_skip_result(tool, perm_result)

        if not runtime_tool_allowed(rt, read_only=read_only, yolo_enabled=yolo_enabled):
            mode = "read-only" if read_only else ("yolo" if yolo_enabled else "normal")
            _emit_error_hook(
                hook_registry,
                ErrorHookTypes.ON_TOOL_PERMISSION_ERROR,
                tool_context,
                f"runtime tool '{tool}' is not allowed in {mode} mode",
            )
            return json_error(
                tool, f"runtime tool '{tool}' is not allowed in {mode} mode"
            )
        if rt.is_destructive and read_only:
            _emit_error_hook(
                hook_registry,
                ErrorHookTypes.ON_TOOL_PERMISSION_ERROR,
                tool_context,
                "runtime tool is destructive and not allowed in read-only mode",
            )
            return json_error(
                tool, "runtime tool is destructive and not allowed in read-only mode"
            )
        if runtime_context is None:
            _emit_error_hook(
                hook_registry,
                ErrorHookTypes.ON_TOOL_EXECUTION_ERROR,
                tool_context,
                f"runtime context missing for '{tool}'",
            )
            return json_error(tool, f"runtime context missing for '{tool}'")
        try:
            hook_registry.execute_hooks(
                LoggingHookTypes.ON_TOOL_START,
                LoggingHookContext(
                    hook_type=LoggingHookTypes.ON_TOOL_START,
                    runtime_state=tool_context.runtime_state,
                    log_level="info",
                    log_message=f"runtime tool start: {tool}",
                    log_source="tool_dispatch",
                ),
            )
            exec_context = tool_context.with_hook_type(ToolHookTypes.PRE_TOOL_EXECUTION)
            pre_exec = hook_registry.execute_hooks(
                ToolHookTypes.PRE_TOOL_EXECUTION, exec_context
            )
            if pre_exec.modified_context:
                tool_context = pre_exec.modified_context
            tool_call, tool, target, args = _apply_tool_context_updates(
                tool_call, tool_context
            )
            if pre_exec.should_cancel:
                return json_error(tool, "tool execution cancelled by hook")
            if pre_exec.should_skip:
                return _hook_skip_result(tool, pre_exec)
            return rt.handler(runtime_context, target, args, meta)
        except Exception as exc:
            if debug_logger:
                debug_logger.log("runtime_tool.error", f"tool={tool} error={exc}")
            _emit_error_hook(
                hook_registry,
                ErrorHookTypes.ON_TOOL_EXECUTION_ERROR,
                tool_context,
                f"runtime tool '{tool}' failed: {exc}",
                exc=exc,
            )
            hook_registry.execute_hooks(
                LoggingHookTypes.ON_TOOL_ERROR,
                LoggingHookContext(
                    hook_type=LoggingHookTypes.ON_TOOL_ERROR,
                    runtime_state=tool_context.runtime_state,
                    log_level="error",
                    log_message=f"runtime tool error: {tool}",
                    log_source="tool_dispatch",
                ),
            )
            return json_error(tool, f"runtime tool '{tool}' failed: {exc}")

    if tool not in plugins:
        _emit_error_hook(
            hook_registry,
            ErrorHookTypes.ON_TOOL_VALIDATION_ERROR,
            tool_context,
            f"tool '{tool}' not found",
        )
        return json_error(tool, f"tool '{tool}' not found")

    path_context = tool_context.with_hook_type(ToolHookTypes.PRE_PATH_SAFETY_CHECK)
    path_result = hook_registry.execute_hooks(
        ToolHookTypes.PRE_PATH_SAFETY_CHECK, path_context
    )
    if path_result.modified_context:
        tool_context = path_result.modified_context
    tool_call, tool, target, args = _apply_tool_context_updates(tool_call, tool_context)
    if path_result.should_cancel:
        return json_error(tool, "tool execution cancelled by hook")
    if path_result.should_skip:
        return _hook_skip_result(tool, path_result)
    try:
        _target_path = safe_target_path(
            target or ".",
            base,
            extra_roots,
            allow_outside=yolo_enabled,
        )
    except Exception as exc:
        if debug_logger:
            debug_logger.log(
                "tool.error", f"tool={tool} target={target!r} path_escape_error={exc}"
            )
        _emit_error_hook(
            hook_registry,
            ErrorHookTypes.ON_TOOL_VALIDATION_ERROR,
            tool_context,
            f"target path '{target}' escapes working directory",
            exc=exc,
        )
        return json_error(tool, f"target path '{target}' escapes working directory")

    plugin = plugins[tool]
    tool_context = replace_tool_context(
        tool_context,
        plugin_info={
            "name": plugin.name,
            "path": str(plugin.path),
            "is_destructive": plugin.is_destructive,
        },
    )
    if not plugin_allowed(plugin, read_only, yolo_enabled):
        mode = "read-only" if read_only else ("yolo" if yolo_enabled else "normal")
        _emit_error_hook(
            hook_registry,
            ErrorHookTypes.ON_TOOL_PERMISSION_ERROR,
            tool_context,
            f"plugin '{tool}' is not allowed in {mode} mode",
        )
        return json_error(tool, f"plugin '{tool}' is not allowed in {mode} mode")
    if plugin.always_confirm:
        if not yolo_enabled:
            if runtime_context and runtime_context.headless:
                _emit_error_hook(
                    hook_registry,
                    ErrorHookTypes.ON_TOOL_PERMISSION_ERROR,
                    tool_context,
                    f"plugin '{tool}' requires confirmation but headless mode is active",
                )
                return json_error(
                    tool,
                    f"plugin '{tool}' requires confirmation but headless mode is active",
                )
            if not confirm_plugin_run(plugin, target, args):
                _emit_error_hook(
                    hook_registry,
                    ErrorHookTypes.ON_TOOL_PERMISSION_ERROR,
                    tool_context,
                    f"plugin '{tool}' not approved by user",
                )
                return json_error(tool, f"plugin '{tool}' not approved by user")
    try:
        hook_registry.execute_hooks(
            LoggingHookTypes.ON_TOOL_START,
            LoggingHookContext(
                hook_type=LoggingHookTypes.ON_TOOL_START,
                runtime_state=tool_context.runtime_state,
                log_level="info",
                log_message=f"tool start: {tool}",
                log_source="tool_dispatch",
            ),
        )
        exec_context = tool_context.with_hook_type(ToolHookTypes.PRE_TOOL_EXECUTION)
        pre_exec = hook_registry.execute_hooks(
            ToolHookTypes.PRE_TOOL_EXECUTION, exec_context
        )
        if pre_exec.modified_context:
            tool_context = pre_exec.modified_context
        tool_call, tool, target, args = _apply_tool_context_updates(
            tool_call, tool_context
        )
        if pre_exec.should_cancel:
            return json_error(tool, "tool execution cancelled by hook")
        if pre_exec.should_skip:
            return _hook_skip_result(tool, pre_exec)

        handler = plugin.handler
        try:
            sig = inspect.signature(handler)
            params = list(sig.parameters.values())
            accepts_varargs = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
            )
            accepts_meta = accepts_varargs or len(params) >= 8
        except Exception:
            accepts_meta = False

        set_yolo_path_mode(yolo_enabled)
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
            _emit_error_hook(
                hook_registry,
                ErrorHookTypes.ON_TOOL_EXECUTION_ERROR,
                tool_context,
                "plugin handlers must return a JSON string",
            )
            return json_error(tool, "plugin handlers must return a JSON string")
        tool_context = replace_tool_context(
            tool_context,
            tool_result=result,
            execution_time=None,
        )
        post_exec = hook_registry.execute_hooks(
            ToolHookTypes.POST_TOOL_EXECUTION,
            tool_context.with_hook_type(ToolHookTypes.POST_TOOL_EXECUTION),
        )
        if post_exec.modified_context:
            tool_context = post_exec.modified_context
        if post_exec.should_cancel:
            return json_error(tool, "tool execution cancelled by hook")
        if post_exec.should_skip:
            return _hook_skip_result(tool, post_exec)
        transform = hook_registry.execute_hooks(
            ToolHookTypes.POST_RESULT_TRANSFORM,
            tool_context.with_hook_type(ToolHookTypes.POST_RESULT_TRANSFORM),
        )
        if (
            transform.modified_context
            and transform.modified_context.tool_result is not None
        ):
            result = transform.modified_context.tool_result
        hook_registry.execute_hooks(
            LoggingHookTypes.ON_TOOL_SUCCESS,
            LoggingHookContext(
                hook_type=LoggingHookTypes.ON_TOOL_SUCCESS,
                runtime_state=tool_context.runtime_state,
                log_level="info",
                log_message=f"tool success: {tool}",
                log_source="tool_dispatch",
            ),
        )
        return result
    except Exception as exc:
        if debug_logger:
            debug_logger.log(
                "plugin.error", f"tool={tool} path={plugin.path} error={exc}"
            )
        _emit_error_hook(
            hook_registry,
            ErrorHookTypes.ON_TOOL_EXECUTION_ERROR,
            tool_context,
            f"plugin '{tool}' failed: {exc}",
            exc=exc,
        )
        hook_registry.execute_hooks(
            LoggingHookTypes.ON_TOOL_ERROR,
            LoggingHookContext(
                hook_type=LoggingHookTypes.ON_TOOL_ERROR,
                runtime_state=tool_context.runtime_state,
                log_level="error",
                log_message=f"tool error: {tool}",
                log_source="tool_dispatch",
            ),
        )
        return json_error(tool, f"plugin '{tool}' failed: {exc}")
    finally:
        set_yolo_path_mode(False)


def _apply_tool_context_updates(
    tool_call: ToolCall, context: ToolHookContext
) -> tuple[ToolCall, str, str, Any]:
    if context.tool_call is not None:
        tool_call = context.tool_call
    tool_name = context.tool_name or tool_call.tool
    target = tool_call.target
    if context.tool_target is not None:
        target = context.tool_target
    args = tool_call.args
    if context.tool_args is not None:
        args = context.tool_args
    if (
        tool_name != tool_call.tool
        or target != tool_call.target
        or args is not tool_call.args
    ):
        tool_call = ToolCall(
            tool=tool_name, target=target, args=args, meta=tool_call.meta
        )
    return tool_call, tool_name, target, args


def _hook_skip_result(tool: str, result: Any) -> str:
    data = getattr(result, "data", None)
    if isinstance(data, dict) and "result" in data:
        return data["result"]
    return json_error(tool, "tool execution skipped by hook")


def _emit_error_hook(
    hook_registry,
    hook_type: str,
    context: ToolHookContext,
    message: str,
    *,
    exc: Optional[Exception] = None,
) -> None:
    error_context = ErrorHookContext(
        hook_type=hook_type,
        runtime_state=context.runtime_state,
        error_type=type(exc).__name__ if exc else "ToolError",
        error_message=message,
        error_exception=exc,
        original_context=context,
    )
    hook_registry.execute_hooks(hook_type, error_context)


def replace_tool_context(context: ToolHookContext, **updates: Any) -> ToolHookContext:
    for key, value in updates.items():
        setattr(context, key, value)
    return context
