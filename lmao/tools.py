from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .debug_log import DebugLogger
from .plugins import PluginTool
from .task_list import TaskListManager

BUILTIN_TOOLS: set[str] = {
    "read",
    "write",
    "mkdir",
    "move",
    "ls",
    "find",
    "grep",
    "list_skills",
    "add_task",
    "complete_task",
    "delete_task",
    "list_tasks",
    "git_add",
    "git_commit",
    "bash",
}
READ_ONLY_DISABLED_TOOLS: set[str] = set()
TOOL_ORDER: list[str] = []

def json_success(tool: str, data: Any, note: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {"tool": tool, "success": True, "data": data}
    if note:
        payload["note"] = note
    return json.dumps(payload, ensure_ascii=False)


def json_error(tool: str, message: str) -> str:
    return json.dumps({"tool": tool, "success": False, "error": message}, ensure_ascii=False)


def normalize_path_for_output(path: Path, base: Path) -> str:
    try:
        rel = path.relative_to(base)
    except Exception:
        rel = path
    rel_str = str(rel)
    if path.is_dir() and not rel_str.endswith("/"):
        rel_str += "/"
    return rel_str


@dataclass
class ToolCall:
    tool: str
    target: str
    args: str

    @classmethod
    def from_raw_message(cls, raw_text: str, allowed_tools: Optional[Sequence[str]] = None) -> Optional["ToolCall"]:
        calls = parse_tool_calls(raw_text, allowed_tools=allowed_tools)
        return calls[0] if calls else None


def get_allowed_tools(read_only: bool, yolo_enabled: bool, plugins: Optional[Sequence["PluginTool"]] = None) -> List[str]:
    allowed: List[str] = []
    disabled = READ_ONLY_DISABLED_TOOLS if read_only else set()
    for tool in TOOL_ORDER:
        if tool in disabled:
            continue
        allowed.append(tool)
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


def iter_json_candidates(raw_text: str) -> List[str]:
    candidates: List[str] = []
    stripped = raw_text.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL)
    candidates.extend(fenced)
    candidates.extend(extract_braced_objects(raw_text))
    if stripped:
        candidates.append(stripped)
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        unique.append(cand)
    return unique


def parse_tool_calls(raw_text: str, allowed_tools: Optional[Sequence[str]] = None) -> List[ToolCall]:
    calls: List[ToolCall] = []
    allowed = set(allowed_tools) if allowed_tools is not None else BUILTIN_TOOLS
    for candidate in iter_json_candidates(raw_text):
        parsed = load_candidate(candidate)
        if not parsed:
            continue
        if isinstance(parsed, list):
            parsed_items = parsed
        else:
            parsed_items = [parsed]
        for obj in parsed_items:
            if not isinstance(obj, dict):
                continue
            tool = str(obj.get("tool", "")).strip()
            if tool not in allowed:
                continue
            target = str(obj.get("target", "") or "").strip()
            args = obj.get("args", "")
            args_str = args if isinstance(args, str) else json.dumps(args)
            calls.append(ToolCall(tool=tool, target=target, args=args_str))
    return calls


def covers_message(candidate: str, raw_text: str) -> bool:
    stripped = raw_text.strip()
    cand = candidate.strip()
    if not cand:
        return False
    if cand == stripped:
        return True
    fenced_json = f"```json\n{cand}\n```"
    fenced_plain = f"```\n{cand}\n```"
    return stripped == fenced_json or stripped == fenced_plain


def extract_braced_objects(raw_text: str) -> List[str]:
    """Extract top-level brace-delimited JSON-ish substrings."""
    objs: List[str] = []
    depth = 0
    start = None
    for idx, ch in enumerate(raw_text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(raw_text[start : idx + 1].strip())
                    start = None
    return objs


def load_candidate(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    cleaned = text.strip()
    if not cleaned:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(cleaned)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            continue
    return None


def safe_target_path(target: str, base: Path, extra_roots: Sequence[Path]) -> Path:
    raw_path = Path(target).expanduser()
    target_path = raw_path.resolve() if raw_path.is_absolute() else (base / raw_path).resolve()

    allowed_roots = [base] + [p.resolve() for p in extra_roots]
    for root in allowed_roots:
        try:
            target_path.relative_to(root)
            return target_path
        except Exception:
            continue
    raise ValueError("path escapes allowed roots")


def parse_line_range(arg: str) -> Optional[tuple]:
    match = re.search(r"lines?[:=]\s*(\d+)(?:[-:](\d+))?", arg)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    if start < 1:
        start = 1
    if end < start:
        end = start
    return start, end


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

    if tool not in BUILTIN_TOOLS and tool not in plugins:
        return json_error(tool, f"unsupported tool '{tool}'")

    if read_only and tool in READ_ONLY_DISABLED_TOOLS:
        return json_error(tool, f"tool '{tool}' is disabled in read-only mode")

    try:
        target_path = safe_target_path(target or ".", base, extra_roots)
    except Exception as exc:
        if debug_logger:
            debug_logger.log("tool.error", f"tool={tool} target={target!r} path_escape_error={exc}")
        return json_error(tool, f"target path '{target}' escapes working directory")

    if tool in plugins:
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

    return json_error(tool, "unexpected tool handling branch")


def summarize_output(output: str, max_lines: int = 8, max_chars: int = 400) -> str:
    if max_lines <= 0 and max_chars <= 0:
        return ""
    lines = output.splitlines()
    head = lines[:max_lines] if max_lines > 0 else lines
    summary = "\n".join(head)
    if max_lines > 0 and len(lines) > max_lines:
        summary += "\n...[truncated]"
    if max_chars > 0 and len(summary) > max_chars:
        summary = summary[:max_chars] + "\n...[truncated]"
    return summary


def normalize_task_text(text: str) -> str:
    """Normalize task text: remove numbering, collapse whitespace, avoid newlines."""
    cleaned = text.replace("\n", " ").replace("\r", " ")
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def display_path_for_output(path: str) -> str:
    """Quote paths with whitespace so the model can reuse them accurately."""
    return f'"{path}"' if any(ch.isspace() for ch in path) else path
