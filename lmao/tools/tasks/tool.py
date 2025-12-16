from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugin_helpers import normalize_task_text
from lmao.plugins import PLUGIN_API_VERSION

PLUGINS = [
    {
        "name": "add_task",
        "description": "Add a task to the active task list.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "v2 args: {task:'...'}; v1 args: task string; target fallback supported",
        "usage": [
            "{\"tool\":\"add_task\",\"target\":\"\",\"args\":\"task description\"}",
            "{\"tool\":\"add_task\",\"target\":\"\",\"args\":{\"task\":\"task description\"}}",
        ],
    },
    {
        "name": "complete_task",
        "description": "Mark a task done by id.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "v2 args: {id: int}; v1 args: id string; target fallback supported",
        "usage": [
            "{\"tool\":\"complete_task\",\"target\":\"\",\"args\":\"1\"}",
            "{\"tool\":\"complete_task\",\"target\":\"\",\"args\":{\"id\":1}}",
        ],
    },
    {
        "name": "delete_task",
        "description": "Delete a task by id.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "v2 args: {id: int}; v1 args: id string; target fallback supported",
        "usage": [
            "{\"tool\":\"delete_task\",\"target\":\"\",\"args\":\"1\"}",
            "{\"tool\":\"delete_task\",\"target\":\"\",\"args\":{\"id\":1}}",
        ],
    },
    {
        "name": "list_tasks",
        "description": "Render the current task list.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "none (v2 args ignored)",
        "usage": [
            "{\"tool\":\"list_tasks\",\"target\":\"\",\"args\":\"\"}",
            "{\"tool\":\"list_tasks\",\"target\":\"\",\"args\":{}}",
        ],
    },
]


def _success(tool: str, data: dict) -> str:
    return json.dumps({"tool": tool, "success": True, "data": data}, ensure_ascii=False)


def _error(tool: str, message: str) -> str:
    return json.dumps({"tool": tool, "success": False, "error": message}, ensure_ascii=False)


def _parse_task_id(payload: object, target: str) -> Optional[int]:
    raw = payload if payload is not None else ""
    if isinstance(raw, str):
        raw = raw.strip()
    if (not raw) and target:
        raw = str(target).strip()
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        if raw.startswith("{") and raw.endswith("}"):
            try:
                obj = json.loads(raw)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                for key in ("id", "task_id", "taskId", "args", "target"):
                    value = obj.get(key)
                    if isinstance(value, int):
                        return value
                    if isinstance(value, str) and value.strip().isdigit():
                        return int(value.strip())
        try:
            return int(raw)
        except Exception:
            return None
    return None


def run(
    tool_name: str,
    target: str,
    args: Any,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    task_manager=None,
    debug_logger: Optional[object] = None,
    meta: Optional[dict] = None,
) -> str:
    if task_manager is None:
        return _error(tool_name, "task list manager unavailable")

    if tool_name == "add_task":
        payload: object = args if args is not None else ""
        if isinstance(payload, dict):
            payload = payload.get("task") or payload.get("text") or payload.get("args") or payload.get("content") or ""
        if isinstance(payload, str):
            payload = payload.strip()
        if (not payload) and target:
            payload = str(target).strip()
        text = normalize_task_text(str(payload).strip())
        if not text:
            return _error(tool_name, "task text is required")
        task = task_manager.add_task(text)
        data = {
            "task": {"id": task.id, "text": task.text, "done": task.done},
            "tasks": task_manager.to_payload(),
        }
        return _success(tool_name, data)

    if tool_name == "complete_task":
        payload: object = args
        if isinstance(payload, dict):
            payload = payload.get("id") or payload.get("task_id") or payload.get("taskId") or payload.get("args") or payload.get("target") or payload
        task_id = _parse_task_id(payload, target)
        if task_id is None:
            return _error(tool_name, "task id must be an integer")
        message = task_manager.complete_task(task_id)
        if message.startswith("error:"):
            return _error(tool_name, message)
        return _success(tool_name, {"message": message, "tasks": task_manager.to_payload()})

    if tool_name == "delete_task":
        payload = args
        if isinstance(payload, dict):
            payload = payload.get("id") or payload.get("task_id") or payload.get("taskId") or payload.get("args") or payload.get("target") or payload
        task_id = _parse_task_id(payload, target)
        if task_id is None:
            return _error(tool_name, "task id must be an integer")
        message = task_manager.delete_task(task_id)
        if message.startswith("error:"):
            return _error(tool_name, message)
        return _success(tool_name, {"message": message, "tasks": task_manager.to_payload()})

    if tool_name == "list_tasks":
        data = {"tasks": task_manager.to_payload(), "render": task_manager.render_tasks()}
        return _success(tool_name, data)

    return _error(tool_name, f"unsupported tool '{tool_name}'")
