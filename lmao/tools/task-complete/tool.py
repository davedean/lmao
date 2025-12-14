from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION

PLUGIN = {
    "name": "complete_task",
    "description": "Mark a task done by id.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "task id in args (or target)",
    "usage": "{'tool':'complete_task','target':'','args':'task id'}",
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
    if task_manager is None:
        return _error("task list manager unavailable")
    payload = args if args is not None else ""
    if isinstance(payload, str):
        payload = payload.strip()
    if (not payload) and target:
        payload = str(target).strip()
    try:
        task_id = int(str(payload).strip())
    except Exception:
        return _error("task id must be an integer")
    message = task_manager.complete_task(task_id)
    if message.startswith("error:"):
        return _error(message)
    data = {"message": message, "tasks": task_manager.to_payload()}
    return _success(data)
