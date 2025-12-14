from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.debug_log import DebugLogger
from lmao.plugin_helpers import normalize_task_text

PLUGIN = {
    "name": "add_task",
    "description": "Add a task to the active task list.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "task text in args (or target)",
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
    debug_logger: Optional[DebugLogger] = None,
) -> str:
    if task_manager is None:
        return _error("task list manager unavailable")
    payload = args if args is not None else ""
    if isinstance(payload, str):
        payload = payload.strip()
    if (not payload) and target:
        payload = str(target).strip()
    text = normalize_task_text(str(payload).strip())
    if not text:
        return _error("task text is required")
    task = task_manager.add_task(text)
    data = {
        "task": {"id": task.id, "text": task.text, "done": task.done},
        "tasks": task_manager.to_payload(),
    }
    return _success(data)
