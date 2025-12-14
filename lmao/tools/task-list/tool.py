from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from lmao.debug_log import DebugLogger
from lmao.plugins import PLUGIN_API_VERSION

PLUGIN = {
    "name": "list_tasks",
    "description": "Render the current task list.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "none",
    "usage": "{'tool':'list_tasks','target':'','args':''}",
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
    data = {"tasks": task_manager.to_payload(), "render": task_manager.render_tasks()}
    return _success(data)
