from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from lmao.debug_log import DebugLogger
from lmao.plugins import PLUGIN_API_VERSION
from lmao.skills import list_skill_info

PLUGIN = {
    "name": "list_skills",
    "description": "List available skills (repo + user).",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "none",
}


def _success(data: dict) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False)


def run(
    target: str,
    args: str,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    task_manager=None,
    debug_logger: Optional[DebugLogger] = None,
) -> str:
    skills = list_skill_info(skill_roots)
    data = [{"name": name, "path": str(path)} for name, path in skills]
    return _success(data)
