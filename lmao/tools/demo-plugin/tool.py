from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION

PLUGIN = {
    "name": "echo_plugin",
    "description": "Echoes the provided target/args payload for demo purposes.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "string payload",
    "usage": "{'tool':'echo_plugin','target':'hello','args':'world'}",
}


def run(
    target: str,
    args: str,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    task_manager=None,
    debug_logger: Optional[object] = None,
) -> str:
    """Echo plugin: returns the inputs to show plugin plumbing works."""
    payload = {"tool": PLUGIN["name"], "success": True, "data": {"target": target, "args": args}}
    return json.dumps(payload, ensure_ascii=False)
