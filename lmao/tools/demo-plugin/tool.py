from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

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
    "input_schema": "v2 args: any JSON (echoed back); v1 args: string",
    "usage": [
        "{\"tool\":\"echo_plugin\",\"target\":\"hello\",\"args\":\"world\"}",
        "{\"tool\":\"echo_plugin\",\"target\":\"hello\",\"args\":{\"x\":1}}",
    ],
}


def run(
    target: str,
    args: Any,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    task_manager=None,
    debug_logger: Optional[object] = None,
    meta: Optional[dict] = None,
) -> str:
    """Echo plugin: returns the inputs to show plugin plumbing works."""
    payload = {"tool": PLUGIN["name"], "success": True, "data": {"target": target, "args": args, "meta": meta}}
    return json.dumps(payload, ensure_ascii=False)
