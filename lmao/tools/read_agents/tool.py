from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.context import find_nearest_agents, find_repo_root

PLUGIN = {
    "name": "read_agents",
    "description": "Read the nearest repo AGENTS.md (walking up to the repo root).",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "none",
    "usage": "{'tool':'read_agents','target':'','args':''}",
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
    try:
        repo_root = find_repo_root(base)
        nearest = find_nearest_agents(base, repo_root)
        if not nearest or not nearest.exists():
            return _error("no AGENTS.md found between workdir and repo root")
        # Safety: ensure we only read within the discovered repo root.
        try:
            nearest.resolve().relative_to(repo_root.resolve())
        except Exception:
            return _error("refusing to read AGENTS.md outside repo root")
        content = nearest.read_text(encoding="utf-8")
        data = {"path": str(nearest), "content": content}
        return _success(data)
    except Exception as exc:
        return _error(f"unable to read AGENTS.md: {exc}")

