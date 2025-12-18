from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.context import find_nearest_agents, find_repo_root

PLUGIN = {
    "name": "policy",
    "description": "Read the nearest repo policy file (AGENTS.md) with pagination.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "v2 args optional: {offset:int, limit:int, truncate:bool}; returns an excerpt by default (2000 chars)",
    "usage": [
        "{\"tool\":\"policy\",\"target\":\"\",\"args\":\"\"}",
        "{\"tool\":\"policy\",\"target\":\"\",\"args\":{}}",
        "{\"tool\":\"policy\",\"target\":\"\",\"args\":{\"offset\":0,\"limit\":2000}}",
        "{\"tool\":\"policy\",\"target\":\"\",\"args\":{\"truncate\":false}}",
    ],
}

DEFAULT_EXCERPT_LIMIT = 2000


def _success(data: dict) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": False, "error": message}, ensure_ascii=False)


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
    try:
        offset = 0
        limit = DEFAULT_EXCERPT_LIMIT
        truncate = True
        if isinstance(args, dict):
            raw_offset = args.get("offset")
            raw_limit = args.get("limit")
            raw_truncate = args.get("truncate")
            if raw_offset is not None:
                offset = int(raw_offset)
            if raw_limit is not None:
                limit = int(raw_limit)
            if raw_truncate is not None:
                truncate = bool(raw_truncate)
        if offset < 0:
            offset = 0
        if limit <= 0:
            limit = DEFAULT_EXCERPT_LIMIT

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
        total_chars = len(content)
        if not truncate:
            offset = 0
            limit = total_chars
        excerpt = content[offset : offset + limit]
        next_offset = offset + len(excerpt)
        has_more = next_offset < total_chars
        data = {
            "path": str(nearest),
            "offset": offset,
            "limit": limit,
            "total_chars": total_chars,
            "has_more": has_more,
            "next_offset": next_offset if has_more else None,
            "content": excerpt,
            "content_truncated": has_more,
        }
        return _success(data)
    except Exception as exc:
        return _error(f"unable to read AGENTS.md: {exc}")
