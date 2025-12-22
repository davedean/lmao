from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.plugin_helpers import normalize_path_for_output, safe_target_path

PLUGIN = {
    "name": "grep",
    "description": "Search for a substring in files (skips dot-directories/files unless include_dotfiles=true, truncates by limit).",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "v2 args: {pattern:'...', path?:'.', include_dotfiles?:bool, max_matches?:int|limit?:int}; v1 args: pattern string",
    "usage": [
        "{\"tool\":\"grep\",\"target\":\"./path\",\"args\":\"substring\"}",
        "{\"tool\":\"grep\",\"target\":\"./path\",\"args\":{\"pattern\":\"substring\"}}",
        "{\"tool\":\"grep\",\"target\":\"\",\"args\":{\"path\":\"./path\",\"pattern\":\"substring\",\"include_dotfiles\":true,\"limit\":50}}",
    ],
}

MAX_FILE_BYTES = 2_000_000


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
    debug_logger: Optional[object] = None,
    meta: Optional[dict] = None,
) -> str:
    if isinstance(args, dict):
        if not target:
            target = str(args.get("path") or args.get("target") or "")
        pattern = str(args.get("pattern") or args.get("query") or "")
        include_dotfiles = bool(args.get("include_dotfiles", False))
        max_matches = args.get("max_matches")
        if max_matches is None:
            max_matches = args.get("limit")
        if max_matches is None:
            max_matches = args.get("max_results")
    else:
        include_dotfiles = False
        max_matches = None
        pattern = str(args)
    if not pattern:
        return _error("pattern is required")
    try:
        target_path = safe_target_path(target or ".", base, extra_roots)
    except Exception:
        return _error(f"target path '{target}' escapes working directory")

    if not target_path.exists():
        return _error(f"path '{target}' not found")

    try:
        max_matches_value = int(max_matches) if max_matches is not None else 200
    except Exception:
        max_matches_value = 200
    if max_matches_value <= 0:
        max_matches_value = 200

    candidates: list[Path] = []
    if target_path.is_dir():
        for root, dirs, files in os.walk(target_path):
            if not include_dotfiles:
                dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                if not include_dotfiles and name.startswith("."):
                    continue
                candidates.append(Path(root) / name)
    else:
        if include_dotfiles or not target_path.name.startswith("."):
            candidates = [target_path]

    matches = []
    truncated = False
    for candidate in candidates:
        try:
            if candidate.stat().st_size > MAX_FILE_BYTES:
                continue
        except Exception:
            continue
        try:
            content = candidate.read_text(encoding="utf-8")
        except Exception:
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            if pattern in line:
                matches.append({
                    "path": normalize_path_for_output(candidate, base),
                    "line": lineno,
                    "text": line,
                })
            if len(matches) >= max_matches_value:
                truncated = True
                break
        if truncated:
            break

    data = {
        "path": normalize_path_for_output(target_path, base),
        "pattern": pattern,
        "matches": matches,
        "limit": max_matches_value,
        "limit_chars": None,
        "include_dotfiles": include_dotfiles,
    }
    if truncated:
        data["truncated"] = True
    return _success(data)
