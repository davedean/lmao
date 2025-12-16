from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.plugin_helpers import normalize_path_for_output, safe_target_path

PLUGIN = {
    "name": "find",
    "description": "Walk a directory tree (skips dotfiles) with truncation at 200 entries.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "v2 args: {max_entries:int, include_dotfiles:bool}; v1 args ignored",
    "usage": [
        "{\"tool\":\"find\",\"target\":\".\",\"args\":\"\"}",
        "{\"tool\":\"find\",\"target\":\".\",\"args\":{\"max_entries\":200}}",
    ],
}


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
        target_path = safe_target_path(target or ".", base, extra_roots)
    except Exception:
        return _error(f"target path '{target}' escapes working directory")

    if not target_path.exists():
        return _error(f"path '{target}' not found")
    max_entries = 200
    include_dotfiles = False
    if isinstance(args, dict):
        try:
            max_entries = int(args.get("max_entries", max_entries))
        except Exception:
            max_entries = 200
        include_dotfiles = bool(args.get("include_dotfiles", False))

    results = []
    truncated = False
    try:
        for root, dirs, files in os.walk(target_path):
            if not include_dotfiles:
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                visible_files = [f for f in files if not f.startswith(".")]
            else:
                visible_files = list(files)
            for name in sorted(visible_files + dirs):
                path = Path(root) / name
                rel_display = normalize_path_for_output(path, base)
                results.append({
                    "name": name,
                    "path": rel_display,
                    "type": "dir" if path.is_dir() else "file",
                })
                if len(results) >= max_entries:
                    truncated = True
                    break
            if truncated:
                break
        data = {"path": normalize_path_for_output(target_path, base), "results": results}
        if truncated:
            data["truncated"] = True
        return _success(data)
    except Exception as exc:
        return _error(f"unable to walk '{target}': {exc}")
