from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from lmao.debug_log import DebugLogger
from lmao.plugins import PLUGIN_API_VERSION

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


def _iter_skill_dirs(skill_roots: Sequence[Path]) -> Iterable[Path]:
    for root in skill_roots:
        if not root.exists() or not root.is_dir():
            continue
        for candidate in root.iterdir():
            if candidate.is_dir() and (candidate / "SKILL.md").exists():
                yield candidate


def _list_skill_info(skill_roots: Sequence[Path]) -> list[Tuple[str, Path]]:
    seen = set()
    info: list[Tuple[str, Path]] = []
    for path in _iter_skill_dirs(skill_roots):
        key = (path.name, str(path))
        if key in seen:
            continue
        seen.add(key)
        info.append((path.name, path))
    info.sort(key=lambda pair: pair[0])
    return info


def run(
    target: str,
    args: str,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    task_manager=None,
    debug_logger: Optional[DebugLogger] = None,
) -> str:
    skills = _list_skill_info(skill_roots)
    data = [{"name": name, "path": str(path)} for name, path in skills]
    return _success(data)
