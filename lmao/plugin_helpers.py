from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from .context import find_repo_root as _find_repo_root
from .skills import validate_skill_write_target as _validate_skill_write_target
from .tools import (
    normalize_path_for_output as _normalize_path_for_output,
    normalize_task_text as _normalize_task_text,
    parse_line_range as _parse_line_range,
    safe_target_path as _safe_target_path,
)


def safe_target_path(target: str, base: Path, extra_roots: Sequence[Path]) -> Path:
    """Path-safe resolver for plugins; reuses core sandboxing rules."""
    return _safe_target_path(target, base, extra_roots)


def normalize_path_for_output(path: Path, base: Path) -> str:
    """Consistent display path helper for plugin outputs."""
    return _normalize_path_for_output(path, base)


def parse_line_range(arg: str) -> Optional[Tuple[int, int]]:
    """Parse strings like 'lines:10-20' into (start, end) tuples."""
    return _parse_line_range(arg)


def validate_skill_write_target(target_path: Path, skill_roots: Sequence[Path]) -> Optional[str]:
    """Shared guard to keep skill files inside their own folders."""
    return _validate_skill_write_target(target_path, skill_roots)


def find_repo_root(start: Path) -> Path:
    """Find the nearest git repo root from a starting directory."""
    return _find_repo_root(start)


def normalize_task_text(text: str) -> str:
    """Normalize task text for task list plugins."""
    return _normalize_task_text(text)
