from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


SKILL_WRITE_ERROR = (
    "error: place skill files under skills/<skill-name>/, e.g., skills/demo/; "
    "SKILL.md must live inside that folder. Supporting files are allowed inside skill folders."
)


def is_under_skill_root(path: Path, skill_roots: Sequence[Path]) -> bool:
    """Return True when the path lives under one of the configured skill roots."""
    for root in skill_roots:
        try:
            path.relative_to(root)
            return True
        except Exception:
            continue
    return False


def validate_skill_write_target(target_path: Path, skill_roots: Sequence[Path]) -> Optional[str]:
    """
    Block writes that drop files directly under a skill root (e.g., skills/foo or skills/foo.md).
    Skill files should live inside a named skill folder, e.g., skills/demo/SKILL.md.
    """
    for root in skill_roots:
        try:
            target_path.relative_to(root)
        except Exception:
            continue
        if target_path.parent == root:
            return SKILL_WRITE_ERROR
    return None


def iter_skill_dirs(skill_roots: Sequence[Path]) -> Iterable[Path]:
    """Yield skill directories (those containing a SKILL.md) under the given roots."""
    for root in skill_roots:
        if not root.exists() or not root.is_dir():
            continue
        for candidate in root.iterdir():
            if candidate.is_dir() and (candidate / "SKILL.md").exists():
                yield candidate


def list_skill_paths(skill_roots: Sequence[Path]) -> List[str]:
    """Return unique, sorted skill directory paths across all configured roots."""
    seen = set()
    paths: List[str] = []
    for path in iter_skill_dirs(skill_roots):
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        paths.append(path_str)
    return sorted(paths)


def list_skill_info(skill_roots: Sequence[Path]) -> List[Tuple[str, Path]]:
    """
    Return unique (name, path) pairs for skills, using the directory name as the skill name.
    """
    seen = set()
    info: List[Tuple[str, Path]] = []
    for path in iter_skill_dirs(skill_roots):
        name = path.name
        key = (name, str(path))
        if key in seen:
            continue
        seen.add(key)
        info.append((name, path))
    info.sort(key=lambda pair: pair[0])
    return info
