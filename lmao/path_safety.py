from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Tuple


def normalize_path_for_output(path: Path, base: Path) -> str:
    try:
        rel = path.relative_to(base)
    except Exception:
        rel = path
    rel_str = str(rel)
    if path.is_dir() and not rel_str.endswith("/"):
        rel_str += "/"
    return rel_str


def safe_target_path(target: str, base: Path, extra_roots: Sequence[Path]) -> Path:
    target_str = str(target or "").strip()
    # Model ergonomics: treat "/" as "repo root" (the working directory base), not filesystem root.
    if target_str in ("/", "\\"):
        raw_path = Path(".")
    # Model ergonomics: treat "/foo/bar" as repo-root-relative (base/foo/bar), not filesystem absolute.
    elif target_str.startswith(("/", "\\")):
        raw_path = Path(target_str.lstrip("/\\"))
    else:
        raw_path = Path(target).expanduser()
    target_path = raw_path.resolve() if raw_path.is_absolute() else (base / raw_path).resolve()

    allowed_roots = [base] + [p.resolve() for p in extra_roots]
    for root in allowed_roots:
        try:
            target_path.relative_to(root)
            return target_path
        except Exception:
            continue
    raise ValueError("path escapes allowed roots")


def parse_line_range(arg: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"lines?[:=]\s*(\d+)(?:[-:](\d+))?", arg)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    if start < 1:
        start = 1
    if end < start:
        end = start
    return start, end
