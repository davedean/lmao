from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterator, Optional, Union

_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)

Jsonish = Union[dict, list]


def extract_fenced_blocks(raw_text: str) -> list[str]:
    blocks: list[str] = []
    for match in _FENCED_BLOCK_RE.findall(raw_text):
        cleaned = match.strip()
        if cleaned:
            blocks.append(cleaned)
    return blocks


def extract_braced_objects(raw_text: str) -> list[str]:
    """Extract top-level brace-delimited JSON-ish substrings."""
    objs: list[str] = []
    depth = 0
    start: Optional[int] = None
    for idx, ch in enumerate(raw_text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(raw_text[start : idx + 1].strip())
                    start = None
    return objs


def iter_jsonish_candidates(raw_text: str) -> Iterator[str]:
    stripped = raw_text.strip()
    candidates: list[str] = []
    candidates.extend(extract_fenced_blocks(raw_text))
    candidates.extend(extract_braced_objects(raw_text))
    if stripped:
        candidates.append(stripped)
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        yield cand


def _extract_json_prefix_on_extra_data(raw_text: str) -> Optional[dict]:
    """
    Best-effort recovery for model outputs that accidentally concatenate multiple JSON objects.
    If we can parse the first JSON object cleanly, return it; otherwise None.
    """
    try:
        json.loads(raw_text)
        return None
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            return None
        prefix = raw_text[: exc.pos].strip()
        if not prefix:
            return None
        try:
            obj = json.loads(prefix)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def try_load_jsonish(value: str, *, recover_extra_data: bool = False) -> Optional[Jsonish]:
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, (dict, list)) else None
    except json.JSONDecodeError:
        if recover_extra_data:
            recovered = _extract_json_prefix_on_extra_data(cleaned)
            if recovered is not None:
                return recovered
    except Exception:
        return None
    try:
        literal = ast.literal_eval(cleaned)
        return literal if isinstance(literal, (dict, list)) else None
    except Exception:
        return None


def load_first_jsonish(
    raw_text: str, *, dict_only: bool = False, recover_extra_data: bool = False
) -> Optional[Jsonish]:
    for candidate in iter_jsonish_candidates(raw_text):
        loaded = try_load_jsonish(candidate, recover_extra_data=recover_extra_data)
        if loaded is None:
            continue
        if dict_only and not isinstance(loaded, dict):
            continue
        return loaded
    return None

