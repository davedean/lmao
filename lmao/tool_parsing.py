from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class ToolCall:
    tool: str
    target: str
    args: Any
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_raw_message(
        cls, raw_text: str, allowed_tools: Optional[Sequence[str]] = None
    ) -> Optional["ToolCall"]:
        calls = parse_tool_calls(raw_text, allowed_tools=allowed_tools)
        return calls[0] if calls else None


def iter_json_candidates(raw_text: str) -> List[str]:
    candidates: List[str] = []
    stripped = raw_text.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL)
    candidates.extend(fenced)
    candidates.extend(extract_braced_objects(raw_text))
    if stripped:
        candidates.append(stripped)
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        unique.append(cand)
    return unique


def parse_tool_calls(raw_text: str, allowed_tools: Optional[Sequence[str]] = None) -> List[ToolCall]:
    calls: List[ToolCall] = []
    # Safety: if an allowlist isn't provided, accept no tool calls.
    allowed = set(allowed_tools) if allowed_tools is not None else set()
    for candidate in iter_json_candidates(raw_text):
        parsed = load_candidate(candidate)
        if not parsed:
            continue
        if isinstance(parsed, list):
            parsed_items = parsed
        else:
            parsed_items = [parsed]
        for obj in parsed_items:
            if not isinstance(obj, dict):
                continue
            tool = str(obj.get("tool", "")).strip()
            if tool not in allowed:
                continue
            target = str(obj.get("target", "") or "").strip()
            args = obj.get("args", "")
            meta = obj.get("meta")
            meta_dict = meta if isinstance(meta, dict) else None
            calls.append(ToolCall(tool=tool, target=target, args=args, meta=meta_dict))
    return calls


def covers_message(candidate: str, raw_text: str) -> bool:
    stripped = raw_text.strip()
    cand = candidate.strip()
    if not cand:
        return False
    if cand == stripped:
        return True
    fenced_json = f"```json\n{cand}\n```"
    fenced_plain = f"```\n{cand}\n```"
    return stripped == fenced_json or stripped == fenced_plain


def extract_braced_objects(raw_text: str) -> List[str]:
    """Extract top-level brace-delimited JSON-ish substrings."""
    objs: List[str] = []
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


def load_candidate(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    cleaned = text.strip()
    if not cleaned:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(cleaned)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            continue
    return None
