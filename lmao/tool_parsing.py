from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from .jsonish import extract_braced_objects as _extract_braced_objects
from .jsonish import iter_jsonish_candidates, try_load_jsonish

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
    return list(iter_jsonish_candidates(raw_text))


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
    # Back-compat: exported via `lmao.tools` and used by tests.
    return _extract_braced_objects(raw_text)


def load_candidate(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    loaded = try_load_jsonish(text, recover_extra_data=False)
    if loaded is None:
        return None
    if isinstance(loaded, (dict, list)):
        return loaded
    return None
