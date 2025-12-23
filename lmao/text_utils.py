from __future__ import annotations

import json
import re
from typing import Any, Iterable, List, Optional


def truncate_text(text: str, *, max_lines: int, max_chars: int, suffix: str = "\n...[truncated]") -> str:
    if max_lines <= 0 and max_chars <= 0:
        return ""
    if not text:
        return ""
    lines = text.splitlines()
    head = lines[:max_lines] if max_lines > 0 else lines
    summary = "\n".join(head)
    if max_lines > 0 and len(lines) > max_lines:
        summary += suffix
    if max_chars > 0 and len(summary) > max_chars:
        summary = summary[:max_chars] + suffix
    return summary


def summarize_output(output: str, max_lines: int = 8, max_chars: int = 400) -> str:
    return truncate_text(output, max_lines=max_lines, max_chars=max_chars)


_TOOL_SUMMARY_SKIP_KEYS = {
    "content",
    "text",
    "stdout",
    "stderr",
    "patch",
    "diff",
    "output",
    "raw",
}
_TOOL_SUMMARY_MAX_FIELDS = 6
_TOOL_SUMMARY_STR_LIMIT = 80


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _summarize_tool_value(key: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    if key == "lines" and isinstance(value, dict):
        start = value.get("start")
        end = value.get("end")
        if isinstance(start, int) and isinstance(end, int):
            return f"lines={start}-{end}"
    if isinstance(value, bool):
        return f"{key}={'true' if value else 'false'}"
    if isinstance(value, (int, float)):
        return f"{key}={value}"
    if isinstance(value, str):
        if len(value) > _TOOL_SUMMARY_STR_LIMIT:
            return f"{key}_chars={len(value)}"
        return f"{key}={_single_line(value)}"
    if isinstance(value, list):
        return f"{key}_items={len(value)}"
    if isinstance(value, dict):
        return f"{key}_keys={len(value)}"
    return f"{key}={_single_line(str(value))}"


def _summarize_mapping_items(items: Iterable[tuple[str, Any]]) -> List[str]:
    parts: List[str] = []
    for key, value in items:
        if len(parts) >= _TOOL_SUMMARY_MAX_FIELDS:
            break
        if key in _TOOL_SUMMARY_SKIP_KEYS:
            if isinstance(value, str):
                parts.append(f"{key}_chars={len(value)}")
            elif isinstance(value, list):
                parts.append(f"{key}_items={len(value)}")
            elif isinstance(value, dict):
                parts.append(f"{key}_keys={len(value)}")
            continue
        part = _summarize_tool_value(key, value)
        if part:
            parts.append(part)
    return parts


def summarize_tool_output(output: str, max_lines: int = 1, max_chars: int = 400) -> str:
    if not output:
        return "no tool output"
    try:
        payload = json.loads(output)
    except Exception:
        return "unparseable tool output"
    if not isinstance(payload, dict):
        return "unparseable tool output"

    success = payload.get("success")
    if success is True:
        data = payload.get("data")
        parts: List[str] = []
        if isinstance(data, dict):
            parts = _summarize_mapping_items(data.items())
        elif data is not None:
            part = _summarize_tool_value("data", data)
            if part:
                parts.append(part)
        summary = "ok"
        if parts:
            summary = f"{summary} " + " ".join(parts)
        summary = _single_line(summary)
        return truncate_text(summary, max_lines=max_lines, max_chars=max_chars, suffix="...[truncated]")

    error = payload.get("error")
    if error is None and isinstance(payload.get("data"), str):
        error = payload.get("data")
    if error is None:
        summary = "error"
    else:
        summary = f"error {_single_line(str(error))}"
    summary = _single_line(summary)
    return truncate_text(summary, max_lines=max_lines, max_chars=max_chars, suffix="...[truncated]")


def summarize_tool_args(args: Any, max_chars: int = 120) -> str:
    if args in ("", None, {}):
        return ""
    if isinstance(args, dict):
        parts = _summarize_mapping_items(args.items())
        text = " ".join(parts) if parts else ""
    else:
        text = _single_line(str(args))
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text
