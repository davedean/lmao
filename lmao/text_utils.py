from __future__ import annotations

import re


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

