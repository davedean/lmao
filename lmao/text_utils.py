from __future__ import annotations

import re


def summarize_output(output: str, max_lines: int = 8, max_chars: int = 400) -> str:
    if max_lines <= 0 and max_chars <= 0:
        return ""
    lines = output.splitlines()
    head = lines[:max_lines] if max_lines > 0 else lines
    summary = "\n".join(head)
    if max_lines > 0 and len(lines) > max_lines:
        summary += "\n...[truncated]"
    if max_chars > 0 and len(summary) > max_chars:
        summary = summary[:max_chars] + "\n...[truncated]"
    return summary


def normalize_task_text(text: str) -> str:
    """Normalize task text: remove numbering, collapse whitespace, avoid newlines."""
    cleaned = text.replace("\n", " ").replace("\r", " ")
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

