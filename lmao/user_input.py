from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional


InputFn = Callable[[str], str]
DrainFn = Callable[[], List[str]]


@dataclass(frozen=True)
class PromptReadResult:
    text: Optional[str]
    eof: bool = False


def _default_drain_available_lines() -> List[str]:
    """
    Best-effort "paste join": if multiple lines are already buffered on stdin (e.g. the user pasted
    a multi-line block), drain them immediately and treat them as part of the same prompt.

    This is intentionally non-blocking. If the terminal/OS doesn't support non-blocking readiness
    checks, it simply returns [].
    """
    try:
        import select
        import sys

        if not hasattr(sys.stdin, "fileno"):
            return []
        lines: List[str] = []
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if not r:
                break
            chunk = sys.stdin.readline()
            if chunk == "":
                break
            lines.append(chunk.rstrip("\n"))
        return lines
    except Exception:
        return []


def read_user_prompt(
    prompt: str,
    *,
    input_fn: InputFn = input,
    multiline_default: bool = False,
    sentinel: str = '"""',
    join_pasted: bool = True,
    drain_available_lines_fn: Optional[DrainFn] = None,
) -> PromptReadResult:
    """
    Read a user prompt from stdin.

    - Default: single line (input()).
    - Sentinel mode: if first line is exactly `sentinel`, read lines until sentinel or EOF.
    - Multiline default: read lines until EOF; if EOF occurs before any lines, returns eof=True.

    Returns PromptReadResult(text=..., eof=...) where text is None only when eof=True with no content.
    """
    if multiline_default:
        lines: List[str] = []
        while True:
            try:
                line = input_fn(prompt if not lines else "")
            except EOFError:
                if not lines:
                    return PromptReadResult(text=None, eof=True)
                return PromptReadResult(text="\n".join(lines), eof=False)
            lines.append(line)

    try:
        first = input_fn(prompt)
    except EOFError:
        return PromptReadResult(text=None, eof=True)

    if first.strip() == sentinel:
        lines = []
        while True:
            try:
                line = input_fn("")
            except EOFError:
                break
            if line.strip() == sentinel:
                break
            lines.append(line)
        return PromptReadResult(text="\n".join(lines), eof=False)

    if join_pasted:
        drain_fn = drain_available_lines_fn or _default_drain_available_lines
        extra = drain_fn()
        if extra:
            return PromptReadResult(text="\n".join([first] + extra), eof=False)

    return PromptReadResult(text=first, eof=False)
