from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

from .debug_log import DebugLogger
from .llm import LLMClient, ProviderName, estimate_message_tokens
from .protocol import (
    EndStep,
    MessageStep,
    ProtocolError,
    Step,
    ThinkStep,
    ToolCallStep,
    parse_assistant_turn,
)

TOOL_RESULT_PROMPT_PREFIX = "Tool result for "
TRUNCATION_MARKER = "\n...[truncated: output exceeded prompt limit; narrow the request for more detail]"
MAX_TOOL_RESULT_PROMPT_CHARS = 20_000
PROMPT_TRIGGER_RATIO = 0.7
PROMPT_TARGET_RATIO = 0.6
MIN_PROMPT_TOKENS = 128
DEFAULT_RESERVED_COMPLETION_TOKENS = 4096
DEFAULT_CONTEXT_WINDOW_TOKENS: dict[ProviderName, int] = {
    "lmstudio": 8192,
    "openrouter": 32_768,
}


@dataclass
class MemoryState:
    pinned_message_ids: Set[int] = field(default_factory=set)
    last_user_message: Optional[Dict[str, str]] = None


def sanitize_assistant_reply(reply: str, allowed_tools: Sequence[str]) -> str:
    """Remove think steps when storing assistant history and keep the rest intact."""
    try:
        turn = parse_assistant_turn(reply, allowed_tools=allowed_tools)
    except ProtocolError:
        return reply
    cleaned_steps: List[Dict[str, Any]] = []
    for step in turn.steps:
        if isinstance(step, ThinkStep):
            continue
        cleaned_steps.append(_step_to_dict(step))
    sanitized: Dict[str, Any] = {
        "type": turn.type,
        "version": turn.version,
        "steps": cleaned_steps,
    }
    if getattr(turn, "turn", None) is not None:
        sanitized["turn"] = turn.turn
    return json.dumps(sanitized, ensure_ascii=False)


def _step_to_dict(step: Step) -> Dict[str, Any]:
    if isinstance(step, MessageStep):
        return {
            "type": "message",
            "content": step.content,
            "format": step.format,
            "purpose": step.purpose,
        }
    if isinstance(step, ToolCallStep):
        call = {
            "tool": step.call.tool,
            "target": step.call.target,
            "args": step.call.args,
        }
        if step.call.meta is not None:
            call["meta"] = step.call.meta
        return {"type": "tool_call", "call": call}
    if isinstance(step, EndStep):
        return {"type": "end", "reason": step.reason}
    if isinstance(step, ThinkStep):
        return {"type": "think", "content": step.content}
    return {"type": step.type}


def should_pin_agents_tool_result(tool: str, target: str) -> bool:
    """AGENTS instructions should remain visible to the model."""
    if tool == "policy":
        return True
    if tool == "read":
        try:
            candidate = Path(str(target).strip())
            return candidate.name.lower() == "agents.md"
        except Exception:
            return False
    return False


def truncate_tool_result_for_prompt(
    output: str,
    *,
    max_chars: int = MAX_TOOL_RESULT_PROMPT_CHARS,
    is_pinned: bool = False,
) -> Tuple[str, bool]:
    """Limit tool output stored in the prompt; pinned AGENTS outputs bypass this."""
    if is_pinned or len(output) <= max_chars:
        return output, False
    marker = TRUNCATION_MARKER
    available = max_chars - len(marker)
    if available <= 0:
        return marker, True
    truncated = output[:available] + marker
    return truncated, True


def determine_prompt_budget(client: LLMClient) -> Tuple[int, int, int]:
    """Compute max prompt tokens, and the trigger/target thresholds."""
    provider_value = getattr(client, "provider", "lmstudio")
    if provider_value in DEFAULT_CONTEXT_WINDOW_TOKENS:
        provider = cast(ProviderName, provider_value)
    else:
        provider = "lmstudio"
    override = getattr(client, "context_window_tokens", None)
    context_window = _context_window_for_provider(provider, override=override)
    reserved = _reserved_completion_tokens(client, context_window)
    max_prompt = context_window - reserved
    min_prompt = max(int(context_window * 0.25), MIN_PROMPT_TOKENS)
    if max_prompt < min_prompt:
        max_prompt = min_prompt
    trigger = max(int(max_prompt * PROMPT_TRIGGER_RATIO), 1)
    target = max(int(max_prompt * PROMPT_TARGET_RATIO), 1)
    return max_prompt, trigger, target


def _context_window_for_provider(provider: ProviderName, *, override: Optional[int] = None) -> int:
    if isinstance(override, int) and override > 0:
        return override
    return DEFAULT_CONTEXT_WINDOW_TOKENS.get(provider, DEFAULT_CONTEXT_WINDOW_TOKENS["lmstudio"])


def _reserved_completion_tokens(client: LLMClient, context_window: int) -> int:
    max_tokens = getattr(client, "max_tokens", None)
    reserved = max_tokens if max_tokens is not None else DEFAULT_RESERVED_COMPLETION_TOKENS
    if context_window <= 1:
        return 1
    if reserved >= context_window:
        reserved = max(1, context_window // 2)
    if reserved <= 0:
        return max(1, context_window // 4)
    return reserved


def compact_messages_if_needed(
    messages: List[Dict[str, str]],
    *,
    last_user_message: Optional[Dict[str, str]],
    pinned_message_ids: Set[int],
    trigger_tokens: int,
    target_tokens: int,
    debug_logger: Optional[DebugLogger],
) -> None:
    """Drop older history until the estimated prompt tokens stay under budget."""
    if trigger_tokens <= 0 or target_tokens <= 0:
        return
    start = time.perf_counter()
    current_tokens = estimate_message_tokens(messages)
    if current_tokens <= trigger_tokens:
        if debug_logger:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            debug_logger.log(
                "memory.compact.timing",
                f"elapsed_ms={elapsed_ms:.2f} dropped=0 prompt_tokens={current_tokens} target={target_tokens}",
            )
        return
    dropped = 0
    while current_tokens > target_tokens:
        idx = _next_droppable_index(messages, pinned_message_ids, last_user_message)
        if idx is None:
            break
        removed = messages.pop(idx)
        dropped += 1
        if debug_logger:
            preview = _preview_text_for_log(removed.get("content"))
            debug_logger.log("memory.compact.drop", f"role={removed.get('role')} idx={idx} preview={preview}")
        current_tokens = estimate_message_tokens(messages)
    if debug_logger:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if dropped:
            debug_logger.log(
                "memory.compact",
                f"dropped={dropped} prompt_tokens={current_tokens} target={target_tokens}",
            )
        debug_logger.log(
            "memory.compact.timing",
            f"elapsed_ms={elapsed_ms:.2f} dropped={dropped} prompt_tokens={current_tokens} target={target_tokens}",
        )


def _next_droppable_index(
    messages: List[Dict[str, str]],
    pinned_message_ids: Set[int],
    last_user_message: Optional[Dict[str, str]],
) -> Optional[int]:
    if len(messages) <= 1:
        return None
    last_tool_result_idx: Optional[int] = None
    for idx, message in enumerate(messages):
        if idx == 0:
            continue
        if id(message) in pinned_message_ids:
            continue
        if last_user_message is not None and message is last_user_message:
            continue
        if _is_tool_result_message(message):
            last_tool_result_idx = idx
    for idx, message in enumerate(messages):
        if idx == 0:
            continue
        if id(message) in pinned_message_ids:
            continue
        if last_user_message is not None and message is last_user_message:
            continue
        if _is_tool_result_message(message):
            # Keep the most recent (droppable) tool result message so the model can
            # react to it; otherwise it may re-run the same tool in a loop.
            if last_tool_result_idx is not None and idx == last_tool_result_idx:
                continue
            return idx
    for idx, message in enumerate(messages):
        if idx == 0:
            continue
        if id(message) in pinned_message_ids:
            continue
        if last_user_message is not None and message is last_user_message:
            continue
        if last_tool_result_idx is not None and idx == last_tool_result_idx:
            continue
        return idx
    return None


def _is_tool_result_message(message: Dict[str, str]) -> bool:
    return bool(
        message.get("role") == "user"
        and isinstance(content := message.get("content"), str)
        and content.startswith(TOOL_RESULT_PROMPT_PREFIX)
    )


def _preview_text_for_log(value: Optional[str]) -> str:
    if not value:
        return ""
    text = value.replace("\n", " ").strip()
    return text[:120]


def aggressive_compact_messages(
    messages: List[Dict[str, str]],
    *,
    last_user_message: Optional[Dict[str, str]],
    pinned_message_ids: Set[int],
    debug_logger: Optional[DebugLogger],
) -> None:
    if not messages:
        return
    preserved: List[Dict[str, str]] = [messages[0]]
    seen: Set[int] = {id(messages[0])}
    for message in messages:
        if id(message) in pinned_message_ids and id(message) not in seen:
            preserved.append(message)
            seen.add(id(message))
    if last_user_message and id(last_user_message) not in seen:
        preserved.append(last_user_message)
        seen.add(id(last_user_message))
    messages[:] = preserved
    if debug_logger:
        debug_logger.log("memory.compact.aggressive", f"retained={len(preserved)} messages")


def is_context_length_error(message: str) -> bool:
    text = (message or "").lower()
    patterns = (
        "context length",
        "context-length",
        "context window",
        "context-window",
        "maximum context",
        "requested token count exceeds",
        "input is too long",
    )
    return any(pattern in text for pattern in patterns)
