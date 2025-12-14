from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROTOCOL_VERSION = "1"


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ToolCallPayload:
    tool: str
    target: str
    args: str


@dataclass(frozen=True)
class Step:
    type: str


@dataclass(frozen=True)
class ThinkStep(Step):
    content: str


@dataclass(frozen=True)
class MessageStep(Step):
    content: str
    format: str = "markdown"


@dataclass(frozen=True)
class ToolCallStep(Step):
    call: ToolCallPayload


@dataclass(frozen=True)
class EndStep(Step):
    reason: str = "completed"


@dataclass(frozen=True)
class AssistantTurn:
    type: str
    version: str
    steps: List[Step]
    turn: Optional[int] = None


def _require_dict(obj: Any, where: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ProtocolError(f"{where} must be an object")
    return obj


def _require_list(obj: Any, where: str) -> List[Any]:
    if not isinstance(obj, list):
        raise ProtocolError(f"{where} must be a list")
    return obj


def _require_str(obj: Any, where: str) -> str:
    if not isinstance(obj, str) or not obj.strip():
        raise ProtocolError(f"{where} must be a non-empty string")
    return obj


def _coerce_args(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def parse_assistant_turn(raw_text: str, allowed_tools: Sequence[str]) -> AssistantTurn:
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc

    obj = _require_dict(parsed, "assistant_turn")
    if obj.get("type") != "assistant_turn":
        raise ProtocolError("assistant_turn.type must be 'assistant_turn'")
    version = str(obj.get("version", "")).strip()
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"assistant_turn.version must be '{PROTOCOL_VERSION}'")

    raw_steps = _require_list(obj.get("steps"), "assistant_turn.steps")
    steps: List[Step] = []
    seen_tool_call = False
    for idx, raw_step in enumerate(raw_steps):
        step_obj = _require_dict(raw_step, f"steps[{idx}]")
        step_type = _require_str(step_obj.get("type", ""), f"steps[{idx}].type")

        if step_type == "think":
            content = _require_str(step_obj.get("content", ""), f"steps[{idx}].content")
            steps.append(ThinkStep(type="think", content=content))
            continue

        if step_type == "message":
            content = _require_str(step_obj.get("content", ""), f"steps[{idx}].content")
            fmt = str(step_obj.get("format", "markdown") or "markdown")
            steps.append(MessageStep(type="message", content=content, format=fmt))
            continue

        if step_type == "tool_call":
            if seen_tool_call:
                raise ProtocolError("only one tool_call step is allowed per reply")
            call_obj = _require_dict(step_obj.get("call"), f"steps[{idx}].call")
            tool = _require_str(call_obj.get("tool", ""), f"steps[{idx}].call.tool")
            if tool not in set(allowed_tools):
                raise ProtocolError(f"steps[{idx}].call.tool '{tool}' is not allowed")
            target = str(call_obj.get("target", "") or "")
            args = _coerce_args(call_obj.get("args", ""))
            steps.append(ToolCallStep(type="tool_call", call=ToolCallPayload(tool=tool, target=target, args=args)))
            seen_tool_call = True
            continue

        if step_type == "end":
            reason = str(step_obj.get("reason", "") or "completed").strip() or "completed"
            steps.append(EndStep(type="end", reason=reason))
            continue

        raise ProtocolError(f"unsupported step type '{step_type}' at steps[{idx}]")

    turn_val = obj.get("turn")
    turn: Optional[int] = None
    if turn_val is not None:
        try:
            turn = int(turn_val)
        except Exception:
            raise ProtocolError("assistant_turn.turn must be an integer when provided")

    return AssistantTurn(type="assistant_turn", version=version, steps=steps, turn=turn)


def find_first_tool_call(steps: Sequence[Step]) -> Optional[ToolCallPayload]:
    for step in steps:
        if isinstance(step, ToolCallStep):
            return step.call
    return None


def collect_steps(steps: Sequence[Step]) -> Tuple[List[ThinkStep], List[MessageStep], bool]:
    thinks: List[ThinkStep] = []
    messages: List[MessageStep] = []
    has_end = False
    for step in steps:
        if isinstance(step, ThinkStep):
            thinks.append(step)
        elif isinstance(step, MessageStep):
            messages.append(step)
        elif isinstance(step, EndStep):
            has_end = True
    return thinks, messages, has_end
