from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROTOCOL_VERSION = "1"
TASK_TOOL_NAMES = {"add_task", "complete_task", "delete_task", "list_tasks"}


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
    purpose: str = "progress"


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


def _parse_task_tool_alias_step(step_obj: Dict[str, Any], allowed_tools: Sequence[str], idx: int) -> ToolCallStep:
    tool = _require_str(step_obj.get("type", ""), f"steps[{idx}].type")
    if tool not in set(allowed_tools):
        raise ProtocolError(f"steps[{idx}].type '{tool}' is not allowed")

    payload = step_obj.get("args", "")
    target = str(step_obj.get("target", "") or "")

    if tool == "add_task":
        if isinstance(payload, dict):
            text = payload.get("task") or payload.get("text") or payload.get("args") or ""
            args = _coerce_args(text)
        else:
            args = _coerce_args(payload)
        return ToolCallStep(type="tool_call", call=ToolCallPayload(tool=tool, target=target, args=args))

    if tool in ("complete_task", "delete_task"):
        if isinstance(payload, dict):
            task_id = payload.get("id") or payload.get("task_id") or payload.get("args") or payload.get("target") or ""
            args = _coerce_args(task_id)
        else:
            args = _coerce_args(payload)
        return ToolCallStep(type="tool_call", call=ToolCallPayload(tool=tool, target=target, args=args))

    # list_tasks
    args = _coerce_args(payload)
    return ToolCallStep(type="tool_call", call=ToolCallPayload(tool=tool, target=target, args=args))


def parse_assistant_turn(raw_text: str, allowed_tools: Sequence[str]) -> AssistantTurn:
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        recovered = _extract_json_prefix_on_extra_data(raw_text)
        if recovered is not None:
            parsed = recovered
        else:
            raise ProtocolError(f"invalid JSON: {exc}") from exc

    obj = _require_dict(parsed, "assistant_turn")
    if obj.get("type") != "assistant_turn":
        raise ProtocolError("assistant_turn.type must be 'assistant_turn'")
    version = str(obj.get("version", "")).strip()
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"assistant_turn.version must be '{PROTOCOL_VERSION}'")

    raw_steps = _require_list(obj.get("steps"), "assistant_turn.steps")
    steps: List[Step] = []
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
            purpose = str(step_obj.get("purpose", "progress") or "progress").strip() or "progress"
            allowed_purposes = {"progress", "clarification", "cannot_finish", "final"}
            if purpose not in allowed_purposes:
                raise ProtocolError(
                    f"steps[{idx}].purpose must be one of {sorted(allowed_purposes)}"
                )
            steps.append(MessageStep(type="message", content=content, format=fmt, purpose=purpose))
            continue

        if step_type == "tool_call":
            call_obj = _require_dict(step_obj.get("call"), f"steps[{idx}].call")
            tool = _require_str(call_obj.get("tool", ""), f"steps[{idx}].call.tool")
            if tool not in set(allowed_tools):
                raise ProtocolError(f"steps[{idx}].call.tool '{tool}' is not allowed")
            target = str(call_obj.get("target", "") or "")
            args = _coerce_args(call_obj.get("args", ""))
            steps.append(ToolCallStep(type="tool_call", call=ToolCallPayload(tool=tool, target=target, args=args)))
            continue

        if step_type == "end":
            reason = str(step_obj.get("reason", "") or "completed").strip() or "completed"
            steps.append(EndStep(type="end", reason=reason))
            continue

        # Back-compat / model convenience: allow task tool steps directly (they compile to tool_call).
        if step_type in TASK_TOOL_NAMES:
            if step_type not in set(allowed_tools):
                raise ProtocolError(f"steps[{idx}].type '{step_type}' is not allowed")
            steps.append(_parse_task_tool_alias_step(step_obj, allowed_tools=allowed_tools, idx=idx))
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


def collect_tool_calls(steps: Sequence[Step]) -> List[ToolCallPayload]:
    calls: List[ToolCallPayload] = []
    for step in steps:
        if isinstance(step, ToolCallStep):
            calls.append(step.call)
    return calls


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
