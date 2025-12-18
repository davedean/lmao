from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .jsonish import iter_jsonish_candidates, try_load_jsonish


PROTOCOL_VERSION = "2"
SUPPORTED_VERSIONS = {"1", "2"}


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ToolCallPayload:
    tool: str
    target: str
    args: Any
    meta: Optional[Dict[str, Any]] = None


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


def _coerce_args_v1(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _iter_jsonish_text_candidates(raw_text: str) -> Iterator[str]:
    yield from iter_jsonish_candidates(raw_text)


def _try_load_jsonish_dict(raw_text: str) -> Optional[Dict[str, Any]]:
    loaded = try_load_jsonish(raw_text, recover_extra_data=True)
    return loaded if isinstance(loaded, dict) else None


def _load_jsonish_dict(raw_text: str) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for candidate in _iter_jsonish_text_candidates(raw_text):
        loaded = _try_load_jsonish_dict(candidate)
        if loaded is not None:
            return loaded
    try:
        json.loads(raw_text)
        raise ProtocolError("assistant_turn must be an object")
    except Exception as exc:
        last_error = exc
    raise ProtocolError(f"invalid JSON: {last_error}") from last_error


def _wrap_single_step(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "assistant_turn", "version": PROTOCOL_VERSION, "steps": [obj]}



def parse_assistant_turn(raw_text: str, allowed_tools: Sequence[str]) -> AssistantTurn:
    acceptable_types = {"assistant_turn", "think", "tool_call", "message", "end"}
    obj: Optional[Dict[str, Any]] = None
    first_loaded: Optional[Dict[str, Any]] = None
    for candidate in _iter_jsonish_text_candidates(raw_text):
        loaded = _try_load_jsonish_dict(candidate)
        if loaded is None:
            continue
        if first_loaded is None:
            first_loaded = loaded
        step_type = loaded.get("type")
        if step_type in acceptable_types:
            obj = loaded
            break
    if obj is None:
        obj = first_loaded if first_loaded is not None else _load_jsonish_dict(raw_text)
    if obj.get("type") != "assistant_turn":
        # Back-compat / model convenience: accept a single step dict and wrap it.
        if obj.get("type") in {"think", "tool_call", "message", "end"} or obj.get("type") in set(allowed_tools):
            obj = _wrap_single_step(obj)
        else:
            raise ProtocolError("assistant_turn.type must be 'assistant_turn'")
    version = str(obj.get("version", "")).strip()
    lowered = version.lower()
    if lowered in ("v1", "v2"):
        version = lowered[1:]
    if not version:
        version = PROTOCOL_VERSION
    if version not in SUPPORTED_VERSIONS:
        raise ProtocolError(f"assistant_turn.version must be one of {sorted(SUPPORTED_VERSIONS)}")

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
            purpose_aliases = {
                # Common model mistakes
                "success": "progress",
                "ok": "progress",
                "done": "progress",
                "complete": "progress",
                "completed": "progress",
                "result": "progress",
                "answer": "final",
                "conclusion": "final",
            }
            purpose = purpose_aliases.get(purpose, purpose)
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
            call_meta_val = call_obj.get("meta", None)
            call_meta: Optional[Dict[str, Any]] = None
            if call_meta_val is not None:
                call_meta = _require_dict(call_meta_val, f"steps[{idx}].call.meta")

            if version == "1":
                call_args: Any = _coerce_args_v1(call_obj.get("args", ""))
            else:
                call_args = call_obj.get("args", None)
            steps.append(
                ToolCallStep(
                    type="tool_call",
                    call=ToolCallPayload(tool=tool, target=target, args=call_args, meta=call_meta),
                )
            )
            continue

        if step_type == "end":
            reason = str(step_obj.get("reason", "") or "completed").strip() or "completed"
            steps.append(EndStep(type="end", reason=reason))
            continue

        # Back-compat / model convenience: allow direct tool steps (type == tool name) for any allowed tool.
        # Example: {"type":"write","target":"a.txt","args":"content"}.
        if step_type in set(allowed_tools):
            tool = step_type
            target = str(step_obj.get("target", "") or "")
            direct_meta_val = step_obj.get("meta", None)
            direct_meta: Optional[Dict[str, Any]] = None
            if direct_meta_val is not None:
                direct_meta = _require_dict(direct_meta_val, f"steps[{idx}].meta")
            payload = step_obj.get("args", None)
            if version == "1":
                direct_args: Any = _coerce_args_v1(payload)
            else:
                direct_args = payload
            steps.append(
                ToolCallStep(
                    type="tool_call",
                    call=ToolCallPayload(tool=tool, target=target, args=direct_args, meta=direct_meta),
                )
            )
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
