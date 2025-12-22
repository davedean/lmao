from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .context import build_tool_prompt
from .llm import LLMCallResult, LLMClient, LLMCallStats
from .plugins import PluginTool
from .protocol import (
    ProtocolError,
    collect_steps,
    collect_tool_calls,
    parse_assistant_turn_with_hooks,
)
from .tool_dispatch import json_error, json_success, run_tool
from .tool_parsing import ToolCall


@dataclass(frozen=True)
class SubAgentResult:
    status: str  # ok|blocked|needs_user|error
    summary: str
    findings: List[str]
    assumptions: List[str]
    open_questions: List[str]
    recommended_next_steps: List[str]
    artifacts: List[Dict[str, str]]
    turns: int


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _subagent_system_prompt(
    *,
    allowed_tools: Sequence[str],
    plugin_tools: Dict[str, PluginTool],
) -> Dict[str, str]:
    # Reuse the main tool prompt scaffolding so models don't "forget" the assistant_turn protocol
    # inside sub-agent runs.
    base_prompt = build_tool_prompt(
        list(allowed_tools),
        read_only=True,
        yolo_enabled=False,
        plugins=list(plugin_tools.values()),
        runtime_tools=None,
    )
    content = "\n".join(
        [
            "SUBAGENT MODE:",
            "- You are a sub-agent running inside a larger agent loop.",
            "- You are NOT interacting with the human user directly.",
            "- Use tools to make progress; do not ask the user for permission or clarifications unless strictly required.",
            "- When finished: send a message step with purpose='final' AND include an end step.",
            "",
            base_prompt,
        ]
    )
    return {"role": "system", "content": content}


def _subagent_user_prompt(objective: str, context: str) -> Dict[str, str]:
    parts = [f"Objective:\n{objective.strip()}"]
    if context.strip():
        parts.append(f"Context:\n{context.strip()}")
    parts.append(
        "Return progress via tool calls; do not ask the human user questions unless strictly required."
    )
    return {"role": "user", "content": "\n\n".join(parts)}


def _final_message(messages: List[Any]) -> str:
    finals = [m.content for m in messages if getattr(m, "purpose", "") == "final"]
    if finals:
        return finals[-1].strip()
    rendered = "\n".join(_coerce_str(m.content).strip() for m in messages if getattr(m, "content", None))
    return rendered.strip()


def _default_result(status: str, summary: str, *, turns: int) -> SubAgentResult:
    return SubAgentResult(
        status=status,
        summary=summary.strip(),
        findings=[],
        assumptions=[],
        open_questions=[],
        recommended_next_steps=[],
        artifacts=[],
        turns=turns,
    )


def run_subagent_one_shot(
    *,
    client: LLMClient,
    objective: str,
    context: str,
    max_turns: int,
    plugin_tools: Dict[str, PluginTool],
    base,
    extra_roots,
    skill_roots,
    allowed_tools: Sequence[str],
    yolo_enabled: bool,
    read_only: bool,
    debug_logger=None,
) -> Tuple[SubAgentResult, Optional[LLMCallStats]]:
    messages: List[Dict[str, str]] = [
        _subagent_system_prompt(allowed_tools=allowed_tools, plugin_tools=plugin_tools),
        _subagent_user_prompt(objective, context),
    ]

    last_stats: Optional[LLMCallStats] = None
    invalid_replies = 0
    tool_calls_total = 0

    for turn in range(1, max(1, int(max_turns)) + 1):
        result: LLMCallResult = client.call(messages)
        last_stats = result.stats
        assistant_reply = result.content

        try:
            turn_obj = parse_assistant_turn_with_hooks(
                assistant_reply, allowed_tools=allowed_tools
            )
        except ProtocolError as exc:
            invalid_replies += 1
            messages.append(
                {
                    "role": "user",
                    "content": f"LOOP: Protocol error: {exc}. Return ONLY valid assistant_turn JSON.",
                }
            )
            if invalid_replies >= 3:
                return _default_result("error", f"protocol errors: {exc}", turns=turn), last_stats
            continue

        thinks, msg_steps, has_end = collect_steps(turn_obj.steps)
        calls = collect_tool_calls(turn_obj.steps)

        # Model convenience: if it produces a final report but forgets the explicit end step,
        # treat it as completion rather than looping until max_turns.
        if not calls and any(getattr(step, "purpose", "") == "final" for step in msg_steps):
            summary = _final_message(msg_steps)
            return _default_result("ok", summary or "completed", turns=turn), last_stats

        # Execute tool calls sequentially, feeding results back into the sub-agent.
        for call in calls:
            tool_calls_total += 1
            if tool_calls_total > 50:
                return _default_result("blocked", "tool call budget exceeded", turns=turn), last_stats
            tool_call = ToolCall(tool=call.tool, target=call.target, args=call.args, meta=call.meta)
            tool_result = run_tool(
                tool_call,
                base=base,
                extra_roots=extra_roots,
                skill_roots=skill_roots,
                yolo_enabled=yolo_enabled,
                read_only=read_only,
                plugin_tools=plugin_tools,
                runtime_tools=None,
                runtime_context=runtime_ctx,
                debug_logger=debug_logger,
            )
            messages.append({"role": "user", "content": f"LOOP: Tool result: {tool_result}"})

        if has_end:
            summary = _final_message(msg_steps)
            if not summary and thinks:
                summary = _coerce_str(thinks[-1].content)
            return _default_result("ok", summary or "completed", turns=turn), last_stats

        # Keep the model focused if it doesn't call tools or end.
        if not calls:
            messages.append(
                {
                    "role": "user",
                    "content": "LOOP: No tool calls or end step observed. Either call tools to make progress, or send a final report and end.",
                }
            )

    return _default_result("blocked", "max turns reached", turns=max_turns), last_stats


def subagent_run_tool(runtime_ctx, target: str, args: Any, meta: Optional[dict]) -> str:
    if isinstance(args, dict):
        objective = _coerce_str(args.get("objective") or args.get("task") or "").strip()
        context = _coerce_str(args.get("context") or "").strip()
        max_turns = int(args.get("max_turns") or 6)
        profile = str(args.get("allowed_tools_profile") or args.get("profile") or "read_only").strip()
    else:
        objective = _coerce_str(args).strip()
        context = ""
        max_turns = 6
        profile = "read_only"

    if not objective:
        objective = _coerce_str(target).strip()

    if not objective:
        return json_error("subagent_run", "args.objective is required")
    if max_turns < 1:
        max_turns = 1
    if max_turns > 20:
        max_turns = 20

    if profile != "read_only":
        return json_error("subagent_run", "only allowed_tools_profile='read_only' is supported in v1")

    # Sub-agents run read-only by default, regardless of main mode, to keep delegation safe.
    sub_read_only = True
    allowed_tools = [
        name
        for name, tool in runtime_ctx.plugin_tools.items()
        if tool.allow_in_read_only
    ]

    result, stats = run_subagent_one_shot(
        client=runtime_ctx.client,
        objective=objective,
        context=context,
        max_turns=max_turns,
        plugin_tools=runtime_ctx.plugin_tools,
        base=runtime_ctx.base,
        extra_roots=runtime_ctx.extra_roots,
        skill_roots=runtime_ctx.skill_roots,
        allowed_tools=allowed_tools,
        yolo_enabled=False,
        read_only=sub_read_only,
        debug_logger=runtime_ctx.debug_logger,
    )
    data = {
        "result": {
            "status": result.status,
            "summary": result.summary,
            "findings": result.findings,
            "assumptions": result.assumptions,
            "open_questions": result.open_questions,
            "recommended_next_steps": result.recommended_next_steps,
            "artifacts": result.artifacts,
        },
        "turns": result.turns,
    }
    if stats is not None:
        data["stats"] = {
            "elapsed_s": stats.elapsed_s,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "total_tokens": stats.total_tokens,
            "is_estimate": stats.is_estimate,
        }
    return json_success("subagent_run", data)
