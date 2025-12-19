from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .debug_log import DebugLogger
from .context import build_system_message, gather_context
from .llm import LLMCallStats, LLMClient
from .protocol import ProtocolError, collect_steps, collect_tool_calls, parse_assistant_turn
from .tools import ToolCall, get_allowed_tools, run_tool, summarize_output
from .plugins import PluginTool, discover_plugins
from .text_utils import truncate_text
from .memory import (
    MemoryState,
    aggressive_compact_messages,
    compact_messages_if_needed,
    determine_prompt_budget,
    is_context_length_error,
    sanitize_assistant_reply,
    should_pin_agents_tool_result,
    truncate_tool_result_for_prompt,
)
from .runtime_tools import RuntimeContext, RuntimeTool, build_runtime_tool_registry
from .user_input import read_user_prompt

COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"

ACTION_REQUIRED_PREFIX = "ACTION_REQUIRED:"
LOOP_PREFIX = "LOOP:"

_HEADLESS_INPUT_REQUEST_PATTERNS = (
    "would you like",
    "do you want",
    "can you",
    "could you",
    "please provide",
    "please share",
    "please confirm",
    "please clarify",
    "which one",
    "which should",
    "what should i",
    "should i",
    "let me know",
    "tell me",
    "need more info",
    "need more information",
    "what is your",
    "what are your",
)


def _headless_requests_user_input(messages: Sequence[str]) -> bool:
    for content in messages:
        text = (content or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if any(pat in lowered for pat in _HEADLESS_INPUT_REQUEST_PATTERNS):
            return True
        if "?" in lowered:
            # Question marks alone are too noisy; require at least one "request" indicator.
            if re.search(r"\b(you|your|please|which|what|confirm|clarif|provide|choose)\b", lowered):
                return True
    return False


def _remove_prefixed_messages(messages: List[Dict[str, str]], prefix: str) -> None:
    messages[:] = [
        msg
        for msg in messages
        if not (
            msg.get("role") in ("system", "user")
            and str(msg.get("content", "")).startswith(prefix)
        )
    ]

def _upsert_action_required(messages: List[Dict[str, str]], content: str) -> None:
    # NOTE: Some models (e.g., Qwen3) may not honor system messages after the first response.
    # Emit loop instructions as role='user' with a distinct prefix so the model still follows them.
    _remove_prefixed_messages(messages, ACTION_REQUIRED_PREFIX)
    messages.append({"role": "user", "content": f"{ACTION_REQUIRED_PREFIX}\n{LOOP_PREFIX} {content}"})


def _truncate_preview(text: str, max_lines: int = 4, max_chars: int = 400) -> str:
    return truncate_text(text, max_lines=max_lines, max_chars=max_chars)

MESSAGE_PURPOSE_CLARIFICATION = "clarification"
MESSAGE_PURPOSE_CANNOT_FINISH = "cannot_finish"


def run_agent_turn(
    messages: List[Dict[str, str]],
    client: LLMClient,
    turn: int,
    last_user: str,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    max_tool_output: Tuple[int, int],
    yolo_enabled: bool,
    read_only: bool,
    allowed_tools: Sequence[str],
    plugin_tools: Dict[str, PluginTool],
    show_stats: bool,
    debug_logger: Optional[DebugLogger] = None,
    runtime_tools: Optional[Dict[str, RuntimeTool]] = None,
    runtime_context: Optional[RuntimeContext] = None,
) -> Tuple[int, Optional[LLMCallStats], bool]:
    """
    Run the agent for a single user-visible turn.

    This function repeatedly calls the model and executes any requested tool calls until the model
    produces a terminal step (typically `end`) or it becomes clear we cannot safely proceed.

    Key invariants:
    - Headless mode forbids asking the human user for clarification; the model must either proceed
      autonomously or emit purpose='cannot_finish' before ending.
    - Think-only turns are disallowed: the model must follow up with a tool_call or message/end step.

    Returns `(next_turn, last_stats, ended)` where `ended` indicates the conversation may stop.
    """
    empty_replies = 0
    last_tool_summary: Optional[str] = None
    last_stats: Optional[LLMCallStats] = None
    current_turn = turn
    invalid_replies = 0
    think_only_turns = 0
    progress_only_turns = 0

    def indent(text: str) -> str:
        return "\n".join(f"    {line}" for line in text.splitlines())

    memory_state = runtime_context.memory_state if runtime_context and runtime_context.memory_state else None
    _, trigger_tokens, target_tokens = determine_prompt_budget(client)

    while True:
        headless_run = bool(runtime_context.headless) if runtime_context else False
        pinned_ids = memory_state.pinned_message_ids if memory_state else set()
        last_user_message = memory_state.last_user_message if memory_state else None
        attempt_retry = False
        while True:
            compact_messages_if_needed(
                messages,
                last_user_message=last_user_message,
                pinned_message_ids=pinned_ids,
                trigger_tokens=trigger_tokens,
                target_tokens=target_tokens,
                debug_logger=debug_logger,
            )
            try:
                result = client.call(messages)
                break
            except RuntimeError as exc:
                if not attempt_retry and is_context_length_error(str(exc)):
                    if memory_state:
                        aggressive_compact_messages(
                            messages,
                            last_user_message=last_user_message,
                            pinned_message_ids=pinned_ids,
                            debug_logger=debug_logger,
                        )
                    attempt_retry = True
                    if debug_logger:
                        debug_logger.log(
                            "memory.compact.retry",
                            "aggressive compaction triggered after context-length error",
                        )
                    continue
                raise
        assistant_reply = result.content
        last_stats = result.stats

        if show_stats:
            estimate_mark = "~" if last_stats.is_estimate else ""
            stats_line = (
                f"[stats] in={estimate_mark}{last_stats.prompt_tokens} "
                f"out={estimate_mark}{last_stats.completion_tokens} "
                f"total={estimate_mark}{last_stats.total_tokens} "
                f"time={last_stats.elapsed_s:.2f}s "
                f"bytes={last_stats.request_bytes}/{last_stats.response_bytes} "
                f"msgs={len(messages)}"
            )
            print(f"{COLOR_DIM}{stats_line}{COLOR_RESET}")

        if not assistant_reply or not assistant_reply.strip():
            empty_replies += 1
            if empty_replies >= 4:
                fallback = last_tool_summary or "No tool output available."
                assistant_reply = (
                    f"(auto-generated fallback) Unable to get a response from the model. "
                    f"Based on the latest tool output, here is a summary:\n{fallback}"
                )
                print(f"\n{COLOR_BLUE}[assistant #{current_turn}]{COLOR_RESET}\n{assistant_reply}\n")
                messages.append({"role": "assistant", "content": assistant_reply})
                return current_turn + 1, last_stats, False
            _upsert_action_required(
                messages,
                (
                    "Your last reply was empty (or whitespace). This is not allowed.\n"
                    f"The user asked: {last_user!r}.\n"
                    "Return ONLY a single JSON object matching the assistant protocol.\n"
                    "Do NOT return whitespace.\n\n"
                    "If you are unsure what to do next, emit a tool_call or a short progress message."
                ),
            )
            current_turn += 1
            continue

        empty_replies = 0
        if debug_logger:
            debug_logger.log("assistant.reply", f"turn={current_turn} content={assistant_reply}")

        try:
            turn_obj = parse_assistant_turn(assistant_reply, allowed_tools=allowed_tools)
        except ProtocolError as exc:
            invalid_replies += 1
            if debug_logger:
                print(f"{COLOR_DIM}[protocol] invalid: {exc} (retry {invalid_replies}/2){COLOR_RESET}")
            if invalid_replies > 2:
                message = (
                    "error: model repeatedly returned invalid JSON protocol output.\n"
                    f"last error: {exc}\n"
                    "last reply (verbatim):\n"
                    f"{assistant_reply}"
                )
                print(f"\n{COLOR_BLUE}[assistant #{current_turn}]{COLOR_RESET}\n{message}\n")
                messages.append({"role": "assistant", "content": assistant_reply})
                return current_turn + 1, last_stats, False

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{LOOP_PREFIX} Your reply was not valid for the required JSON assistant protocol.\n"
                        f"Error: {exc}\n"
                        "Return ONLY a single JSON object matching:\n"
                        '{"type":"assistant_turn","version":"2","steps":[...]}\n'
                        "No code fences, no extra text. Retry now."
                    ),
                }
            )
            current_turn += 1
            continue

        # Recovery prompts are only for the immediate next attempt; remove them once we get a valid turn.
        _remove_prefixed_messages(messages, ACTION_REQUIRED_PREFIX)

        label_color = COLOR_BLUE
        thinks, user_messages, has_end = collect_steps(turn_obj.steps)
        tool_call_payloads = collect_tool_calls(turn_obj.steps)
        explicit_clarification_requested = any(
            msg.purpose == MESSAGE_PURPOSE_CLARIFICATION for msg in user_messages
        )
        implicit_input_requested = _headless_requests_user_input(
            [
                msg.content
                for msg in user_messages
                if msg.purpose not in (MESSAGE_PURPOSE_CANNOT_FINISH, "final")
            ]
        )
        input_requested = explicit_clarification_requested or implicit_input_requested
        headless_input_requested = headless_run and (
            explicit_clarification_requested
            or _headless_requests_user_input(
                [
                    msg.content
                    for msg in user_messages
                    if msg.purpose not in (MESSAGE_PURPOSE_CANNOT_FINISH, "final")
                ]
            )
        )

        if debug_logger:
            step_summary = ",".join(step.type for step in turn_obj.steps) if turn_obj.steps else "(no steps)"
            print(f"{COLOR_DIM}[protocol] ok steps={step_summary}{COLOR_RESET}")

        print(f"\n{label_color}[assistant #{current_turn}]{COLOR_RESET}")
        if headless_input_requested:
            print(
                f"{COLOR_DIM}(requests for user input are disallowed in headless mode; proceed autonomously or emit purpose='cannot_finish' and end){COLOR_RESET}"
            )
            if user_messages:
                combined = "\n\n".join(step.content for step in user_messages)
                preview = _truncate_preview(combined, max_lines=4, max_chars=320)
                if preview:
                    purposes = ", ".join((step.purpose or "progress") for step in user_messages)
                    print(f"{COLOR_DIM}(message preview; purpose={purposes})\n{preview}{COLOR_RESET}")
        if thinks:
            think_preview = _truncate_preview("\n\n".join(step.content for step in thinks), max_lines=3, max_chars=240)
            if think_preview:
                print(f"{COLOR_DIM}(think preview)\n{think_preview}{COLOR_RESET}")
        if user_messages and not headless_input_requested:
            combined = "\n\n".join(step.content for step in user_messages)
            print(f"{combined}\n")
        elif tool_call_payloads:
            tools = ", ".join(call.tool for call in tool_call_payloads)
            print(f"{COLOR_DIM}(requesting tool(s): {tools}){COLOR_RESET}\n")
        else:
            print()

        # Preserve protocol conversation history for the model by storing the JSON turn without think steps.
        sanitized_reply = sanitize_assistant_reply(assistant_reply, allowed_tools)
        if turn_obj.steps:
            messages.append({"role": "assistant", "content": sanitized_reply})
        if headless_input_requested:
            _upsert_action_required(
                messages,
                (
                    "Headless mode is active: the human user cannot respond, so do NOT ask questions or request confirmation.\n"
                    "Proceed autonomously: pick reasonable defaults, state assumptions briefly, and continue (call tools if helpful).\n"
                    "If you are truly blocked, send a message step with purpose='cannot_finish' describing what's missing, then end."
                ),
            )
            current_turn += 1
            continue

        if user_messages and not tool_call_payloads and not has_end and not input_requested:
            progress_only_turns += 1
            purposes = {msg.purpose for msg in user_messages}
            needs_explicit_end = bool(purposes.intersection({"final", MESSAGE_PURPOSE_CANNOT_FINISH}))
            if progress_only_turns >= 4:
                reminder = (
                    "You have emitted multiple message-only turns without taking an action. "
                    "This is not allowed.\n"
                )
            else:
                reminder = ""
            if needs_explicit_end:
                instruction = (
                    f"{reminder}"
                    "You sent a terminal message (purpose='final' or 'cannot_finish') but did not include an end step.\n"
                    "Continue immediately by returning a new assistant_turn that includes an explicit end step.\n"
                    "Return ONLY JSON; do not ask the user to type 'ok' or provide follow-ups."
                )
            else:
                instruction = (
                    f"{reminder}"
                    "You sent a progress message but did not call any tools and did not end.\n"
                    "Continue immediately without waiting for user input: either call a tool (tool_call steps) or, if finished, send purpose='final' AND an explicit end step.\n"
                    "Do not ask the user to type 'ok' or otherwise prompt for input unless you truly need clarification (then set purpose='clarification')."
                )
            _upsert_action_required(messages, instruction)
            current_turn += 1
            continue

        if thinks and not user_messages and not tool_call_payloads and not has_end:
            think_only_turns += 1
            next_json = (
                '{"type":"assistant_turn","version":"2","steps":['
                '{"type":"tool_call","call":{"tool":"read","target":"README.md","args":"lines:1-40"}}'
                "]}"
            )
            if think_only_turns > 3:
                _upsert_action_required(
                    messages,
                    (
                        "You have emitted multiple think-only turns. This is not allowed.\n"
                        "Next, emit either tool_call steps or message/end steps.\n\n"
                        "Return ONLY a single JSON object matching the assistant protocol.\n"
                        f"Example next action:\n{next_json}"
                    ),
                )
            else:
                _upsert_action_required(
                    messages,
                    (
                        "You produced only think steps. Continue immediately with the next action.\n\n"
                        "Return ONLY a single JSON object matching the assistant protocol.\n"
                        f"Example next action:\n{next_json}"
                    ),
                )
            current_turn += 1
            continue
        progress_only_turns = 0

        if tool_call_payloads:
            for call in tool_call_payloads:
                tool_call = ToolCall(tool=call.tool, target=call.target, args=call.args, meta=call.meta)
                if debug_logger:
                    debug_logger.log(
                        "tool.call",
                        f"turn={current_turn} tool={tool_call.tool} target={tool_call.target!r} args={tool_call.args!r} meta={tool_call.meta!r}",
                    )
                output = run_tool(
                    tool_call,
                    base,
                    extra_roots,
                    skill_roots,
                    yolo_enabled,
                    read_only=read_only,
                    plugin_tools=plugin_tools,
                    runtime_tools=runtime_tools,
                    runtime_context=runtime_context,
                    debug_logger=debug_logger,
                )
                tool_desc = f"tool '{tool_call.tool}' on '{tool_call.target}'"
                last_tool_summary = summarize_output(output, max_lines=max_tool_output[0], max_chars=max_tool_output[1])
                if max_tool_output[1] > 0:
                    args_repr = ""
                    if tool_call.args not in ("", None, {}):
                        args_repr = f" args: {tool_call.args!r}"
                    tool_header = f"tool: {tool_call.tool}, {tool_call.target!r}{args_repr}"
                    tool_result_body = indent(f"[tool result] {tool_header}\n{last_tool_summary}")
                    print(f"{COLOR_DIM}{tool_result_body}{COLOR_RESET}\n")

                if debug_logger:
                    debug_logger.log(
                        "tool.output",
                        f"turn={current_turn} tool={tool_call.tool} target={tool_call.target!r} args={tool_call.args!r} output={output}",
                    )
                is_pinned = should_pin_agents_tool_result(tool_call.tool, tool_call.target or "")
                tool_result_content = (
                    f"Tool result for {tool_desc}:\n{output}\n"
                    "Use this to continue. If another tool is needed, call it; otherwise continue to the next step."
                )
                truncated_content, _ = truncate_tool_result_for_prompt(
                    tool_result_content,
                    is_pinned=is_pinned,
                )
                tool_message = {"role": "user", "content": truncated_content}
                messages.append(tool_message)
                if is_pinned and memory_state:
                    memory_state.pinned_message_ids.add(id(tool_message))
                current_turn += 1
            continue

        if has_end:
            return current_turn + 1, last_stats, True

        return current_turn + 1, last_stats, False

def run_loop(
    initial_prompt: Optional[str],
    client: LLMClient,
    workdir: Path,
    max_tool_output: Tuple[int, int],
    max_turns: Optional[int],
    silent_tools: bool,
    yolo_enabled: bool,
    read_only: bool,
    show_stats: bool,
    headless: bool = False,
    multiline: bool = False,
    plugin_dirs: Optional[Sequence[Path]] = None,
    debug_logger: Optional[DebugLogger] = None,
    policy_truncate: bool = True,
    policy_truncate_chars: int = 2000,
) -> None:
    base = workdir.resolve()
    if debug_logger:
        debug_logger.log("debug.enabled", f"writing debug logs to {debug_logger.path}")
        debug_logger.log(
            "context",
            f"workdir={base} yolo_enabled={yolo_enabled} read_only={read_only} headless={headless}",
        )

    notes = gather_context(base)

    extra_roots: List[Path] = []
    skill_roots: List[Path] = [base / "skills"]
    if notes.user_skills:
        resolved_user_skills = notes.user_skills.resolve()
        extra_roots.append(resolved_user_skills)
        skill_roots.append(resolved_user_skills)

    plugins = discover_plugins(plugin_dirs or [], base, debug_logger=debug_logger, allow_outside_base=True)
    if debug_logger and plugins:
        debug_logger.log("plugins.loaded", f"{[(name, str(plugin.path)) for name, plugin in plugins.items()]}")
    runtime_tools = build_runtime_tool_registry()
    allowed_tools = get_allowed_tools(read_only=read_only, yolo_enabled=yolo_enabled, plugins=list(plugins.values()))
    allowed_tools = list(allowed_tools) + sorted(runtime_tools.keys())
    mode_label = "read-only" if read_only else ("yolo" if yolo_enabled else "normal")
    if headless:
        mode_label += " (headless)"
    last_prompt_stats: Optional[LLMCallStats] = None

    def format_user_prompt() -> str:
        if not show_stats or not last_prompt_stats:
            return f"{COLOR_GREEN}You [mode: {mode_label}]:{COLOR_RESET} "
        estimate_mark = "~" if last_prompt_stats.is_estimate else ""
        stats = (
            f"in={estimate_mark}{last_prompt_stats.prompt_tokens} "
            f"out={estimate_mark}{last_prompt_stats.completion_tokens} "
            f"t={last_prompt_stats.elapsed_s:.2f}s"
        )
        return f"{COLOR_GREEN}You [mode: {mode_label} | {stats}]:{COLOR_RESET} "

    user_prompt = format_user_prompt()
    messages: List[Dict[str, str]] = [
        build_system_message(
            base,
            notes,
            read_only=read_only,
            yolo_enabled=yolo_enabled,
            allowed_tools=allowed_tools,
            plugins=list(plugins.values()),
            runtime_tools=list(runtime_tools.values()),
            headless=headless,
            debug=debug_logger is not None,
        )
    ]
    user_input = initial_prompt
    turn = 1
    memory_state = MemoryState()
    runtime_ctx = RuntimeContext(
        client=client,
        plugin_tools=plugins,
        base=base,
        extra_roots=extra_roots,
        skill_roots=skill_roots,
        yolo_enabled=yolo_enabled,
        read_only=read_only,
        headless=headless,
        debug_logger=debug_logger,
        memory_state=memory_state,
    )
    startup_prelude_done = False
    startup_user_input: Optional[str] = None

    def _should_include_skills_guide_startup(prompt: str) -> bool:
        lowered = (prompt or "").lower()
        triggers = (
            "skill",
            "skills",
            "skill.md",
            "creating a skill",
            "create a skill",
            "write a skill",
            "list skills",
        )
        return any(trigger in lowered for trigger in triggers)

    def _append_startup_tool_result(tool_name: str, *, last_user: str) -> None:
        if tool_name not in plugins:
            if debug_logger:
                debug_logger.log("startup.prelude.skip", f"missing_tool={tool_name}")
            return
        if tool_name not in allowed_tools:
            if debug_logger:
                debug_logger.log("startup.prelude.skip", f"tool_not_allowed={tool_name}")
            return
        tool_args: object = ""
        if tool_name == "policy":
            tool_args = {"truncate": bool(policy_truncate)}
            if policy_truncate:
                limit = int(policy_truncate_chars)
                if limit <= 0:
                    limit = 2000
                tool_args = {"truncate": True, "offset": 0, "limit": limit}
        tool_call = ToolCall(tool=tool_name, target="", args=tool_args, meta=None)
        output = run_tool(
            tool_call,
            base,
            extra_roots,
            skill_roots,
            yolo_enabled,
            read_only=read_only,
            plugin_tools=plugins,
            runtime_tools=runtime_tools,
            runtime_context=runtime_ctx,
            debug_logger=debug_logger,
        )
        try:
            payload = json.loads(output)
            success = bool(payload.get("success"))
        except Exception:
            success = True
        if not success:
            if debug_logger:
                debug_logger.log("startup.prelude.tool_error", f"tool={tool_name} output={output}")
            return

        tool_desc = f"tool '{tool_name}' on ''"
        is_pinned = should_pin_agents_tool_result(tool_name, "")
        pin_for_memory = is_pinned or tool_name == "skills_guide"
        tool_result_content = (
            f"Tool result for {tool_desc}:\n{output}\n"
            "Startup prelude: treat these instructions as authoritative for this run.\n"
            "Use this to continue."
        )
        truncated_content, _ = truncate_tool_result_for_prompt(tool_result_content, is_pinned=is_pinned)
        tool_message = {"role": "user", "content": truncated_content}
        messages.append(tool_message)
        if pin_for_memory:
            memory_state.pinned_message_ids.add(id(tool_message))
        if debug_logger:
            debug_logger.log("startup.prelude.tool", f"tool={tool_name} pinned={pin_for_memory}")

    if debug_logger and initial_prompt is not None:
        debug_logger.log("user.initial_prompt", initial_prompt)

    if user_input is None:
        if headless:
            return
        result = read_user_prompt(user_prompt, multiline_default=multiline)
        if result.eof:
            return
        user_input = result.text or ""
        if debug_logger:
            debug_logger.log("user.input", f"turn={turn} content={user_input}")

    root_request: Optional[str] = None

    while True:
        if max_turns is not None and turn > max_turns:
            print(f"Reached max turns ({max_turns}); stopping.")
            if debug_logger:
                debug_logger.log("loop.stop", f"reason=max_turns reached={max_turns}")
            return
        if not user_input:
            if headless:
                return
            result = read_user_prompt(user_prompt, multiline_default=multiline)
            if result.eof:
                return
            user_input = result.text or ""
            if debug_logger:
                debug_logger.log("user.input", f"turn={turn} content={user_input}")
            continue

        if not user_input.strip():
            user_input = ""
            continue

        if root_request is None:
            root_request = user_input
            startup_user_input = user_input

        if not startup_prelude_done:
            _append_startup_tool_result("policy", last_user=user_input)
            if startup_user_input and _should_include_skills_guide_startup(startup_user_input):
                _append_startup_tool_result("skills_guide", last_user=user_input)
            startup_prelude_done = True

        user_message = {"role": "user", "content": user_input}
        messages.append(user_message)
        memory_state.last_user_message = user_message
        if debug_logger:
            debug_logger.log("user.input", f"turn={turn} content={user_input}")
        next_turn, last_prompt_stats, ended = run_agent_turn(
            messages,
            client=client,
            turn=turn,
            last_user=user_input,
            base=base,
            extra_roots=extra_roots,
            skill_roots=skill_roots,
            max_tool_output=max_tool_output if not silent_tools else (0, 0),
            yolo_enabled=yolo_enabled,
            read_only=read_only,
            allowed_tools=allowed_tools,
            plugin_tools=plugins,
            runtime_tools=runtime_tools,
            runtime_context=runtime_ctx,
            show_stats=show_stats,
            debug_logger=debug_logger,
        )
        turn = next_turn
        if ended:
            if headless:
                return
            print(
                f"{COLOR_DIM}(model requested end; enter a new prompt to continue, or press Ctrl-D to exit){COLOR_RESET}"
            )
        if headless:
            # Headless mode has no interactive user to provide follow-ups. If the model did not end,
            # prompt it to continue autonomously (tools or final+end) toward the original request.
            request = root_request or ""
            user_input = (
                f"{LOOP_PREFIX} Headless mode: continue autonomously without waiting for user input.\n"
                f"Original request: {request!r}\n"
                "If you can proceed, take the next action now (call tools as needed).\n"
                "If you are finished, send a final summary message AND include an explicit end step.\n"
                "Important: a message step with purpose='final' does NOT stop the run; only an end step stops it.\n"
                "Do not ask questions or request confirmation; proceed autonomously using reasonable defaults.\n"
                "If you are blocked and cannot proceed safely, send purpose='cannot_finish' and then end.\n"
                "Example finish JSON: "
                "{\"type\":\"assistant_turn\",\"version\":\"2\",\"steps\":["
                "{\"type\":\"message\",\"purpose\":\"final\",\"format\":\"markdown\",\"content\":\"...\"},"
                "{\"type\":\"end\",\"reason\":\"completed\"}"
                "]}"
            )
            continue
        user_prompt = format_user_prompt()

        try:
            user_input = input(user_prompt).strip()
        except EOFError:
            return
        if debug_logger and user_input is not None:
            debug_logger.log("user.input", f"turn={turn} content={user_input}")
