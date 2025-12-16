from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .debug_log import DebugLogger
from .context import build_system_message, gather_context
from .governance import (
    can_end_conversation,
    has_incomplete_tasks,
    should_render_user_messages,
)
from .llm import LLMCallStats, LLMClient
from .protocol import ProtocolError, collect_steps, collect_tool_calls, parse_assistant_turn
from .task_list import TaskListManager
from .tools import ToolCall, get_allowed_tools, run_tool, summarize_output
from .plugins import PluginTool, discover_plugins
from .text_utils import truncate_text
from .runtime_tools import RuntimeContext, RuntimeTool, build_runtime_tool_registry
from .user_input import read_user_prompt

COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"

MAX_TOOL_CALLS_PER_TURN = 6
MAX_DESTRUCTIVE_TOOL_CALLS_PER_TURN = 1

GOVERNANCE_NOTICE_PREFIX = "GOVERNANCE_NOTICE:"
ACTION_REQUIRED_PREFIX = "ACTION_REQUIRED:"
LOOP_PREFIX = "LOOP:"


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


def _upsert_governance_notice(
    messages: List[Dict[str, str]],
    task_manager: TaskListManager,
    gate_violations: int,
) -> None:
    """
    Keep a single up-to-date governance notice at the end of the message history.

    This avoids the prompt growing with repeated reminders and helps keep the model un-stuck.
    """
    if not has_incomplete_tasks(task_manager) and gate_violations <= 0:
        _remove_prefixed_messages(messages, GOVERNANCE_NOTICE_PREFIX)
        return

    rendered = task_manager.render_tasks()
    strict_next = ""
    if gate_violations > 0:
        first_incomplete = next((t for t in task_manager.tasks if not t.done), None)
        if first_incomplete:
            strict_next = (
                "\nExample next step (valid JSON):\n"
                '{"type":"assistant_turn","version":"1","steps":[{"type":"tool_call","call":{"tool":"complete_task","target":"","args":"'
                + str(first_incomplete.id)
                + '"}}]}'
            )
    content = (
        f"{GOVERNANCE_NOTICE_PREFIX}\n"
        f"Current task list:\n{rendered}\n\n"
        "Rules:\n"
        "- Do not send message steps with purpose 'progress' or 'final' until all tasks are complete.\n"
        "- If you need user input, send a message step with purpose 'clarification'.\n"
        "- If you cannot finish, send a message step with purpose 'cannot_finish' (and you may end).\n"
        "- Otherwise, use tool_call steps to add/complete/delete tasks until the list is complete."
        f"{strict_next}"
    )
    # Same Qwen3 caveat: send as a user-role instruction with a clear prefix.
    _remove_prefixed_messages(messages, GOVERNANCE_NOTICE_PREFIX)
    messages.append({"role": "user", "content": f"{GOVERNANCE_NOTICE_PREFIX}\n{LOOP_PREFIX} {content}"})


def _truncate_preview(text: str, max_lines: int = 4, max_chars: int = 400) -> str:
    return truncate_text(text, max_lines=max_lines, max_chars=max_chars)

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
    task_manager: TaskListManager,
    show_stats: bool,
    debug_logger: Optional[DebugLogger] = None,
    runtime_tools: Optional[Dict[str, RuntimeTool]] = None,
    runtime_context: Optional[RuntimeContext] = None,
) -> Tuple[int, Optional[LLMCallStats], bool]:
    """Run one model turn, executing tools until the model stops requesting them."""
    empty_replies = 0
    last_tool_summary: Optional[str] = None
    last_stats: Optional[LLMCallStats] = None
    current_turn = turn
    invalid_replies = 0
    gate_violations = 0
    think_only_turns = 0
    pending_withheld_message: Optional[str] = None
    need_user_update = False

    def indent(text: str) -> str:
        return "\n".join(f"    {line}" for line in text.splitlines())

    while True:
        _upsert_governance_notice(messages, task_manager, gate_violations)
        result = client.call(messages)
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
            rendered = task_manager.render_tasks()
            next_json = (
                '{"type":"assistant_turn","version":"1","steps":['
                '{"type":"add_task","args":{"task":"list files in the repo (ls .)"}},'
                '{"type":"add_task","args":{"task":"inspect three different files (read ...)"}},'
                '{"type":"add_task","args":{"task":"summarize findings and report"}},'
                '{"type":"list_tasks"}'
                "]}"
            )
            _upsert_action_required(
                messages,
                (
                    "Your last reply was empty (or whitespace). This is not allowed.\n"
                    f"The user asked: {last_user!r}.\n"
                    "Return ONLY a single JSON object matching the assistant protocol.\n"
                    "Do NOT return whitespace.\n\n"
                    f"Current task list:\n{rendered}\n\n"
                    "If you need to proceed, add tasks and list them (example JSON to copy):\n"
                    f"{next_json}"
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
                        '{"type":"assistant_turn","version":"1","steps":[...]}\n'
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

        if debug_logger:
            step_summary = ",".join(step.type for step in turn_obj.steps) if turn_obj.steps else "(no steps)"
            print(f"{COLOR_DIM}[protocol] ok steps={step_summary}{COLOR_RESET}")

        print(f"\n{label_color}[assistant #{current_turn}]{COLOR_RESET}")
        if thinks:
            think_preview = _truncate_preview("\n\n".join(step.content for step in thinks), max_lines=3, max_chars=240)
            if think_preview:
                print(f"{COLOR_DIM}(think preview)\n{think_preview}{COLOR_RESET}")
        if user_messages and should_render_user_messages(task_manager, user_messages):
            combined = "\n\n".join(step.content for step in user_messages)
            print(f"{combined}\n")
            # Once we have shown any message to the user, clear any previously withheld message requirement.
            # (The model may rephrase; we mainly want to avoid it thinking the user saw withheld output.)
            pending_withheld_message = None
            need_user_update = False
        elif tool_call_payloads:
            tools = ", ".join(call.tool for call in tool_call_payloads)
            print(f"{COLOR_DIM}(requesting tool(s): {tools}){COLOR_RESET}\n")
        elif user_messages:
            # Withhold user-visible content until tasks are complete (unless clarification/cannot_finish).
            combined = "\n\n".join(step.content for step in user_messages)
            pending_withheld_message = combined
            need_user_update = True
            preview = _truncate_preview(combined, max_lines=6, max_chars=600)
            print(f"{COLOR_DIM}(withheld message until task list is complete; preview)\n{preview}{COLOR_RESET}\n")
        else:
            print()

        # Preserve protocol conversation history for the model by storing the JSON turn verbatim.
        messages.append({"role": "assistant", "content": assistant_reply})

        if thinks and not user_messages and not tool_call_payloads and not has_end:
            think_only_turns += 1
            rendered = task_manager.render_tasks()
            next_json = (
                '{"type":"assistant_turn","version":"1","steps":['
                '{"type":"add_task","args":{"task":"list files in the repo (ls .)"}},'
                '{"type":"add_task","args":{"task":"inspect three different files (read ...)"}},'
                '{"type":"add_task","args":{"task":"summarize findings and report"}},'
                '{"type":"list_tasks"}'
                "]}"
            )
            if think_only_turns > 3:
                _upsert_action_required(
                    messages,
                    (
                        "You have emitted multiple think-only turns. This is not allowed.\n"
                        "Next, emit either tool_call steps or message/end steps.\n\n"
                        f"Current task list:\n{rendered}\n\n"
                        "Return ONLY a single JSON object matching the assistant protocol.\n"
                        "If you are unsure what to do next, add tasks and list them (example JSON to copy):\n"
                        f"{next_json}"
                    ),
                )
            else:
                _upsert_action_required(
                    messages,
                    (
                        "You produced only think steps. Continue immediately with the next action.\n\n"
                        f"Current task list:\n{rendered}\n\n"
                        "Return ONLY a single JSON object matching the assistant protocol.\n"
                        "Next, add tasks and list them (example JSON to copy):\n"
                        f"{next_json}"
                    ),
                )
            current_turn += 1
            continue

        if tool_call_payloads:
            if len(tool_call_payloads) > MAX_TOOL_CALLS_PER_TURN:
                calls_preview = ", ".join(call.tool for call in tool_call_payloads[:MAX_TOOL_CALLS_PER_TURN])
                _upsert_action_required(
                    messages,
                    (
                        f"Too many tool calls in one reply ({len(tool_call_payloads)}). Limit is {MAX_TOOL_CALLS_PER_TURN}.\n"
                        "Split the work across multiple assistant_turn replies.\n"
                        f"First {MAX_TOOL_CALLS_PER_TURN} tools requested: {calls_preview}\n"
                        "Return ONLY a single JSON object matching the assistant protocol."
                    ),
                )
                current_turn += 1
                continue

            destructive = 0
            for call in tool_call_payloads:
                plugin = plugin_tools.get(call.tool)
                if plugin is None:
                    destructive += 1
                elif plugin.is_destructive:
                    destructive += 1
            if destructive > MAX_DESTRUCTIVE_TOOL_CALLS_PER_TURN:
                tools_list = ", ".join(call.tool for call in tool_call_payloads)
                _upsert_action_required(
                    messages,
                    (
                        f"Too many destructive tool calls in one reply ({destructive}). Limit is {MAX_DESTRUCTIVE_TOOL_CALLS_PER_TURN}.\n"
                        "Split destructive actions across multiple replies.\n"
                        f"Tools requested: {tools_list}\n"
                        "Return ONLY a single JSON object matching the assistant protocol."
                    ),
                )
                current_turn += 1
                continue

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
                    task_manager=task_manager,
                    debug_logger=debug_logger,
                )
                tool_desc = f"tool '{tool_call.tool}' on '{tool_call.target}'"
                last_tool_summary = summarize_output(output, max_lines=max_tool_output[0], max_chars=max_tool_output[1])
                if max_tool_output[1] > 0:
                    tool_result_body = indent(f"[tool result] {tool_desc}\n{last_tool_summary}")
                    print(f"{COLOR_DIM}{tool_result_body}{COLOR_RESET}\n")

                if debug_logger:
                    debug_logger.log(
                        "tool.output",
                        f"turn={current_turn} tool={tool_call.tool} target={tool_call.target!r} args={tool_call.args!r} output={output}",
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool result for {tool_desc}:\n{output}\n"
                            f"Use this to continue helping the user (request: {last_user!r}). "
                            "If another tool is needed, call it; otherwise reply to the user now."
                        ),
                    }
                )
                # The user hasn't necessarily seen the implications of this tool result yet.
                need_user_update = True
                current_turn += 1
            continue

        if has_end:
            if pending_withheld_message is not None or need_user_update:
                rendered = task_manager.render_tasks()
                preview = _truncate_preview(pending_withheld_message or "", max_lines=12, max_chars=900)
                extra = ""
                if pending_withheld_message:
                    extra = f"\n\nWithheld message preview (resend or rephrase):\n{preview}"
                _upsert_action_required(
                    messages,
                    (
                        "You emitted an end step, but the human user has not received a final visible summary of what happened.\n"
                        "Do NOT end yet. Send a message step now (prefer purpose='final') summarizing results, then end.\n\n"
                        f"Current task list:\n{rendered}"
                        f"{extra}"
                    ),
                )
                current_turn += 1
                continue
            if not can_end_conversation(task_manager, user_messages):
                gate_violations += 1
                current_turn += 1
                continue
            return current_turn + 1, last_stats, True

        if user_messages and has_incomplete_tasks(task_manager) and not should_render_user_messages(task_manager, user_messages):
            combined = "\n\n".join(step.content for step in user_messages)
            preview = _truncate_preview(combined, max_lines=12, max_chars=900)
            rendered = task_manager.render_tasks()
            _upsert_action_required(
                messages,
                (
                    "Your previous message was NOT shown to the human user because the task list is incomplete.\n"
                    "Do not assume the user saw it. First, complete the outstanding tasks using tool calls.\n"
                    "After the task list is complete, resend the message to the user (prefer purpose='final').\n\n"
                    f"Current task list:\n{rendered}\n\n"
                    "Withheld message preview (for you to resend later):\n"
                    f"{preview}"
                ),
            )
            gate_violations += 1
            current_turn += 1
            continue

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
    multiline: bool = False,
    plugin_dirs: Optional[Sequence[Path]] = None,
    debug_logger: Optional[DebugLogger] = None,
) -> None:
    task_manager = TaskListManager()
    base = workdir.resolve()
    if debug_logger:
        debug_logger.log("debug.enabled", f"writing debug logs to {debug_logger.path}")
        debug_logger.log("context", f"workdir={base} yolo_enabled={yolo_enabled} read_only={read_only}")

    notes = gather_context(base)
    # Ensure an active task list exists (may be empty).
    task_manager.new_list()

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
    initial_task_list_text = task_manager.render_tasks()
    messages: List[Dict[str, str]] = [
        build_system_message(
            base,
            notes,
            initial_task_list=initial_task_list_text,
            read_only=read_only,
            allowed_tools=allowed_tools,
            plugins=list(plugins.values()),
            runtime_tools=list(runtime_tools.values()),
        )
    ]
    user_input = initial_prompt
    turn = 1
    runtime_ctx = RuntimeContext(
        client=client,
        plugin_tools=plugins,
        base=base,
        extra_roots=extra_roots,
        skill_roots=skill_roots,
        yolo_enabled=yolo_enabled,
        read_only=read_only,
        task_manager=task_manager,
        debug_logger=debug_logger,
    )

    if debug_logger and initial_prompt is not None:
        debug_logger.log("user.initial_prompt", initial_prompt)

    if user_input is None:
        result = read_user_prompt(user_prompt, multiline_default=multiline)
        if result.eof:
            return
        user_input = result.text or ""
        if debug_logger:
            debug_logger.log("user.input", f"turn={turn} content={user_input}")

    while True:
        if max_turns is not None and turn > max_turns:
            print(f"Reached max turns ({max_turns}); stopping.")
            if debug_logger:
                debug_logger.log("loop.stop", f"reason=max_turns reached={max_turns}")
            return
        if not user_input:
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

        messages.append({"role": "user", "content": user_input})
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
            task_manager=task_manager,
            show_stats=show_stats,
            debug_logger=debug_logger,
        )
        turn = next_turn
        if ended:
            return
        user_prompt = format_user_prompt()

        try:
            user_input = input(user_prompt).strip()
        except EOFError:
            return
        if debug_logger and user_input is not None:
            debug_logger.log("user.input", f"turn={turn} content={user_input}")
