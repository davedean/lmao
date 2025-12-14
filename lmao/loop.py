from __future__ import annotations

import sys
from pathlib import Path
import json
from typing import Dict, List, Optional, Sequence, Tuple

from .debug_log import DebugLogger
from .context import build_system_message, gather_context
from .llm import LLMClient
from .task_list import TaskListManager
from .tools import ToolCall, get_allowed_tools, parse_tool_calls, run_tool, summarize_output
from .plugins import PluginTool, discover_plugins

COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"


def run_agent_turn(
    messages: List[Dict[str, str]],
    client: LLMClient,
    turn: int,
    last_user: str,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    git_allowed: bool,
    max_tool_output: Tuple[int, int],
    yolo_enabled: bool,
    read_only: bool,
    allowed_tools: Sequence[str],
    plugin_tools: Dict[str, PluginTool],
    task_manager: TaskListManager,
    debug_logger: Optional[DebugLogger] = None,
) -> None:
    """Run one model turn, executing tools until the model stops requesting them."""
    empty_replies = 0
    last_tool_summary: Optional[str] = None

    def indent(text: str) -> str:
        return "\n".join(f"    {line}" for line in text.splitlines())

    while True:
        assistant_reply = client.call(messages)
        if not assistant_reply or not assistant_reply.strip():
            empty_replies += 1
            if empty_replies >= 2:
                fallback = last_tool_summary or "No tool output available."
                assistant_reply = (
                    f"(auto-generated fallback) Unable to get a response from the model. "
                    f"Based on the latest tool output, here is a summary:\n{fallback}"
                )
                print(f"\n{COLOR_BLUE}[assistant #{turn}]{COLOR_RESET}\n{assistant_reply}\n")
                messages.append({"role": "assistant", "content": assistant_reply})
                return
            messages.append({"role": "system", "content": f"Your last reply was empty. The user asked: {last_user!r}. Continue the conversation now using the latest tool results and reply to the user."})
            continue

        if debug_logger:
            debug_logger.log("assistant.reply", f"turn={turn} content={assistant_reply}")
        empty_replies = 0
        tool_calls = parse_tool_calls(assistant_reply, allowed_tools=allowed_tools)
        tool_call = tool_calls[0] if tool_calls else None
        is_tool_phase = tool_call is not None
        label_color = COLOR_BLUE
        reply_body = indent(assistant_reply) if is_tool_phase else assistant_reply
        body_color_start = COLOR_DIM if is_tool_phase else ""
        body_color_end = COLOR_RESET if is_tool_phase else ""

        print(f"\n{label_color}[assistant #{turn}]{COLOR_RESET}\n{body_color_start}{reply_body}{body_color_end}\n")
        messages.append({"role": "assistant", "content": assistant_reply})

        if not tool_call:
            return

        if debug_logger:
            debug_logger.log(
                "tool.call",
                f"turn={turn} tool={tool_call.tool} target={tool_call.target!r} args={tool_call.args!r}",
            )
        output = run_tool(
            tool_call,
            base,
            extra_roots,
            skill_roots,
            git_allowed,
            yolo_enabled,
            read_only=read_only,
            plugin_tools=plugin_tools,
            task_manager=task_manager,
            debug_logger=debug_logger,
        )
        if len(tool_calls) > 1:
            note = "multiple tool calls were detected; only the first was executed. Issue tools one at a time."
            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    parsed["note"] = note
                    output = json.dumps(parsed, ensure_ascii=False)
                else:
                    output = f"{output}\nNote: {note}"
            except Exception:
                output = f"{output}\nNote: {note}"
        tool_desc = f"tool '{tool_call.tool}' on '{tool_call.target}'"
        last_tool_summary = summarize_output(output, max_lines=max_tool_output[0], max_chars=max_tool_output[1])
        if max_tool_output[1] > 0:
            tool_result_body = indent(f"[tool result] {tool_desc}\n{last_tool_summary}")
            print(f"{COLOR_DIM}{tool_result_body}{COLOR_RESET}\n")

        if debug_logger:
            debug_logger.log(
                "tool.output",
                f"turn={turn} tool={tool_call.tool} target={tool_call.target!r} args={tool_call.args!r} output={output}",
            )
        messages.append({
            "role": "user",
            "content": (
                f"Tool result for {tool_desc}:\n{output}\n"
                f"Use this to continue helping the user (request: {last_user!r}). "
                "If another tool is needed, call it; otherwise reply to the user now."
            ),
        })
        turn += 1


def run_loop(
    initial_prompt: Optional[str],
    client: LLMClient,
    workdir: Path,
    git_allowed: bool,
    max_tool_output: Tuple[int, int],
    max_turns: Optional[int],
    silent_tools: bool,
    yolo_enabled: bool,
    read_only: bool,
    plugin_dirs: Optional[Sequence[Path]] = None,
    debug_logger: Optional[DebugLogger] = None,
) -> None:
    task_manager = TaskListManager()
    base = workdir.resolve()
    if debug_logger:
        debug_logger.log("debug.enabled", f"writing debug logs to {debug_logger.path}")
        debug_logger.log("context", f"workdir={base} git_allowed={git_allowed} yolo_enabled={yolo_enabled} read_only={read_only}")

    notes = gather_context(base)
    # Seed a default task list so the model starts with a plan step (task list always exists)
    task_manager.new_list(seed_plan=True)

    extra_roots: List[Path] = []
    skill_roots: List[Path] = [base / "skills"]
    if notes.user_skills:
        resolved_user_skills = notes.user_skills.resolve()
        extra_roots.append(resolved_user_skills)
        skill_roots.append(resolved_user_skills)

    plugins = discover_plugins(plugin_dirs or [], base, debug_logger=debug_logger)
    if debug_logger and plugins:
        debug_logger.log("plugins.loaded", f"{[(name, str(plugin.path)) for name, plugin in plugins.items()]}")
    allowed_tools = get_allowed_tools(read_only=read_only, git_allowed=git_allowed, yolo_enabled=yolo_enabled, plugins=list(plugins.values()))
    initial_task_list_text = task_manager.render_tasks()
    messages: List[Dict[str, str]] = [build_system_message(base, git_allowed, yolo_enabled, notes, initial_task_list=initial_task_list_text, read_only=read_only, allowed_tools=allowed_tools, plugins=list(plugins.values()))]
    user_input = initial_prompt
    turn = 1

    if debug_logger and initial_prompt is not None:
        debug_logger.log("user.initial_prompt", initial_prompt)

    if user_input is None:
        try:
            user_input = input(f"{COLOR_GREEN}You:{COLOR_RESET} ").strip()
        except EOFError:
            return
        if debug_logger:
            debug_logger.log("user.input", f"turn={turn} content={user_input}")

    while True:
        if max_turns is not None and turn > max_turns:
            print(f"Reached max turns ({max_turns}); stopping.")
            if debug_logger:
                debug_logger.log("loop.stop", f"reason=max_turns reached={max_turns}")
            return
        if not user_input:
            try:
                user_input = input(f"{COLOR_GREEN}You:{COLOR_RESET} ").strip()
            except EOFError:
                return
            if debug_logger:
                debug_logger.log("user.input", f"turn={turn} content={user_input}")
            continue

        messages.append({"role": "user", "content": user_input})
        if debug_logger:
            debug_logger.log("user.input", f"turn={turn} content={user_input}")
        run_agent_turn(
            messages,
            client=client,
            turn=turn,
            last_user=user_input,
            base=base,
            extra_roots=extra_roots,
            skill_roots=skill_roots,
            git_allowed=git_allowed,
            max_tool_output=max_tool_output if not silent_tools else (0, 0),
            yolo_enabled=yolo_enabled,
            read_only=read_only,
            allowed_tools=allowed_tools,
            plugin_tools=plugins,
            task_manager=task_manager,
            debug_logger=debug_logger,
        )
        turn += 1

        try:
            user_input = input(f"{COLOR_GREEN}You:{COLOR_RESET} ").strip()
        except EOFError:
            return
        if debug_logger and user_input is not None:
            debug_logger.log("user.input", f"turn={turn} content={user_input}")
