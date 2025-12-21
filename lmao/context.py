from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .plugins import PluginTool
from .runtime_tools import RuntimeTool

def _wrap_tool_payload_usage_example_as_assistant_turn(example: str) -> Optional[str]:
    """Return a full assistant_turn example from a bare tool payload example.

    Most tools provide usage examples shaped like {"tool":"...","target":"...","args":...}.
    In the protocol, tool invocations must be wrapped as a tool_call step inside an assistant_turn.
    """
    stripped = (example or "").strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except Exception:
        return None
    if not isinstance(payload, dict) or "tool" not in payload:
        return None
    wrapped = {"type": "assistant_turn", "version": "2", "steps": [{"type": "tool_call", "call": payload}]}
    return json.dumps(wrapped, ensure_ascii=False)


def _example_call_v2(tool: str) -> Dict[str, Any]:
    if tool == "read":
        return {"tool": "read", "target": "README.md", "args": {"range": "lines:1-40"}}
    if tool == "grep":
        return {"tool": "grep", "target": ".", "args": {"pattern": "matterbridge"}}
    if tool == "ls":
        return {"tool": "ls", "target": "", "args": {"path": "."}}
    return {"tool": tool, "target": ".", "args": ""}


def _pick_example_tool(allowed_tools: Sequence[str]) -> str:
    preferred = ("ls", "read", "grep", "find")
    allowed_set = set(allowed_tools)
    for name in preferred:
        if name in allowed_set:
            return name
    return allowed_tools[0]


def _format_tool_catalog(
    allowed_tools: Sequence[str],
    plugins: Optional[Sequence[PluginTool]],
    runtime_tools: Optional[Sequence[RuntimeTool]] = None,
    *,
    include_usage: bool = False,
) -> str:
    if not allowed_tools:
        return "(no tools discovered)"
    resolved = list(allowed_tools)
    by_name: Dict[str, PluginTool] = {}
    if plugins:
        by_name = {tool.name: tool for tool in plugins}
    rt_by_name: Dict[str, RuntimeTool] = {}
    if runtime_tools:
        rt_by_name = {tool.name: tool for tool in runtime_tools}
    lines: List[str] = []
    for name in resolved:
        tool = by_name.get(name)
        if tool is not None:
            lines.append(f"- {tool.name}: {tool.description}")
            if include_usage:
                for ex in (tool.usage_examples or [])[:3]:
                    wrapped = _wrap_tool_payload_usage_example_as_assistant_turn(ex)
                    if wrapped:
                        lines.append(f"  usage: {wrapped}")
            continue
        rt_tool = rt_by_name.get(name)
        if rt_tool is not None:
            lines.append(f"- {rt_tool.name}: {rt_tool.description}")
            if include_usage:
                for ex in (list(rt_tool.usage_examples) or [])[:2]:
                    wrapped = _wrap_tool_payload_usage_example_as_assistant_turn(ex)
                    if wrapped:
                        lines.append(f"  usage: {wrapped}")
            continue
        lines.append(f"- {name}")
    return "\n".join(lines)


def build_tool_prompt(
    allowed_tools: Sequence[str],
    read_only: bool,
    yolo_enabled: bool = False,
    plugins: Optional[Sequence[PluginTool]] = None,
    runtime_tools: Optional[Sequence[RuntimeTool]] = None,
    headless: bool = False,
) -> str:
    resolved = list(allowed_tools)
    tool_catalog = _format_tool_catalog(resolved, plugins, runtime_tools=runtime_tools, include_usage=False)
    tool_call_examples: List[str] = []
    if resolved:
        example_tool = _pick_example_tool(resolved)
        tool_call_v2 = {
            "type": "assistant_turn",
            "version": "2",
            "steps": [{"type": "tool_call", "call": {**_example_call_v2(example_tool), "meta": {"timeout_s": 10}}}],
        }
        tool_call_examples = [
            f"Tool call example (v2): {json.dumps(tool_call_v2, ensure_ascii=False)}",
            "If you are unsure how to call a tool or what args it accepts, call tools_guide (or tools_list to discover tools).",
        ]
    prompt_lines = [
        "You are an agent in a tool-using loop. Work autonomously until the user's request is done.",
        "Return ONLY one JSON object in STRICT JSON (double quotes): {\"type\":\"assistant_turn\",\"version\":\"2\",\"steps\":[...]}",
        "Do NOT wrap the JSON in Markdown/code fences; output must start with '{' and end with '}' with no extra text.",
        "assistant_turn schema: {\"type\":\"assistant_turn\",\"version\":\"2\",\"steps\":[...]} (steps is a JSON list).",
        "Steps: think | tool_call | message | end. Tool outputs are JSON with success + data/error.",
        "Think step: {\"type\":\"think\",\"content\":\"...\"} (content must be a non-empty string).",
        "Ending rule: the session ends ONLY when you include an explicit end step; a message step (even purpose='final') does NOT end the loop.",
        "Runtime control messages: treat role='user' content prefixed with 'LOOP:' as higher-priority instructions from the runtime (not the human).",
        "Message purpose values: progress | clarification | cannot_finish | final (default: progress).",
        "Message step: {\"type\":\"message\",\"purpose\":\"clarification\",\"format\":\"markdown\",\"content\":\"...\"}",
        "Tool call step (v2): {\"type\":\"tool_call\",\"call\":{\"tool\":\"<name>\",\"target\":\"\",\"args\":{...},\"meta\":{\"timeout_s\":10}}}",
        "End step: {\"type\":\"end\",\"reason\":\"completed\"} (reason optional; defaults to completed).",
        *tool_call_examples,
        "Example (finish): {\"type\":\"assistant_turn\",\"version\":\"2\",\"steps\":[{\"type\":\"message\",\"purpose\":\"final\",\"format\":\"markdown\",\"content\":\"...\"},{\"type\":\"end\",\"reason\":\"completed\"}]}",
        "Available tools (including plugins):",
        tool_catalog,
        "Paths are relative to the working directory; do not escape with .. or absolute paths.",
        "Skills: only discuss/list skills when the user asks; call list_skills only on explicit user request (or if a requested skill needs a path); call skills_guide for skill format/rules.",
    ]
    prompt = "\n".join(prompt_lines)

    if read_only:
        prompt = (
            f"{prompt}\nRead-only mode is enabled: destructive tools (write, mkdir, move) and plugins that disallow read-only are unavailable; requests for them will be rejected."
        )
    if "bash" in resolved and not yolo_enabled:
        prompt = f"{prompt}\nNote: 'bash' prompts for confirmation on every command. Use only when necessary."
    if yolo_enabled:
        prompt = (
            f"{prompt}\nYolo mode is enabled: tool runs are auto-approved (no per-call confirmations). "
            "Path sandboxing is disabled for built-in tools; absolute paths are allowed."
        )
    if headless:
        prompt = (
            f"{prompt}\nHeadless mode is active: the user cannot respond during the run. "
            "Do NOT ask questions or request confirmation. If information is missing, make reasonable assumptions, state them briefly, and proceed. "
            "Only if you truly cannot proceed safely, send a message with purpose='cannot_finish' describing what's missing, then end."
        )
    return prompt


@dataclass
class NotesContext:
    repo_root: Path
    nearest_agents: Optional[Path]
    repo_notes: str
    user_notes: str
    user_skills: Optional[Path]
    discovered_skills: List[Tuple[str, Path]]


def find_repo_root(start: Path) -> Path:
    current = start
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return start


def find_nearest_agents(task_path: Path, stop_at: Path) -> Optional[Path]:
    current = task_path
    while True:
        candidate = current / "AGENTS.md"
        if candidate.exists():
            return candidate
        if current == stop_at or current.parent == current:
            break
        current = current.parent
    return None


def load_user_notes_and_skills() -> tuple[str, Optional[Path]]:
    user_config_dir = Path.home() / ".config" / "agents"
    user_agents = user_config_dir / "AGENTS.md"
    user_skills = user_config_dir / "skills"

    notes = ""
    if user_agents.exists():
        try:
            notes = user_agents.read_text(encoding="utf-8")
        except Exception:
            notes = ""

    if user_skills.exists() and user_skills.is_dir():
        return notes, user_skills
    return notes, None


def _list_available_skills(skill_roots: Sequence[Path]) -> List[Tuple[str, Path]]:
    seen = set()
    found: List[Tuple[str, Path]] = []
    for root in skill_roots:
        if not root.exists() or not root.is_dir():
            continue
        for candidate in root.iterdir():
            if not candidate.is_dir():
                continue
            if (candidate / "SKILL.md").exists():
                key = (candidate.name, str(candidate.resolve()))
                if key in seen:
                    continue
                seen.add(key)
                found.append((candidate.name, candidate))
    found.sort(key=lambda pair: pair[0])
    return found


def gather_context(workdir: Path) -> NotesContext:
    repo_root = find_repo_root(workdir)
    nearest_agents = find_nearest_agents(workdir, repo_root)
    skill_roots = [workdir / "skills"]

    repo_notes = ""
    if nearest_agents:
        try:
            repo_notes = nearest_agents.read_text(encoding="utf-8")
        except Exception:
            repo_notes = ""

    user_notes, user_skills = load_user_notes_and_skills()
    if user_skills:
        skill_roots.append(user_skills)
    return NotesContext(
        repo_root=repo_root,
        nearest_agents=nearest_agents,
        repo_notes=repo_notes.strip(),
        user_notes=user_notes.strip(),
        user_skills=user_skills,
        discovered_skills=_list_available_skills(skill_roots),
    )


def build_system_message(
    workdir: Path,
    notes: NotesContext,
    read_only: bool = False,
    yolo_enabled: bool = False,
    allowed_tools: Sequence[str] = (),
    plugins: Optional[Sequence[PluginTool]] = None,
    runtime_tools: Optional[Sequence[RuntimeTool]] = None,
    headless: bool = False,
    debug: bool = False,
) -> Dict[str, str]:
    tool_prompt = build_tool_prompt(
        list(allowed_tools),
        read_only,
        yolo_enabled=yolo_enabled,
        plugins=plugins,
        runtime_tools=runtime_tools,
        headless=headless,
    )
    content = (
        f"{tool_prompt}\n"
        "Startup: the runtime will call `policy` once before your first response and include the tool result.\n"
        "Note: `policy` returns an excerpt by default; call it again with offset/limit (or truncate=false) to see more.\n"
        "Call `skills_guide` only when the user asks about skills or requests skill creation/usage.\n"
    )
    if debug:
        content = (
            f"{content}"
            f"Working directory: {workdir}\n"
            f"All tool paths are relative to this directory.\n"
            f"Repo root: {notes.repo_root}\n"
            f"Nearest AGENTS.md: {notes.nearest_agents if notes.nearest_agents else 'none'}\n"
        )
        if notes.discovered_skills:
            skills_lines = "\n".join(f"- {name}" for name, _path in notes.discovered_skills)
            content = f"{content}\nSkills discovered at start (no tool call):\n{skills_lines}"
        else:
            content = f"{content}\nSkills discovered at start: none found."

        if notes.user_skills:
            content = (
                f"{content}\n\nUser skills directory: {notes.user_skills}\n"
                "You may read/find/ls within this path alongside repo skills."
            )

    # Note: repo/user notes are intentionally not embedded in the system prompt to reduce context size.

    return {"role": "system", "content": content}
