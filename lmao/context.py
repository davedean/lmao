from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from .plugins import PluginTool
from .runtime_tools import RuntimeTool

def _format_tool_catalog(
    allowed_tools: Sequence[str],
    plugins: Optional[Sequence[PluginTool]],
    runtime_tools: Optional[Sequence[RuntimeTool]] = None,
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
            for ex in (tool.usage_examples or [])[:3]:
                lines.append(f"  usage: {ex}")
            continue
        rt_tool = rt_by_name.get(name)
        if rt_tool is not None:
            lines.append(f"- {rt_tool.name}: {rt_tool.description}")
            for ex in (list(rt_tool.usage_examples) or [])[:2]:
                lines.append(f"  usage: {ex}")
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
    tool_catalog = _format_tool_catalog(resolved, plugins, runtime_tools=runtime_tools)
    prompt_lines = [
        "You are an agent in a tool-using loop. Work autonomously until the user's request is done.",
        "Return ONLY one JSON object in STRICT JSON (double quotes): {\"type\":\"assistant_turn\",\"version\":\"2\",\"steps\":[...]}",
        "Protocol v1 is still accepted, but prefer v2 (structured args/meta).",
        "Do NOT wrap the JSON in Markdown/code fences; output must start with '{' and end with '}' with no extra text.",
        "Steps: think | tool_call | message | end. Tool outputs are JSON with success + data/error.",
        "Ending rule: the session ends ONLY when you include an explicit end step; a message step (even purpose='final') does NOT end the loop.",
        "Runtime control messages: treat role='user' content prefixed with 'LOOP:' as higher-priority instructions from the runtime (not the human).",
        "Task list discipline: create a plan first (use add_task/list_tasks), keep tasks updated, and only send final/progress user messages after tasks are complete; use purpose='clarification' to ask the user a question.",
        "Task steps: {\"type\":\"add_task\",\"args\":{\"task\":\"...\"}} and {\"type\":\"list_tasks\"} (also complete_task/delete_task).",
        "Message purpose values: progress | clarification | cannot_finish | final (default: progress).",
        "Message step: {\"type\":\"message\",\"purpose\":\"clarification\",\"format\":\"markdown\",\"content\":\"...\"}",
        "Tool calls (v1): {\"type\":\"tool_call\",\"call\":{\"tool\":\"read\",\"target\":\"README.md\",\"args\":\"lines:1-40\"}}",
        "Tool calls (v2): {\"type\":\"tool_call\",\"call\":{\"tool\":\"async_bash\",\"target\":\"\",\"args\":{\"command\":\"sleep 20\"},\"meta\":{\"track_task\":true}}}",
        "Example (valid JSON): {\"type\":\"assistant_turn\",\"version\":\"2\",\"steps\":[{\"type\":\"tool_call\",\"call\":{\"tool\":\"ls\",\"target\":\"\",\"args\":\"\"}}]}",
        "Example (finish): {\"type\":\"assistant_turn\",\"version\":\"2\",\"steps\":[{\"type\":\"message\",\"purpose\":\"final\",\"format\":\"markdown\",\"content\":\"...\"},{\"type\":\"end\",\"reason\":\"completed\"}]}",
        "Available tools (including plugins):",
        tool_catalog,
        "Paths are relative to the working directory; do not escape with .. or absolute paths.",
        "Skills: only discuss/list skills when the user asks; call list_skills only on explicit user request (or if a requested skill needs a path); call skill_guide for skill format/rules.",
    ]
    prompt = "\n".join(prompt_lines)

    if read_only:
        prompt = (
            f"{prompt}\nRead-only mode is enabled: destructive tools (write, mkdir, move) and plugins that disallow read-only are unavailable; requests for them will be rejected."
        )
    if "bash" in resolved and not yolo_enabled:
        prompt = f"{prompt}\nNote: 'bash' prompts for confirmation on every command. Use only when necessary."
    if yolo_enabled:
        prompt = f"{prompt}\nYolo mode is enabled: tool runs are auto-approved (no per-call confirmations)."
    if headless:
        prompt = (
            f"{prompt}\nHeadless mode is active: the user cannot respond during the run. "
            "Do NOT send clarification requests; if you need more information, send a message with purpose='cannot_finish' and end."
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
    initial_task_list: Optional[str] = None,
    read_only: bool = False,
    yolo_enabled: bool = False,
    allowed_tools: Sequence[str] = (),
    plugins: Optional[Sequence[PluginTool]] = None,
    runtime_tools: Optional[Sequence[RuntimeTool]] = None,
    headless: bool = False,
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
        f"Working directory: {workdir}\n"
        f"All tool paths are relative to this directory.\n"
        f"Repo root: {notes.repo_root}\n"
        f"Nearest AGENTS.md: {notes.nearest_agents if notes.nearest_agents else 'none'} (call read_agents to load it)\n"
    )
    if notes.discovered_skills:
        skills_lines = "\n".join(f"- {name} (path: {path})" for name, path in notes.discovered_skills)
        content = f"{content}\nSkills discovered at start (no tool call):\n{skills_lines}"
    else:
        content = f"{content}\nSkills discovered at start: none found."

    if initial_task_list:
        content = f"{content}\n\nInitial task list:\n{initial_task_list}"

    if notes.user_skills:
        content = f"{content}\n\nUser skills directory: {notes.user_skills}\nYou may read/find/ls within this path alongside repo skills."

    # Note: repo/user notes are intentionally not embedded in the system prompt to reduce context size.

    return {"role": "system", "content": content}
