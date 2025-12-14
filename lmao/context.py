from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from .plugins import PluginTool

SKILL_GUIDE = """Skills live under ./skills. Each Skill must be in its own folder: ./skills/<skill-name>

The entrypoint for a skill is a SKILL.md in the skill folder. Supporting files can live alongside SKILL.md within the skill folder.

SKILL.md has YAML frontmatter:
```
---
name: your-skill-name
description: brief description of what this Skill does and when to use it
---

## Instructions:
Instructions for the skill
```

There should be sections like '## Instructions' (step-by-step guidance) and '## Examples' (concrete usage).
Constraints:
- name: <=64 chars, lowercase letters/numbers/hyphens only; no XML tags; not 'anthropic' or 'claude'
- description: non-empty, <=1024 chars, no XML tags; include what the Skill does and when to use it."""

# When not provided, the allowed tool list should come from discovered plugins;
# this fallback remains empty to avoid drifting from runtime discovery.
DEFAULT_ALLOWED_TOOLS: list[str] = []

GENERAL_INTRO_PROMPT = """You are running inside an agentic loop that can repeatedly call tools and skills to complete the user's request. Act autonomously: plan, use tools, and work through your task list until the job is done.
- ALWAYS maintain a task list (create one if missing) with an initial item like "create a plan to respond". ALWAYS manage a task list by using the task list tools.
- Use one tool call at a time; do not batch multiple JSON blocks.
- Put payloads in 'args' (leave 'target' empty unless it's a path); quote paths with spaces.
- Use skills by reading their SKILL.md and following instructions; do not guess.
- Avoid tools for trivial Q&A; prefer non-destructive tools when possible.
- After updating the task list or running a tool, keep working—do not pause for confirmation unless blocked.
- If a tool fails, adjust and retry once before asking the user.
- Mid-work replies should be brief: status + next action or what you need.

Always use the task list tool to manage the task list before starting work. Add tasks you are intending to do onto the task list (one task list item per task).

Items on the task list should be kept up-to-date in terms of completion, by updating them before responding to the user.

Items on the task list should be checked for incomplete tasks before responding to the user.

If there are incomplete tasks on the task list, you should complete them before responding to the user.

Always check that all tasks on the task list are complete before responding to the user.

- Always validate your beliefs with tools where possible, before responding to the user. For example, if you believe a folder does not exist or is empty, you should find/list it first to confirm. Once you've validated the facts you can repond to the user with more clarity.

  """


def build_tool_prompt(allowed_tools: Sequence[str], yolo_enabled: bool, read_only: bool, plugins: Optional[Sequence[PluginTool]] = None) -> str:
    resolved = list(allowed_tools) if allowed_tools else list(DEFAULT_ALLOWED_TOOLS)
    tool_list = ", ".join(resolved) if resolved else "(no tools discovered)"
    example_lines: list[str] = []
    if plugins:
        plugin_map = {plugin.name: plugin for plugin in plugins}
        for name in resolved:
            plugin = plugin_map.get(name)
            if plugin and plugin.usage_examples:
                example_lines.extend(plugin.usage_examples)
    examples_block = "\n".join(example_lines) if example_lines else "(no tool examples provided by plugins)"

    prompt = (
        "You are a local file-editing agent. Act autonomously: plan and use tools as needed until the user's request is completed. Do not wait for permission to run tools.\n"
        f"Available tools (including plugins): {tool_list}. Tool outputs are JSON with 'success' plus structured 'data' or 'error'; use the JSON directly (paths may contain spaces). Skills are separate: each lives under ./skills/<name>/ with a SKILL.md; user-specific skills may also live under ~/.config/agents/skills/<name>/SKILL.md. Do NOT use list_skills to search for tools—tools are only the ones listed here. To list skills, use the list_skills tool; AGENTS.md is not a skill registry.\n"
        "Task lists: a task list is always available. Immediately add tasks for the planned steps (at least \"create a plan to respond\"). Tasks should be single-line, concise, and unnumbered; IDs are assigned automatically and stored in the list. Before running tools each turn, sync the list (add/complete/delete). After updating the list, keep working—do not pause for confirmation. Before replying, check the list: if tasks remain and you are not blocked, keep working instead of replying. If blocked/awaiting user, give a brief status + ask. Use list_tasks only when you need to show the list; avoid spamming. For task tools, put the content in 'args' (leave 'target' empty); if you accidentally put data in 'target', it will be treated as the payload.\n"
        "Skills: when asked to use a skill, (1) call list_skills if you have not already to see available skills; (2) read the skill's SKILL.md using its folder path from list_skills (e.g., /path/to/skills/foo/SKILL.md); (3) apply the instructions/examples to produce the requested output. If you cannot apply the skill after reading SKILL.md, ask one concise clarifying question.\n"
        "General questions (stories, planning, “which tools are available?”) should be answered directly without calling tools. Do NOT enumerate or read skills at session start unless the user asks about skills or to use a specific skill. When asked which tools are available, answer from the tool list above and do NOT call tools. Bash always asks for confirmation per command; use it only when necessary.\n"
        "When you need to use tools, respond ONLY with a JSON object (no surrounding text). Use one of:\n"
        f"{examples_block}\n"
        "Rules: paths are relative to the working directory; do not escape with .. or absolute paths. Quote paths with spaces when issuing tools. Issue one tool call at a time (multiple tool JSON blocks will only run the first). After a tool call, immediately continue: if more tools are needed, call them; otherwise reply directly to the user. Do not stay silent or return empty replies. The user cannot see tool output. Keep outputs concise. If asked about available tools, answer directly from the list above."
    )

    if read_only:
        prompt = (
            f"{prompt}\nRead-only mode is enabled: destructive tools (write, mkdir, move) and plugins that disallow read-only are unavailable; requests for them will be rejected."
        )
    if "bash" in resolved:
        prompt = f"{prompt}\nNote: 'bash' prompts for confirmation on every command. Use only when necessary."
    return prompt


@dataclass
class NotesContext:
    repo_root: Path
    nearest_agents: Optional[Path]
    repo_notes: str
    user_notes: str
    user_skills: Optional[Path]


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


def gather_context(workdir: Path) -> NotesContext:
    repo_root = find_repo_root(workdir)
    nearest_agents = find_nearest_agents(workdir, repo_root)

    repo_notes = ""
    if nearest_agents:
        try:
            repo_notes = nearest_agents.read_text(encoding="utf-8")
        except Exception:
            repo_notes = ""

    user_notes, user_skills = load_user_notes_and_skills()
    return NotesContext(
        repo_root=repo_root,
        nearest_agents=nearest_agents,
        repo_notes=repo_notes.strip(),
        user_notes=user_notes.strip(),
        user_skills=user_skills,
    )


def build_system_message(workdir: Path, yolo_enabled: bool, notes: NotesContext, initial_task_list: Optional[str] = None, read_only: bool = False, allowed_tools: Optional[Sequence[str]] = None, plugins: Optional[Sequence[PluginTool]] = None) -> Dict[str, str]:
    resolved_allowed = list(allowed_tools) if allowed_tools is not None else list(DEFAULT_ALLOWED_TOOLS)
    tool_prompt = f"{GENERAL_INTRO_PROMPT}\n\n{build_tool_prompt(resolved_allowed, yolo_enabled, read_only, plugins=plugins)}"
    content = (
        f"{tool_prompt}\n"
        f"Working directory: {workdir}\n"
        f"All tool paths are relative to this directory.\n"
        f"AGENTS discovery: starting from the task path ({workdir}), walk up to the repo root ({notes.repo_root}) and use the nearest AGENTS.md found. "
        f"Nearest found: {notes.nearest_agents if notes.nearest_agents else 'none'}.\n"
        f"User-level AGENTS at ~/.config/agents/AGENTS.md are used when present; repo-nearest instructions take precedence when conflicting.\n"
        f"Read-only mode: {'enabled (destructive tools disabled)' if read_only else 'disabled'}.\n\n"
        f"Skill structure:\n{SKILL_GUIDE}"
    )

    if initial_task_list:
        content = f"{content}\n\nInitial task list:\n{initial_task_list}\nUpdate it before starting work."

    if notes.user_skills:
        content = f"{content}\n\nUser skills directory: {notes.user_skills}\nYou may read/find/ls within this path alongside repo skills."

    if plugins:
        plugin_lines = []
        for plugin in plugins:
            safety = "destructive" if plugin.is_destructive else "non-destructive"
            plugin_lines.append(f"- {plugin.name}: {plugin.description} ({safety} plugin)")
        content = f"{content}\n\nPlugins enabled in this run:\n" + "\n".join(plugin_lines)

    if notes.repo_notes:
        source = notes.nearest_agents if notes.nearest_agents else workdir / "AGENTS.md"
        content = f"{content}\n\nRepo notes (from {source}):\n{notes.repo_notes}"

    if notes.user_notes:
        content = f"{content}\n\nUser notes:\n{notes.user_notes}"

    return {"role": "system", "content": content}
