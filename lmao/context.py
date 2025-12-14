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

GENERAL_INTRO_PROMPT = """You are running inside an agentic loop with tools. Act autonomously: plan, use tools, and keep going until the user's request is done.

Task list: manage it with the task tools (seed with "create a plan to respond"). Keep items concise and up to date before replying; if tasks remain and you're not blocked, keep working instead of replying.

Tooling: one tool call at a time; payloads go in 'args' (use 'target' only for paths). Avoid tools for trivial Q&A, and prefer non-destructive tools when possible. Validate assumptions with tools when you can; if a tool fails, adjust and retry once before asking the user.

Flow: after updating the task list or running a tool, continue without pausing for confirmation. Mid-work replies should be brief: status plus next action or what you need.

General Q&A should be answered directly without calling tools unless needed.
  """

SKILL_USAGE_PROMPT = """Skills live under ./skills/<name>/SKILL.md and may also exist under ~/.config/agents/skills/<name>/SKILL.md.
- When asked to use a skill, call list_skills if you need the paths, then read that SKILL.md and follow its instructions/examples exactly; do not guess.
- Do not enumerate or read skills unless the user asks about skills or requests one.
- If you cannot apply the skill after reading, ask one concise clarifying question."""


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
        f"Available tools (including plugins): {tool_list}.\n"
        "Tool outputs are JSON with 'success' plus structured 'data' or 'error'; use the JSON directly (paths may contain spaces).\n"
        "When you need to use a tool, respond ONLY with a single JSON object (no surrounding text) shaped like:\n"
        f"{examples_block}\n"
        "Paths are relative to the working directory; do not escape with .. or absolute paths. Quote paths with spaces. Issue one tool call at a time; only the first JSON block will run. Do not stay silent or return empty replies. If asked about available tools, answer from the list above without running tools."
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
        f"{tool_prompt}\n\n{SKILL_USAGE_PROMPT}\n"
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
