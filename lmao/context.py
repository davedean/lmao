from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from .plugins import PluginTool

SKILL_GUIDE = """Skills are instructions for agents, on how to accomplish tasks. They are not strictly "software" and should not be thought of as such.

Skills live under ./skills. Each Skill must be in its own folder: ./skills/<skill-name>

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

GENERAL_INTRO_PROMPT = """You are running inside an agentic loop with tools. Act autonomously: plan, use tools, and keep going until the user's request is done. You will be called multiple times, with history of previous calls, so you do not need to resolve the users request in one go. Its preferred that in the first step all you do is create a plan to complete the request. 

Operate by working step-by-step to complete the user's request.

To complete a request you should:
- Create a step-by-step plan to complete the request.
- Use the task tools to manage the plan by adding each step to the task list.
- If the task involves using a skill, include "reading the skill" as a step in the plan.
- After reading a skill, update the task list with any changes or updates to the plan.
- Use the tools to complete the request as required.
- Use the task list to manage the plan, keeping it up to date with your progress.
- When the task list is complete, respond to the user with the final result.

Tools are operational commands; skills are separate playbooks — do not treat skills as tools. If the user asks about tools, answer directly without calling any tool. Calling list_skills without an explicit user request for skills is a mistake.

Task list: manage it with the task tools. Keep items concise and up to date before replying; if tasks remain and you're not blocked, keep working instead of replying. 

Tooling: one tool call at a time; payloads go in 'args' (use 'target' only for paths). Avoid tools for trivial Q&A, and prefer non-destructive tools when possible. Validate assumptions with tools when you can; if a tool fails, adjust and retry once before asking the user.

Flow: after updating the task list or running a tool, continue without pausing for confirmation. Mid-work replies should be brief: status plus next action or what you need.

General Q&A should be answered directly without calling tools unless needed.
  """

SKILL_USAGE_PROMPT = """Skills live under ./skills/<name>/SKILL.md and may also exist under ~/.config/agents/skills/<name>/SKILL.md.
- Skills are not tools. Only talk about skills when the user asks about skills or requests one.
- list_skills must only be called if the user explicitly asks to list skills or requests a skill and you lack the path. Do NOT call list_skills to answer tool questions.
- When asked to use a skill, consult the prelisted skills below for paths; if you still need paths, call list_skills, then read that SKILL.md and follow its instructions/examples exactly—do not guess.
- Do not enumerate or read skills unless the user asks about skills or requests one; use the prelisted skills below and refresh with list_skills only if paths change or the user explicitly asks for a fresh list.
- If you cannot apply the skill after reading, ask one concise clarifying question."""


def build_tool_prompt(allowed_tools: Sequence[str], yolo_enabled: bool, read_only: bool, plugins: Optional[Sequence[PluginTool]] = None) -> str:
    resolved = list(allowed_tools) if allowed_tools else list(DEFAULT_ALLOWED_TOOLS)
    tool_list = ", ".join(resolved) if resolved else "(no tools discovered)"
    examples_block = "\n".join(
        [
            '{"type":"assistant_turn","version":"1","steps":[{"type":"message","format":"markdown","content":"...user-visible response..."},{"type":"end","reason":"completed"}]}',
            '{"type":"assistant_turn","version":"1","steps":[{"type":"tool_call","call":{"tool":"read","target":"README.md","args":"lines:1-40"}}]}',
            '{"type":"assistant_turn","version":"1","steps":[{"type":"message","purpose":"clarification","format":"markdown","content":"Which file should I edit?"}]}',
            '{"type":"assistant_turn","version":"1","steps":[{"type":"add_task","args":{"task":"read README.md"}},{"type":"add_task","args":{"task":"summarize CLI flags"}},{"type":"list_tasks"}]}',
        ]
    )

    prompt_lines = [
        f"Available tools (including plugins): {tool_list}.",
        "Tool outputs are JSON with 'success' plus structured 'data' or 'error'; use the JSON directly (paths may contain spaces).",
        "You MUST respond with a single JSON object and nothing else (no code fences, no surrounding text).",
        "Important: the runtime loop may send additional instructions as role='user' messages prefixed with 'LOOP:'. Treat these as higher-priority control messages from the agent runtime (not the human) and follow them exactly.",
        "Important: the runtime may WITHHOLD some of your message steps until the task list is complete. If you receive a LOOP message saying a message was withheld, the human did NOT see it; resend it after completing tasks.",
        "Important: do not emit an end step if you have any withheld message to resend; send the final user-visible message first, then end.",
        "Important: do not emit an end step immediately after tool calls unless you also send a final user-visible message summarizing the results.",
        "Required schema: {'type':'assistant_turn','version':'1','steps':[...]}",
        "Supported step types: think, tool_call, message, end.",
        "Tool calls must be emitted as a tool_call step with call={tool,target,args}. You may include multiple tool_call steps in one reply, but keep batches small.",
        "Task tool shorthand: you may emit steps with type add_task/complete_task/delete_task/list_tasks directly; they are treated as tool calls. You may include multiple task-tool steps in one reply.",
        "Messages: message steps support purpose={'progress','clarification','cannot_finish','final'} (default: progress).",
        "Governance: if the task list has incomplete items, the loop will withhold progress/final messages until tasks are completed. Use purpose='clarification' to ask the user something, or purpose='cannot_finish' to explain why you cannot finish.",
        "Use end when you believe the user request is complete (the loop may block end if the task list is not complete).",
        "Examples (copy the shape exactly):",
        f"{examples_block}",
        "Paths are relative to the working directory; do not escape with .. or absolute paths. Quote paths with spaces. Do not stay silent or return empty replies.",
        "If asked about available tools, answer directly from the list above without running any tool. Skills are already listed below—do not call list_skills unless the user asks about skills or you suspect the list changed and the user wants an update.",
        "Calling list_skills to answer tool questions is forbidden; reply directly instead.",
    ]
    prompt = "\n".join(prompt_lines)

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


def build_system_message(workdir: Path, yolo_enabled: bool, notes: NotesContext, initial_task_list: Optional[str] = None, read_only: bool = False, allowed_tools: Optional[Sequence[str]] = None, plugins: Optional[Sequence[PluginTool]] = None) -> Dict[str, str]:
    resolved_allowed = list(allowed_tools) if allowed_tools is not None else list(DEFAULT_ALLOWED_TOOLS)
    tool_prompt = f"{GENERAL_INTRO_PROMPT}\n\n{build_tool_prompt(resolved_allowed, yolo_enabled, read_only, plugins=plugins)}"
    content = (
        f"{tool_prompt}\n\n{SKILL_USAGE_PROMPT}\n"
        "Reminder: do NOT call any tool just to answer \"what tools are available\"—respond directly. Skills are listed below; do not refresh them unless the user asks.\n"
        f"Working directory: {workdir}\n"
        f"All tool paths are relative to this directory.\n"
        f"AGENTS discovery: starting from the task path ({workdir}), walk up to the repo root ({notes.repo_root}) and use the nearest AGENTS.md found. "
        f"Nearest found: {notes.nearest_agents if notes.nearest_agents else 'none'}.\n"
        "Repo/user AGENTS are not preloaded into this prompt to keep context small.\n"
        "If you need repo instructions to proceed, call the tool `read_agents`.\n"
        f"Read-only mode: {'enabled (destructive tools disabled)' if read_only else 'disabled'}.\n\n"
        f"Skill structure:\n{SKILL_GUIDE}"
    )
    if notes.discovered_skills:
        skills_lines = "\n".join(f"- {name} (path: {path})" for name, path in notes.discovered_skills)
        content = f"{content}\n\nSkills discovered at start (no tool call):\n{skills_lines}\nUse this list unless the user asks for an updated list."
    else:
        content = f"{content}\n\nSkills discovered at start: none found."

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

    # Note: repo/user notes are intentionally not embedded in the system prompt to reduce context size.

    return {"role": "system", "content": content}
