from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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

DEFAULT_TOOL_PROMPT = """You are a local file-editing agent. Act autonomously: plan and use tools as needed until the user's request is completed. Do not wait for permission to run tools.
Available tools (do not probe the filesystem to answer this): read, write, mkdir, move, ls, find, grep, list_skills, add_task, complete_task, delete_task, list_tasks, git_add, git_commit. Tool outputs are JSON with 'success' plus structured 'data' or 'error'; use the JSON directly (paths may contain spaces). Skills are separate: each lives under ./skills/<name>/ with a SKILL.md; user-specific skills may also live under ~/.config/agents/skills/<name>/SKILL.md. To list skills, use the list_skills tool; AGENTS.md is not a skill registry.
Task lists: a task list is always available. Immediately add tasks for the planned steps (at least "create a plan to respond"). Tasks should be single-line, concise, and unnumbered; IDs are assigned automatically and stored in the list. Before running tools each turn, sync the list (add/complete/delete). After updating the list, keep working—do not pause for confirmation. Before replying, check the list: if tasks remain and you are not blocked, keep working instead of replying. If blocked/awaiting user, give a brief status + ask. Use list_tasks only when you need to show the list; avoid spamming. For task tools, put the content in 'args' (leave 'target' empty); if you accidentally put data in 'target', it will be treated as the payload.
Skills: when asked to use a skill, (1) call list_skills if you have not already to see available skills; (2) read the skill's SKILL.md using its folder path from list_skills (e.g., /path/to/skills/foo/SKILL.md); (3) apply the instructions/examples to produce the requested output. If you cannot apply the skill after reading SKILL.md, ask one concise clarifying question.
General questions (stories, planning, “which tools are available?”) should be answered directly without calling tools. Do NOT enumerate or read skills at session start unless the user asks about skills or to use a specific skill. When asked which tools are available, answer from the tool list above and do NOT call tools.
When you need to use tools, respond ONLY with a JSON object (no surrounding text). Use one of:
{'tool':'read','target':'./filename','args':''}
{'tool':'read','target':'./filename','args':'lines:10-40'}
{'tool':'write','target':'./filename','args':'new content'}
{'tool':'mkdir','target':'./dirname','args':''}
{'tool':'move','target':'./old_path','args':'./new_path'}
{'tool':'find','target':'.','args':''}
{'tool':'ls','target':'.','args':''}
{'tool':'grep','target':'./path','args':'substring'}
{'tool':'list_skills','target':'','args':''}
{'tool':'add_task','target':'','args':'task description'}
{'tool':'complete_task','target':'','args':'task id'}
{'tool':'delete_task','target':'','args':'task id'}
{'tool':'list_tasks','target':'','args':''}
{'tool':'git_add','target':'./path','args':''}
{'tool':'git_commit','target':'','args':'commit message'}
Rules: paths are relative to the working directory; do not escape with .. or absolute paths. Quote paths with spaces when issuing tools. Issue one tool call at a time (multiple tool JSON blocks will only run the first). After a tool call, immediately continue: if more tools are needed, call them; otherwise reply directly to the user. Do not stay silent or return empty replies. The user cannot see tool output. Keep outputs concise. If asked about available tools, answer directly from the list above."""

GENERAL_INTRO_PROMPT = """You are running inside an agentic loop that can repeatedly call tools and skills to complete the user's request. Act autonomously: plan, use tools, and work through your task list until the job is done.
- ALWAYS maintain a task list (create one if missing) with an initial item like "create a plan to respond". ALWAYS manage a task list by using the task list tools.
- Use one tool call at a time; do not batch multiple JSON blocks.
- Put payloads in 'args' (leave 'target' empty unless it's a path); quote paths with spaces.
- Use skills by reading their SKILL.md and following instructions; do not guess.
- Avoid tools for trivial Q&A; only call bash if explicitly needed and enabled.
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


def build_system_message(workdir: Path, git_allowed: bool, yolo_enabled: bool, notes: NotesContext, initial_task_list: Optional[str] = None) -> Dict[str, str]:
    tool_prompt = f"{GENERAL_INTRO_PROMPT}\n\n{DEFAULT_TOOL_PROMPT}"
    if yolo_enabled:
        tool_prompt = tool_prompt.replace(
            "Rules: paths are relative to the working directory; do not escape with .. or absolute paths.",
            "Additional tool (unsafe; requires user confirmation): {'tool':'bash','target':'optional_cwd','args':'command'}.\n"
            "Rules: paths are relative to the working directory; do not escape with .. or absolute paths.",
        )

    content = (
        f"{tool_prompt}\n"
        f"Working directory: {workdir}\n"
        f"All tool paths are relative to this directory.\n"
        f"AGENTS discovery: starting from the task path ({workdir}), walk up to the repo root ({notes.repo_root}) and use the nearest AGENTS.md found. "
        f"Nearest found: {notes.nearest_agents if notes.nearest_agents else 'none'}.\n"
        f"User-level AGENTS at ~/.config/agents/AGENTS.md are used when present; repo-nearest instructions take precedence when conflicting.\n"
        f"Git tools enabled: {'yes' if git_allowed else 'no'}.\n\n"
        f"Skill structure:\n{SKILL_GUIDE}"
    )

    if yolo_enabled:
        content = (
            f"{content}\n\nUnsafe tools: 'bash' is enabled (user confirmation required per command). "
            f"To run bash, use JSON like: "
            f"{{'tool':'bash','target':'optional_cwd','args':'command'}}. "
            f"Use only when necessary."
        )

    if initial_task_list:
        content = f"{content}\n\nInitial task list:\n{initial_task_list}\nUpdate it before starting work."

    if notes.user_skills:
        content = f"{content}\n\nUser skills directory: {notes.user_skills}\nYou may read/find/ls within this path alongside repo skills."

    if notes.repo_notes:
        source = notes.nearest_agents if notes.nearest_agents else workdir / "AGENTS.md"
        content = f"{content}\n\nRepo notes (from {source}):\n{notes.repo_notes}"

    if notes.user_notes:
        content = f"{content}\n\nUser notes:\n{notes.user_notes}"

    return {"role": "system", "content": content}
