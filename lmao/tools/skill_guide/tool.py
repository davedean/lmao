from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION

PLUGIN = {
    "name": "skill_guide",
    "description": "Return guidance on how skills are structured and when to consult/list them.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "none",
    "usage": "{'tool':'skill_guide','target':'','args':''}",
}


SKILL_GUIDE = """Skills are instructions for agents on how to accomplish tasks. They are not tools.

Location:
- Repo skills live under ./skills/<skill-name>/SKILL.md
- User skills may also exist under ~/.config/agents/skills/<skill-name>/SKILL.md

Format:
- Each skill is a folder under the skills root (e.g., ./skills/my-skill/)
- The entrypoint is SKILL.md in that folder; supporting files can live alongside it.
- SKILL.md must start with YAML frontmatter, for example:

---
name: your-skill-name
description: brief description of what this Skill does and when to use it
---

## Instructions
Step-by-step guidance.

## Examples
Concrete usage examples.

Constraints:
- name: <=64 chars, lowercase letters/numbers/hyphens only; no XML tags; not 'anthropic' or 'claude'
- description: non-empty, <=1024 chars, no XML tags; include what the Skill does and when to use it

Usage rules:
- Only talk about skills when the user asks about skills or requests one.
- Only call list_skills if the user explicitly asks to list skills, or requests a skill and you lack its path.
- When asked to use a skill: prefer any prelisted/discovered paths; otherwise call list_skills, then read that SKILL.md and follow its instructions/examples exactly (do not guess).
- Do not enumerate or read random skills unless the user asks about skills or requests one.
- If you cannot apply a skill after reading it, ask one concise clarifying question."""


def _success(data: dict) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False)


def run(
    target: str,
    args: str,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    task_manager=None,
    debug_logger: Optional[object] = None,
) -> str:
    return _success({"guide": SKILL_GUIDE})

