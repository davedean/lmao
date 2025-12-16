from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.plugin_helpers import safe_target_path

PLUGIN = {
    "name": "bash",
    "description": "Run a shell command with user confirmation.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": True,
    "allow_in_read_only": False,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": True,
    "input_schema": "v2 args: {command:'...'}; v1 args: command string; target may be cwd",
    "usage": [
        "{\"tool\":\"bash\",\"target\":\"\",\"args\":\"echo ok\"}",
        "{\"tool\":\"bash\",\"target\":\"\",\"args\":{\"command\":\"echo ok\"}}",
    ],
}


def _success(data: dict) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": False, "error": message}, ensure_ascii=False)


def run(
    target: str,
    args: Any,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    task_manager=None,
    debug_logger: Optional[object] = None,
    meta: Optional[dict] = None,
) -> str:
    if isinstance(args, dict):
        command = str(args.get("command") or args.get("cmd") or "").strip()
    else:
        command = str(args or target).strip()
    if not command:
        return _error("bash command is empty")

    cwd = base
    if target and args:
        try:
            cwd = safe_target_path(target, base, extra_roots)
        except Exception:
            return _error(f"cwd '{target}' escapes allowed roots")

    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=120)
    except Exception as exc:
        return _error(f"bash exception: {exc}")

    if result.returncode != 0:
        return _error(f"bash exit {result.returncode}")

    data = {
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "exit_code": result.returncode,
    }
    return _success(data)
