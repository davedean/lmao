from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION
from lmao.plugin_helpers import safe_target_path

PLUGIN = {
    "name": "bash",
    "description": "Run a shell command (per-call confirmation outside yolo mode). Use short timeouts for most operations (default: 10s).",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": True,
    "allow_in_read_only": False,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": True,
    "input_schema": "v2 args: {command:'...', timeout:10}; v1 args: command string; target may be cwd; timeout in seconds via args.timeout or meta.timeout_s, default 10s",
    "usage": [
        '{"tool":"bash","target":"","args":"echo ok"}',
        '{"tool":"bash","target":"","args":{"command":"echo ok"}}',
        '{"tool":"bash","target":"","args":{"command":"git status","timeout":5}}',
        '{"tool":"bash","target":"","args":{"command":"npm install","timeout":60}}',
        '{"tool":"bash","target":"","args":{"command":"python3 -m lmao \"analyze the code in src/ directory for security issues\" --headless --max-turns 10", "timeout: 180"}}',
        '{"tool":"bash","target":"","args":{"command":"python train.py","timeout":300}}',
        '{"tool":"bash","target":"","args":{"command":"sleep 2"},"meta":{"timeout_s":5}}',
    ],
}


def _success(data: dict) -> str:
    return json.dumps(
        {"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False
    )


def _error(message: str) -> str:
    return json.dumps(
        {"tool": PLUGIN["name"], "success": False, "error": message}, ensure_ascii=False
    )


def run(
    target: str,
    args: Any,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    debug_logger: Optional[object] = None,
    meta: Optional[dict] = None,
) -> str:
    if isinstance(args, dict):
        command = str(args.get("command") or args.get("cmd") or "").strip()
        timeout_raw = args.get("timeout")
        if timeout_raw is None:
            timeout_raw = args.get("timeout_s")
    else:
        command = str(args or target).strip()
        timeout_raw = None

    if timeout_raw is None and isinstance(meta, dict):
        timeout_raw = meta.get("timeout_s") if meta.get("timeout_s") is not None else meta.get("timeout")

    if not command:
        return _error("bash command is empty")

    # Parse timeout parameter, default to 10 seconds
    timeout = 10  # default
    if timeout_raw is not None:
        try:
            timeout = int(timeout_raw)
            if timeout <= 0:
                return _error("timeout must be a positive integer")
        except (ValueError, TypeError):
            return _error(f"invalid timeout value: {timeout_raw}")

    cwd = base
    if target and args:
        try:
            cwd = safe_target_path(target, base, extra_roots)
        except Exception:
            return _error(f"cwd '{target}' escapes allowed roots")

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _error(f"bash command timed out after {timeout} seconds")
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
