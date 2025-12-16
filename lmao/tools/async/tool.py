from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional, Sequence

from lmao.async_jobs import get_async_job_manager
from lmao.plugin_helpers import safe_target_path
from lmao.plugins import PLUGIN_API_VERSION

PLUGINS = [
    {
        "name": "sleep",
        "description": "Sleep for N seconds (float).",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "args: seconds (float) (also accepts target when args empty)",
        "usage": "{'tool':'sleep','target':'','args':'0.5'}",
    },
    {
        "name": "async_tail",
        "description": "Start tailing a file; use async_poll for updates.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "target: file path; args: 'start_at=end' or 'start_at=start'",
        "usage": "{'tool':'async_tail','target':'logs/app.log','args':'start_at=end'}",
    },
    {
        "name": "async_bash",
        "description": "Run a shell command asynchronously (requires confirmation); use async_poll for updates.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": True,
        "allow_in_read_only": False,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": True,
        "input_schema": "args: command; target: optional cwd (when args provided)",
        "usage": "{'tool':'async_bash','target':'','args':'./long_process.sh'}",
    },
    {
        "name": "async_poll",
        "description": "Poll an async job by id.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "args: job_id or 'job_id since_seq' (also accepts target when args empty)",
        "usage": "{'tool':'async_poll','target':'','args':'job_1 0'}",
    },
    {
        "name": "async_stop",
        "description": "Stop/cancel an async job by id.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": True,
        "allow_in_read_only": False,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": True,
        "input_schema": "args: job_id (also accepts target when args empty)",
        "usage": "{'tool':'async_stop','target':'','args':'job_1'}",
    },
    {
        "name": "async_list",
        "description": "List async jobs.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "none",
        "usage": "{'tool':'async_list','target':'','args':''}",
    },
]


def _success(tool: str, data: dict) -> str:
    return json.dumps({"tool": tool, "success": True, "data": data}, ensure_ascii=False)


def _error(tool: str, message: str) -> str:
    return json.dumps({"tool": tool, "success": False, "error": message}, ensure_ascii=False)


def _parse_kv_args(args: str) -> dict:
    parsed: dict = {}
    for part in str(args or "").split():
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            parsed[k] = v
    return parsed


_JOB_ID_RE = re.compile(r"^job_\d+$")


def _looks_like_job_id(value: str) -> bool:
    return bool(_JOB_ID_RE.match(str(value or "").strip()))


def _parse_since_seq(value: str) -> int:
    raw = str(value or "").strip()
    if not raw:
        return 0
    if raw.isdigit():
        return int(raw)
    kv = _parse_kv_args(raw)
    if "since_seq" in kv:
        try:
            return int(str(kv["since_seq"]).strip())
        except Exception:
            return 0
    return 0


def run(
    tool_name: str,
    target: str,
    args: str,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    task_manager=None,
    debug_logger: Optional[object] = None,
) -> str:
    manager = get_async_job_manager()

    if tool_name == "sleep":
        raw = str(args or target or "").strip()
        if not raw:
            return _error(tool_name, "sleep args must be seconds (float)")
        try:
            seconds = float(raw)
        except Exception:
            return _error(tool_name, f"invalid seconds: {raw!r}")
        if seconds < 0 or seconds > 60:
            return _error(tool_name, "sleep seconds must be between 0 and 60")
        time.sleep(seconds)
        return _success(tool_name, {"slept_s": seconds})

    if tool_name == "async_tail":
        if not target.strip():
            return _error(tool_name, "async_tail target must be a file path")
        try:
            path = safe_target_path(target, base, extra_roots)
        except Exception:
            return _error(tool_name, f"target '{target}' escapes allowed roots")
        if not path.exists() or not path.is_file():
            return _error(tool_name, f"file not found: {path}")
        kv = _parse_kv_args(args)
        start_at = kv.get("start_at", "end")
        if start_at not in ("start", "end"):
            return _error(tool_name, "start_at must be 'start' or 'end'")
        job_id = manager.start_tail(path, start_at=start_at)
        return _success(tool_name, {"job_id": job_id, "path": str(path), "start_at": start_at})

    if tool_name == "async_bash":
        command = str(args or target).strip()
        if not command:
            return _error(tool_name, "async_bash command is empty")
        cwd = base
        if target and args:
            try:
                cwd = safe_target_path(target, base, extra_roots)
            except Exception:
                return _error(tool_name, f"cwd '{target}' escapes allowed roots")
        job_id = manager.start_bash(command, cwd=cwd)
        return _success(tool_name, {"job_id": job_id, "cwd": str(cwd), "command": command})

    if tool_name == "async_poll":
        job_id = str(target or "").strip()
        if _looks_like_job_id(job_id):
            since_seq = _parse_since_seq(args)
        else:
            raw = str(args or target or "").strip()
            if not raw:
                return _error(tool_name, "async_poll expects: target=job_id (preferred) or args='job_id [since_seq]'")
            parts = raw.split()
            job_id = parts[0]
            since_seq = _parse_since_seq(" ".join(parts[1:])) if len(parts) > 1 else 0
        payload = manager.poll(job_id, since_seq=since_seq)
        if payload is None:
            return _error(tool_name, f"job not found: {job_id}")
        return _success(tool_name, payload)

    if tool_name == "async_stop":
        job_id = str(args or target or "").strip()
        if not job_id:
            return _error(tool_name, "async_stop args must be: job_id")
        ok = manager.stop(job_id)
        if not ok:
            return _error(tool_name, f"job not found: {job_id}")
        return _success(tool_name, {"job_id": job_id, "stopped": True})

    if tool_name == "async_list":
        return _success(tool_name, {"jobs": manager.list_jobs()})

    return _error(tool_name, f"unsupported tool '{tool_name}'")
