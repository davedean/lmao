from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional, Sequence

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
        "input_schema": "v2 args: {seconds: float}; v1 args: seconds (float) string; target fallback supported",
        "usage": [
            "{\"tool\":\"sleep\",\"target\":\"\",\"args\":\"0.5\"}",
            "{\"tool\":\"sleep\",\"target\":\"\",\"args\":{\"seconds\":0.5}}",
        ],
        "details": [
            "Use this to wait between async_poll calls.",
            "This tool blocks the process for up to 60 seconds.",
        ],
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
        "input_schema": "v2 target: file path; v2 args: {start_at:'end'|'start'}; v2 meta: {track_task:true|false}",
        "usage": [
            "{\"tool\":\"async_tail\",\"target\":\"logs/app.log\",\"args\":\"start_at=end\"}",
            "{\"tool\":\"async_tail\",\"target\":\"logs/app.log\",\"args\":{\"start_at\":\"end\"}}",
            "{\"tool\":\"async_tail\",\"target\":\"logs/app.log\",\"args\":{\"start_at\":\"end\"},\"meta\":{\"track_task\":true}}",
        ],
        "details": [
            "Starts a background tail job and returns a job_id immediately.",
            "Nothing is streamed automatically; call async_poll to fetch new lines.",
            "Events from async_poll have stream='tail' and include one line per event.",
            "If meta.track_task=true is passed, the tool adds a task list item to track completion.",
        ],
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
        "input_schema": "v2 args: {command: string, cwd?: string, track_task?: bool}; v2 meta: {track_task?: bool}. v1 fallback supports 'track_task=true -- <command>'",
        "usage": [
            "{\"tool\":\"async_bash\",\"target\":\"\",\"args\":\"-- ./long_process.sh\"}",
            "{\"tool\":\"async_bash\",\"target\":\"\",\"args\":\"track_task=true -- sleep 20\"}",
            "{\"tool\":\"async_bash\",\"target\":\"\",\"args\":{\"command\":\"sleep 20\"},\"meta\":{\"track_task\":true}}",
        ],
        "details": [
            "Runs a shell command in the background and returns a job_id.",
            "Call async_poll to read stdout/stderr events (stream='stdout'/'stderr').",
            "This tool always requires confirmation and is disabled in read-only mode.",
            "To attach a task list item in v2, prefer meta.track_task=true (or args.track_task=true).",
            "In v1, prefer: 'track_task=true -- <command>'.",
            "For compatibility, a trailing 'track_task=true' token is stripped from the command.",
        ],
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
        "input_schema": "v2 args: {job_id: string, since_seq?: int}; v1 target: job_id, args: since_seq (int) or args='job_id since_seq'",
        "usage": [
            "{\"tool\":\"async_poll\",\"target\":\"job_1\",\"args\":\"0\"}",
            "{\"tool\":\"async_poll\",\"target\":\"\",\"args\":\"job_1 0\"}",
            "{\"tool\":\"async_poll\",\"target\":\"\",\"args\":{\"job_id\":\"job_1\",\"since_seq\":0}}",
        ],
        "details": [
            "Preferred calling pattern: target=<job_id>, args=<since_seq>.",
            "Returns {events:[{seq,stream,text}], next_seq:<int>, status:<...>} in data.",
            "Use since_seq from a prior poll to fetch only new events.",
            "If the job was started with track_task=true, async_poll will mark the associated task complete when the job finishes.",
        ],
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
        "input_schema": "v2 args: {job_id: string}; v1 args: job_id string (also accepts target)",
        "usage": [
            "{\"tool\":\"async_stop\",\"target\":\"job_1\",\"args\":\"\"}",
            "{\"tool\":\"async_stop\",\"target\":\"\",\"args\":\"job_1\"}",
            "{\"tool\":\"async_stop\",\"target\":\"\",\"args\":{\"job_id\":\"job_1\"}}",
        ],
        "details": [
            "Stops/cancels a running job.",
            "This tool always requires confirmation and is disabled in read-only mode.",
        ],
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
        "usage": "{\"tool\":\"async_list\",\"target\":\"\",\"args\":\"\"}",
        "details": [
            "Returns currently known jobs and their status.",
        ],
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


def _parse_bool(value: object) -> bool:
    raw = str(value or "").strip().lower()
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off", ""):
        return False
    return False


def _split_command_and_meta(args: str, target: str) -> tuple[str, dict]:
    """
    Extract meta flags (e.g. track_task=true) from an async_bash call.

    Supported forms:
    - args="track_task=true -- <command...>" (recommended; unambiguous)
    - args="<command...> track_task=true" (common model behavior; we strip recognized meta tokens at the end)
    - args="<command...>" (no meta)
    - when args empty, command may be in target
    """
    raw = str(args or "").strip()
    meta: dict = {}
    if raw:
        if raw.startswith("-- "):
            return raw[3:].strip(), meta
        if " -- " in raw:
            meta_part, cmd_part = raw.split(" -- ", 1)
            meta = _parse_kv_args(meta_part)
            return cmd_part.strip(), meta

        tokens = raw.split()
        while tokens:
            last = tokens[-1]
            if last.startswith("track_task="):
                meta.update(_parse_kv_args(last))
                tokens.pop()
                continue
            break
        return " ".join(tokens).strip(), meta

    return str(target or "").strip(), meta


def run(
    tool_name: str,
    target: str,
    args: Any,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    task_manager=None,
    debug_logger: Optional[object] = None,
    meta: Optional[dict] = None,
) -> str:
    manager = get_async_job_manager()
    meta = meta if isinstance(meta, dict) else {}

    if tool_name == "sleep":
        if isinstance(args, dict):
            raw = str(args.get("seconds") or args.get("s") or args.get("args") or "").strip()
        else:
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
        if isinstance(args, dict):
            start_at = str(args.get("start_at") or "end").strip() or "end"
            track_task = bool(meta.get("track_task") if "track_task" in meta else args.get("track_task"))
        else:
            kv = _parse_kv_args(str(args))
            start_at = kv.get("start_at", "end")
            track_task = _parse_bool(kv.get("track_task")) or _parse_bool(meta.get("track_task"))
        if start_at not in ("start", "end"):
            return _error(tool_name, "start_at must be 'start' or 'end'")
        job_id = manager.start_tail(path, start_at=start_at)
        task_added = False
        if task_manager is not None and track_task:
            task = task_manager.add_task(f"wait for async job {job_id}: tail {path}")
            manager.attach_task(job_id, task_id=task.id, label="tail")
            task_added = True
        data = {"job_id": job_id, "path": str(path), "start_at": start_at, "task_added": task_added}
        return _success(tool_name, data)

    if tool_name == "async_bash":
        if isinstance(args, dict):
            command = str(args.get("command") or "").strip()
            track_task = bool(meta.get("track_task") if "track_task" in meta else args.get("track_task"))
            cwd_hint = str(args.get("cwd") or "").strip()
        else:
            command, extracted = _split_command_and_meta(str(args), target)
            meta.update(extracted)
            track_task = _parse_bool(meta.get("track_task"))
            cwd_hint = ""
        if not command:
            return _error(tool_name, "async_bash command is empty")
        cwd = base
        if cwd_hint:
            try:
                cwd = safe_target_path(cwd_hint, base, extra_roots)
            except Exception:
                return _error(tool_name, f"cwd '{cwd_hint}' escapes allowed roots")
        elif target and args:
            try:
                cwd = safe_target_path(target, base, extra_roots)
            except Exception:
                return _error(tool_name, f"cwd '{target}' escapes allowed roots")
        job_id = manager.start_bash(command, cwd=cwd)
        task_added = False
        if task_manager is not None and track_task:
            task = task_manager.add_task(f"wait for async job {job_id}: bash {command}")
            manager.attach_task(job_id, task_id=task.id, label="bash")
            task_added = True
        return _success(tool_name, {"job_id": job_id, "cwd": str(cwd), "command": command, "task_added": task_added})

    if tool_name == "async_poll":
        if isinstance(args, dict):
            job_id = str(args.get("job_id") or target or "").strip()
            try:
                since_seq = int(args.get("since_seq") or 0)
            except Exception:
                since_seq = 0
        else:
            job_id = str(target or "").strip()
            if _looks_like_job_id(job_id):
                since_seq = _parse_since_seq(str(args))
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
        task_updated = False
        if task_manager is not None and payload.get("task_id") and payload.get("status") in ("done", "error", "canceled"):
            try:
                task_id = int(payload["task_id"])
            except Exception:
                task_id = None
            if task_id is not None:
                msg = task_manager.complete_task(task_id)
                if not msg.startswith("error:"):
                    task_updated = True
                    payload["task_complete_message"] = msg
                    payload["tasks"] = task_manager.to_payload()
        payload["task_updated"] = task_updated
        return _success(tool_name, payload)

    if tool_name == "async_stop":
        if isinstance(args, dict):
            job_id = str(args.get("job_id") or target or "").strip()
        else:
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
