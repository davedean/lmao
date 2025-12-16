from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from lmao.plugin_helpers import find_repo_root, safe_target_path
from lmao.plugins import PLUGIN_API_VERSION

PLUGINS = [
    {
        "name": "git_status",
        "description": "Show git status (porcelain + branch) for the current repo.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "target ignored; args ignored",
        "usage": [
            "{\"tool\":\"git_status\",\"target\":\"\",\"args\":\"\"}",
            "{\"tool\":\"git_status\",\"target\":\"\",\"args\":{}}",
        ],
    },
    {
        "name": "git_diff",
        "description": "Show git diff; supports args: '--staged' and/or '--stat'.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "target optional path; args: '--staged' and/or '--stat'",
        "usage": [
            "{\"tool\":\"git_diff\",\"target\":\"\",\"args\":\"\"}",
            "{\"tool\":\"git_diff\",\"target\":\"\",\"args\":\"--staged\"}",
            "{\"tool\":\"git_diff\",\"target\":\"\",\"args\":{\"staged\":true,\"stat\":false}}",
            "{\"tool\":\"git_diff\",\"target\":\"\",\"args\":{\"staged\":true,\"stat\":false,\"path\":\"path/to/file\"}}",
        ],
    },
    {
        "name": "git_add",
        "description": "Stage a file or directory with git add.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": True,
        "allow_in_read_only": False,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "v2 args: {path:'./path'}; v1 target: path to add",
        "usage": [
            "{\"tool\":\"git_add\",\"target\":\"./path\",\"args\":\"\"}",
            "{\"tool\":\"git_add\",\"target\":\"\",\"args\":{\"path\":\"./path\"}}",
        ],
    },
    {
        "name": "git_commit",
        "description": "Create a git commit with the provided message.",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": True,
        "allow_in_read_only": False,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "input_schema": "v2 args: {message:'...'}; v1 args: commit message string",
        "usage": [
            "{\"tool\":\"git_commit\",\"target\":\"\",\"args\":\"commit message\"}",
            "{\"tool\":\"git_commit\",\"target\":\"\",\"args\":{\"message\":\"commit message\"}}",
        ],
    },
]


def _success(tool: str, data: dict) -> str:
    return json.dumps({"tool": tool, "success": True, "data": data}, ensure_ascii=False)


def _error(tool: str, message: str) -> str:
    return json.dumps({"tool": tool, "success": False, "error": message}, ensure_ascii=False)


def _trim_output(text: str, limit: int = 200_000) -> Tuple[str, bool]:
    content = (text or "").rstrip()
    if len(content) > limit:
        return content[:limit], True
    return content, False


def _parse_diff_args(arg_str: str) -> tuple[bool, bool] | tuple[None, None]:
    tokens = [t for t in (arg_str or "").strip().split() if t]
    staged = False
    stat = False
    for token in tokens:
        if token in ("--staged", "staged"):
            staged = True
            continue
        if token in ("--stat", "stat"):
            stat = True
            continue
        return None, None
    return staged, stat


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
    repo_root = find_repo_root(base)
    if not (repo_root / ".git").exists():
        return _error(tool_name, "not inside a git repository")

    if tool_name == "git_status":
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain=v1", "-b"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            return _error(tool_name, f"git status exception: {exc}")
        if result.returncode != 0:
            return _error(tool_name, f"git status failed: {result.stderr.strip() or result.stdout.strip()}")
        output, truncated = _trim_output(result.stdout)
        data = {"repo_root": str(repo_root), "output": output}
        if truncated:
            data["truncated"] = True
        return _success(tool_name, data)

    if tool_name == "git_diff":
        if isinstance(args, dict):
            staged = bool(args.get("staged", False))
            stat = bool(args.get("stat", False))
            target = str(args.get("path") or target or "")
        else:
            staged, stat = _parse_diff_args(str(args))
        if staged is None:
            return _error(tool_name, "unsupported args; allowed: '--staged' and/or '--stat'")

        try:
            target_path = safe_target_path(target or ".", base, extra_roots)
        except Exception as exc:
            return _error(tool_name, f"invalid target '{target}': {exc}")

        rel_path: Optional[str]
        if target_path == base and target in ("", ".", "/", "\\"):
            rel_path = None
        else:
            try:
                rel_path = str(target_path.relative_to(repo_root))
            except Exception:
                rel_path = str(target_path)

        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if stat:
            cmd.append("--stat")
        if rel_path and rel_path not in (".", ""):
            cmd.extend(["--", rel_path])

        try:
            result = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            return _error(tool_name, f"git diff exception: {exc}")
        if result.returncode != 0:
            return _error(tool_name, f"git diff failed: {result.stderr.strip() or result.stdout.strip()}")

        output, truncated = _trim_output(result.stdout)
        data = {"repo_root": str(repo_root), "staged": staged, "stat": stat, "output": output}
        if rel_path:
            data["path"] = rel_path
        if truncated:
            data["truncated"] = True
        return _success(tool_name, data)

    if tool_name == "git_add":
        if isinstance(args, dict) and not target:
            target = str(args.get("path") or "")
        try:
            target_path = safe_target_path(target or ".", base, extra_roots)
        except Exception as exc:
            return _error(tool_name, f"invalid target '{target}': {exc}")

        try:
            rel = str(target_path.relative_to(repo_root))
        except Exception:
            rel = str(target_path)

        try:
            result = subprocess.run(
                ["git", "add", rel],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            return _error(tool_name, f"git add exception: {exc}")
        if result.returncode != 0:
            return _error(tool_name, f"git add failed: {result.stderr.strip() or result.stdout.strip()}")
        data = {"path": rel, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
        return _success(tool_name, data)

    if tool_name == "git_commit":
        if isinstance(args, dict):
            message = str(args.get("message") or args.get("msg") or "").strip()
        else:
            message = str(args or "").strip()
        if not message:
            return _error(tool_name, "commit message is required")
        try:
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as exc:
            return _error(tool_name, f"git commit exception: {exc}")
        if result.returncode != 0:
            return _error(tool_name, f"git commit failed: {result.stderr.strip() or result.stdout.strip()}")
        data = {"message": result.stdout.strip() or "ok", "stderr": result.stderr.strip()}
        return _success(tool_name, data)

    return _error(tool_name, f"unsupported tool '{tool_name}'")
