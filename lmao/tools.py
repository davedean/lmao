from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .context import find_repo_root
from .debug_log import DebugLogger
from .skills import list_skill_info, validate_skill_write_target
from .task_list import TaskListManager

ALLOWED_TOOLS = {
    "read",
    "write",
    "grep",
    "find",
    "ls",
    "mkdir",
    "move",
    "git_add",
    "git_commit",
    "bash",
    "list_skills",
    "add_task",
    "complete_task",
    "delete_task",
    "list_tasks",
}

def json_success(tool: str, data: Any, note: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {"tool": tool, "success": True, "data": data}
    if note:
        payload["note"] = note
    return json.dumps(payload, ensure_ascii=False)


def json_error(tool: str, message: str) -> str:
    return json.dumps({"tool": tool, "success": False, "error": message}, ensure_ascii=False)


def normalize_path_for_output(path: Path, base: Path) -> str:
    try:
        rel = path.relative_to(base)
    except Exception:
        rel = path
    rel_str = str(rel)
    if path.is_dir() and not rel_str.endswith("/"):
        rel_str += "/"
    return rel_str


@dataclass
class ToolCall:
    tool: str
    target: str
    args: str

    @classmethod
    def from_raw_message(cls, raw_text: str) -> Optional["ToolCall"]:
        calls = parse_tool_calls(raw_text)
        return calls[0] if calls else None


def iter_json_candidates(raw_text: str) -> List[str]:
    candidates: List[str] = []
    stripped = raw_text.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL)
    candidates.extend(fenced)
    candidates.extend(extract_braced_objects(raw_text))
    if stripped:
        candidates.append(stripped)
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        unique.append(cand)
    return unique


def parse_tool_calls(raw_text: str) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for candidate in iter_json_candidates(raw_text):
        parsed = load_candidate(candidate)
        if not parsed:
            continue
        if isinstance(parsed, list):
            parsed_items = parsed
        else:
            parsed_items = [parsed]
        for obj in parsed_items:
            if not isinstance(obj, dict):
                continue
            tool = str(obj.get("tool", "")).strip()
            if tool not in ALLOWED_TOOLS:
                continue
            target = str(obj.get("target", "") or "").strip()
            args = obj.get("args", "")
            args_str = args if isinstance(args, str) else json.dumps(args)
            calls.append(ToolCall(tool=tool, target=target, args=args_str))
    return calls


def covers_message(candidate: str, raw_text: str) -> bool:
    stripped = raw_text.strip()
    cand = candidate.strip()
    if not cand:
        return False
    if cand == stripped:
        return True
    fenced_json = f"```json\n{cand}\n```"
    fenced_plain = f"```\n{cand}\n```"
    return stripped == fenced_json or stripped == fenced_plain


def extract_braced_objects(raw_text: str) -> List[str]:
    """Extract top-level brace-delimited JSON-ish substrings."""
    objs: List[str] = []
    depth = 0
    start = None
    for idx, ch in enumerate(raw_text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(raw_text[start : idx + 1].strip())
                    start = None
    return objs


def load_candidate(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    if not cleaned:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def safe_target_path(target: str, base: Path, extra_roots: Sequence[Path]) -> Path:
    raw_path = Path(target).expanduser()
    target_path = raw_path.resolve() if raw_path.is_absolute() else (base / raw_path).resolve()

    allowed_roots = [base] + [p.resolve() for p in extra_roots]
    for root in allowed_roots:
        try:
            target_path.relative_to(root)
            return target_path
        except Exception:
            continue
    raise ValueError("path escapes allowed roots")


def parse_line_range(arg: str) -> Optional[tuple]:
    match = re.search(r"lines?[:=]\s*(\d+)(?:[-:](\d+))?", arg)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    if start < 1:
        start = 1
    if end < start:
        end = start
    return start, end


def run_tool(
    tool_call: ToolCall,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    git_allowed: bool,
    yolo_enabled: bool,
    task_manager: Optional[TaskListManager] = None,
    debug_logger: Optional[DebugLogger] = None,
) -> str:
    tool = tool_call.tool
    target = tool_call.target
    args = tool_call.args

    if debug_logger:
        debug_logger.log(
            "tool.dispatch",
            f"tool={tool} target={target!r} args={args!r} base={base} extra_roots={[str(r) for r in extra_roots]}",
    )

    if tool not in ALLOWED_TOOLS:
        return json_error(tool, f"unsupported tool '{tool}'")

    try:
        target_path = safe_target_path(target or ".", base, extra_roots)
    except Exception as exc:
        if debug_logger:
            debug_logger.log("tool.error", f"tool={tool} target={target!r} path_escape_error={exc}")
        return json_error(tool, f"target path '{target}' escapes working directory")

    if tool == "read":
        if not target_path.exists():
            return json_error(tool, f"file '{target}' not found")
        try:
            content = target_path.read_text(encoding="utf-8")
        except Exception as exc:
            return json_error(tool, f"unable to read '{target}': {exc}")
        line_range = parse_line_range(str(args))
        truncated = False
        if line_range:
            start, end = line_range
            lines = content.splitlines()
            selected = lines[start - 1 : end]
            content = "\n".join(selected)
            range_info = {"start": start, "end": end}
        else:
            range_info = None
        if len(content) > 200_000:
            content = content[:200_000]
            truncated = True
        data: Dict[str, Any] = {
            "path": normalize_path_for_output(target_path, base),
            "content": content,
        }
        if range_info:
            data["lines"] = range_info
        if truncated:
            data["truncated"] = True
        return json_success(tool, data)

    if tool == "write":
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() and target_path.is_dir():
                return json_error(tool, f"'{target}' is a directory; use mkdir instead")
            if str(target).rstrip().endswith(("/", "\\")):
                return json_error(tool, f"'{target}' looks like a directory path; use mkdir instead")
            skill_error = validate_skill_write_target(target_path, skill_roots)
            if skill_error:
                return json_error(tool, skill_error)
            with target_path.open("w", encoding="utf-8") as fh:
                fh.write(args or "")
            data = {
                "path": normalize_path_for_output(target_path, base),
                "bytes": len(args or ""),
            }
            return json_success(tool, data)
        except Exception as exc:
            return json_error(tool, f"unable to write '{target}': {exc}")

    if tool == "mkdir":
        try:
            target_path.mkdir(parents=True, exist_ok=True)
            data = {"path": normalize_path_for_output(target_path, base), "created": True}
            return json_success(tool, data)
        except Exception as exc:
            return json_error(tool, f"unable to create directory '{target}': {exc}")

    if tool == "move":
        try:
            dest_path = safe_target_path(str(args), base, extra_roots)
        except Exception:
            return json_error(tool, f"destination path '{args}' escapes working directory")
        try:
            if not target_path.exists():
                return json_error(tool, f"source '{target}' not found")
            if dest_path.exists():
                return json_error(tool, f"destination '{args}' already exists")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.replace(dest_path)
            data = {
                "from": normalize_path_for_output(target_path, base),
                "to": normalize_path_for_output(dest_path, base),
            }
            return json_success(tool, data)
        except Exception as exc:
            return json_error(tool, f"unable to move '{target}' to '{args}': {exc}")

    if tool == "ls":
        if not target_path.exists():
            return json_error(tool, f"path '{target}' not found")
        try:
            if target_path.is_file():
                entries = [{
                    "name": target_path.name,
                    "path": normalize_path_for_output(target_path, base),
                    "type": "file",
                }]
                data = {"path": normalize_path_for_output(target_path, base), "entries": entries}
                return json_success(tool, data)
            entries_list = []
            for p in sorted(target_path.iterdir(), key=lambda x: x.name):
                entries_list.append({
                    "name": p.name,
                    "path": normalize_path_for_output(p, base),
                    "type": "dir" if p.is_dir() else "file",
                })
            data = {"path": normalize_path_for_output(target_path, base), "entries": entries_list}
            return json_success(tool, data)
        except Exception as exc:
            return json_error(tool, f"unable to list '{target}': {exc}")

    if tool == "find":
        if not target_path.exists():
            return json_error(tool, f"path '{target}' not found")
        results: List[Dict[str, str]] = []
        truncated = False
        try:
            for root, dirs, files in os.walk(target_path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                visible_files = [f for f in files if not f.startswith(".")]
                for name in sorted(visible_files + dirs):
                    path = Path(root) / name
                    rel_display = normalize_path_for_output(path, base)
                    results.append({
                        "name": name,
                        "path": rel_display,
                        "type": "dir" if path.is_dir() else "file",
                    })
                    if len(results) >= 200:
                        truncated = True
                        break
                if truncated:
                    break
            data: Dict[str, Any] = {
                "path": normalize_path_for_output(target_path, base),
                "results": results,
            }
            if truncated:
                data["truncated"] = True
            return json_success(tool, data)
        except Exception as exc:
            return json_error(tool, f"unable to walk '{target}': {exc}")

    if tool == "grep":
        pattern = str(args)
        if target_path.is_dir():
            candidates = [p for p in target_path.rglob("*") if p.is_file()]
        else:
            candidates = [target_path]
        matches: List[Dict[str, Any]] = []
        truncated = False
        for candidate in candidates:
            try:
                content = candidate.read_text(encoding="utf-8")
            except Exception:
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    matches.append({
                        "path": normalize_path_for_output(candidate, base),
                        "line": lineno,
                        "text": line,
                    })
                if len(matches) >= 200:
                    truncated = True
                    break
            if truncated:
                break
        data: Dict[str, Any] = {
            "path": normalize_path_for_output(target_path, base),
            "pattern": pattern,
            "matches": matches,
        }
        if truncated:
            data["truncated"] = True
        return json_success(tool, data)

    if tool in {"add_task", "complete_task", "delete_task", "list_tasks"}:
        if task_manager is None:
            return json_error(tool, "task list manager unavailable")
        payload = args if args is not None else ""
        if isinstance(payload, str):
            payload = payload.strip()
        if (not payload) and target:
            payload = str(target).strip()
        if tool == "add_task":
            text = normalize_task_text(str(payload).strip())
            if not text:
                return json_error(tool, "task text is required")
            task = task_manager.add_task(text)
            data = {
                "task": {"id": task.id, "text": task.text, "done": task.done},
                "tasks": task_manager.to_payload(),
            }
            return json_success(tool, data)
        if tool == "complete_task":
            try:
                task_id = int(str(payload).strip())
            except Exception:
                return json_error(tool, "task id must be an integer")
            message = task_manager.complete_task(task_id)
            if message.startswith("error:"):
                return json_error(tool, message)
            data = {"message": message, "tasks": task_manager.to_payload()}
            return json_success(tool, data)
        if tool == "delete_task":
            try:
                task_id = int(str(payload).strip())
            except Exception:
                return json_error(tool, "task id must be an integer")
            message = task_manager.delete_task(task_id)
            if message.startswith("error:"):
                return json_error(tool, message)
            data = {"message": message, "tasks": task_manager.to_payload()}
            return json_success(tool, data)
        if tool == "list_tasks":
            data = {"tasks": task_manager.to_payload(), "render": task_manager.render_tasks()}
            return json_success(tool, data)

    if tool == "git_add":
        if not git_allowed:
            return json_error(tool, "git operations are disabled (enable with --allow-git)")
        repo_root = find_repo_root(base)
        if not (repo_root / ".git").exists():
            return json_error(tool, "not inside a git repository")
        try:
            rel = str(target_path.relative_to(repo_root))
        except Exception:
            rel = str(target_path)
        try:
            result = subprocess.run(["git", "add", rel], cwd=repo_root, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return json_error(tool, f"git add failed: {result.stderr.strip() or result.stdout.strip()}")
            data = {"path": rel, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
            return json_success(tool, data)
        except Exception as exc:
            return json_error(tool, f"git add exception: {exc}")

    if tool == "git_commit":
        if not git_allowed:
            return json_error(tool, "git operations are disabled (enable with --allow-git)")
        repo_root = find_repo_root(base)
        if not (repo_root / ".git").exists():
            return json_error(tool, "not inside a git repository")
        message = str(args).strip()
        if not message:
            return json_error(tool, "commit message is required")
        try:
            result = subprocess.run(["git", "commit", "-m", message], cwd=repo_root, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                return json_error(tool, f"git commit failed: {result.stderr.strip() or result.stdout.strip()}")
            data = {"message": result.stdout.strip() or "ok", "stderr": result.stderr.strip()}
            return json_success(tool, data)
        except Exception as exc:
            return json_error(tool, f"git commit exception: {exc}")

    if tool == "bash":
        if not yolo_enabled:
            return json_error(tool, "bash tool is disabled (enable with --yolo)")
        command = str(args or target).strip()
        if not command:
            return json_error(tool, "bash command is empty")
        cwd = base
        if target and args:
            try:
                cwd = safe_target_path(target, base, extra_roots)
            except Exception:
                return json_error(tool, f"cwd '{target}' escapes allowed roots")
        print(f"[bash request] {command}")
        try:
            approval = input("Allow running this bash command? [y/N]: ").strip().lower()
        except EOFError:
            approval = ""
        if not approval.startswith("y"):
            return json_error(tool, "bash command not approved by user")
        try:
            result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                return json_error(
                    tool,
                    f"bash exit {result.returncode}",
                )
            data = {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "exit_code": result.returncode,
            }
            return json_success(tool, data)
        except Exception as exc:
            return json_error(tool, f"bash exception: {exc}")

    if tool == "list_skills":
        skills = list_skill_info(skill_roots)
        data = [{"name": name, "path": str(path)} for name, path in skills]
        return json_success(tool, data)

    return json_error(tool, "unexpected tool handling branch")


def summarize_output(output: str, max_lines: int = 8, max_chars: int = 400) -> str:
    if max_lines <= 0 and max_chars <= 0:
        return ""
    lines = output.splitlines()
    head = lines[:max_lines] if max_lines > 0 else lines
    summary = "\n".join(head)
    if max_lines > 0 and len(lines) > max_lines:
        summary += "\n...[truncated]"
    if max_chars > 0 and len(summary) > max_chars:
        summary = summary[:max_chars] + "\n...[truncated]"
    return summary


def normalize_task_text(text: str) -> str:
    """Normalize task text: remove numbering, collapse whitespace, avoid newlines."""
    cleaned = text.replace("\n", " ").replace("\r", " ")
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def display_path_for_output(path: str) -> str:
    """Quote paths with whitespace so the model can reuse them accurately."""
    return f'"{path}"' if any(ch.isspace() for ch in path) else path
