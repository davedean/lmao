from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .hooks import LoggingHookContext, LoggingHookTypes, get_global_hook_registry

_MAX_LOG_DETAIL_CHARS = 2000


def _truncate_value(value: Any, limit: int = _MAX_LOG_DETAIL_CHARS) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...[truncated]"
    if isinstance(value, dict):
        return {key: _truncate_value(val, limit) for key, val in value.items()}
    if isinstance(value, list):
        return [_truncate_value(item, limit) for item in value]
    return value


def _coerce_log_payload(detail: Any) -> tuple[str, str, Dict[str, Any]]:
    level = "debug"
    message = ""
    data: Dict[str, Any] = {}
    if isinstance(detail, dict):
        level = str(detail.get("level") or level)
        message = str(detail.get("message") or "")
        extra = detail.get("data")
        if isinstance(extra, dict):
            data.update(extra)
        if "detail" in detail and "detail" not in data:
            data["detail"] = detail.get("detail")
        for key, value in detail.items():
            if key in {"level", "message", "data", "detail"}:
                continue
            if key not in data:
                data[key] = value
    elif detail not in (None, ""):
        data["detail"] = detail
    return level, message, data


@dataclass
class DebugLogger:
    """Minimal file logger for capturing debug traces without altering behavior."""

    path: Path

    def log(self, event: str, detail: Any) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        level, message, data = _coerce_log_payload(detail)
        if not message:
            message = event
        registry = get_global_hook_registry()
        context = LoggingHookContext(
            hook_type=LoggingHookTypes.ON_LOG_MESSAGE,
            runtime_state={"event": event, "detail": detail, "timestamp": timestamp, "data": data},
            log_level=level,
            log_message=message,
            log_data=data,
            log_source=event,
        )
        on_log_result = registry.execute_hooks(LoggingHookTypes.ON_LOG_MESSAGE, context)
        if on_log_result.should_skip:
            return
        if on_log_result.modified_context and isinstance(on_log_result.modified_context, LoggingHookContext):
            context = on_log_result.modified_context
        pre_result = registry.execute_hooks(LoggingHookTypes.PRE_LOG_WRITE, context)
        if pre_result.should_skip:
            return
        if pre_result.modified_context and isinstance(pre_result.modified_context, LoggingHookContext):
            context = pre_result.modified_context
        entry = {
            "ts": timestamp,
            "level": context.log_level or level,
            "event": event,
            "message": context.log_message or message,
            "data": _truncate_value(context.log_data or {}),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            # Logging failures should never interrupt the run loop.
            pass
        registry.execute_hooks(LoggingHookTypes.POST_LOG_WRITE, context)
