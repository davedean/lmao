from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

_MAX_LOG_DETAIL_CHARS = 2000


def _truncate_value(value: Any, limit: int = _MAX_LOG_DETAIL_CHARS) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...[truncated]"
    if isinstance(value, dict):
        return {key: _truncate_value(val, limit) for key, val in value.items()}
    if isinstance(value, list):
        return [_truncate_value(item, limit) for item in value]
    return value


@dataclass
class ErrorLogger:
    """Structured logger for tool/protocol failures without affecting runtime behavior."""

    path: Path

    def log(self, event: str, data: Dict[str, Any]) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        entry = {
            "ts": timestamp,
            "level": "error",
            "event": event,
            "message": event,
            "data": _truncate_value(data or {}),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            # Logging failures should never interrupt the run loop.
            pass
