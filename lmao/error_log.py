from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@dataclass
class ErrorLogger:
    """Structured logger for tool/protocol failures without affecting runtime behavior."""

    path: Path

    def log(self, event: str, data: Dict[str, Any]) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        entry = {"ts": timestamp, "event": event, "data": data}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            # Logging failures should never interrupt the run loop.
            pass
