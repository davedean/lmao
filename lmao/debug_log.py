from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class DebugLogger:
    """Minimal file logger for capturing debug traces without altering behavior."""

    path: Path

    def log(self, event: str, detail: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        entry = f"{timestamp} [{event}] {detail}".rstrip("\n") + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(entry)
        except Exception:
            # Logging failures should never interrupt the run loop.
            pass
