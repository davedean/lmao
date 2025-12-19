from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .hooks import LoggingHookContext, LoggingHookTypes, get_global_hook_registry

@dataclass
class DebugLogger:
    """Minimal file logger for capturing debug traces without altering behavior."""

    path: Path

    def log(self, event: str, detail: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        registry = get_global_hook_registry()
        context = LoggingHookContext(
            hook_type=LoggingHookTypes.ON_LOG_MESSAGE,
            runtime_state={"event": event, "detail": detail, "timestamp": timestamp},
            log_level="debug",
            log_message=detail,
            log_source=event,
        )
        on_log_result = registry.execute_hooks(LoggingHookTypes.ON_LOG_MESSAGE, context)
        if on_log_result.should_skip:
            return
        if on_log_result.modified_context:
            context = on_log_result.modified_context
        pre_result = registry.execute_hooks(LoggingHookTypes.PRE_LOG_WRITE, context)
        if pre_result.should_skip:
            return
        if pre_result.modified_context:
            context = pre_result.modified_context
        entry = f"{timestamp} [{event}] {context.log_message}".rstrip("\n") + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(entry)
        except Exception:
            # Logging failures should never interrupt the run loop.
            pass
        registry.execute_hooks(LoggingHookTypes.POST_LOG_WRITE, context)
