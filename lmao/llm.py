from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from .debug_log import DebugLogger


@dataclass
class LLMClient:
    """Lightweight client wrapper for LM Studio chat completions."""

    endpoint: str
    model: str
    temperature: float = 0.2
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    debug_logger: Optional[DebugLogger] = None

    def call(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers={"Content-Type": "application/json"})

        if self.debug_logger:
            self.debug_logger.log("llm.request", json.dumps(payload, ensure_ascii=False))

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # pragma: no cover - network error
            detail = exc.read().decode("utf-8", errors="ignore")
            if self.debug_logger:
                self.debug_logger.log("llm.error", f"HTTP {exc.code}: {detail}")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network error
            if self.debug_logger:
                self.debug_logger.log("llm.error", f"URL error: {exc}")
            raise RuntimeError(f"Failed to reach model endpoint: {exc}") from exc

        try:
            parsed = json.loads(body)
            content = parsed["choices"][0]["message"]["content"]
            if self.debug_logger:
                self.debug_logger.log("llm.response", content)
            return content
        except Exception as exc:  # pragma: no cover - defensive parsing
            if self.debug_logger:
                self.debug_logger.log("llm.error", f"Unexpected response: {body}")
            raise RuntimeError(f"Unexpected response: {body}") from exc
