from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .debug_log import DebugLogger


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", flags=re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Best-effort token estimate without external tokenizers."""
    if not text:
        return 0
    return len(_TOKEN_PATTERN.findall(text))


def estimate_message_tokens(messages: List[Dict[str, str]]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(str(message.get("role", "")))
        total += estimate_tokens(str(message.get("content", "")))
        total += 4  # rough per-message overhead
    return total


def _parse_chat_completion(body: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    parsed = json.loads(body)
    content = parsed["choices"][0]["message"]["content"]
    usage = parsed.get("usage")
    return content, usage if isinstance(usage, dict) else None


@dataclass
class LLMCallStats:
    elapsed_s: float
    request_bytes: int
    response_bytes: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    is_estimate: bool


@dataclass
class LLMCallResult:
    content: str
    stats: LLMCallStats


@dataclass
class LLMClient:
    """Lightweight client wrapper for LM Studio chat completions."""

    endpoint: str
    model: str
    temperature: float = 0.2
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    debug_logger: Optional[DebugLogger] = None

    def call(self, messages: List[Dict[str, str]]) -> LLMCallResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        prompt_tokens_est = estimate_message_tokens(messages)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers={"Content-Type": "application/json"})

        if self.debug_logger:
            self.debug_logger.log("llm.request", json.dumps(payload, ensure_ascii=False))

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw_body = resp.read()
        except urllib.error.HTTPError as exc:  # pragma: no cover - network error
            detail = exc.read().decode("utf-8", errors="ignore")
            if self.debug_logger:
                self.debug_logger.log("llm.error", f"HTTP {exc.code}: {detail}")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network error
            if self.debug_logger:
                self.debug_logger.log("llm.error", f"URL error: {exc}")
            raise RuntimeError(f"Failed to reach model endpoint: {exc}") from exc
        elapsed_s = time.perf_counter() - start
        body = raw_body.decode("utf-8", errors="replace")

        try:
            content, usage = _parse_chat_completion(body)
            completion_tokens_est = estimate_tokens(content)

            prompt_tokens = prompt_tokens_est
            completion_tokens = completion_tokens_est
            total_tokens = prompt_tokens + completion_tokens
            is_estimate = True
            if usage:
                try:
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens_est))
                    completion_tokens = int(usage.get("completion_tokens", completion_tokens_est))
                    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
                    is_estimate = False
                except Exception:
                    prompt_tokens = prompt_tokens_est
                    completion_tokens = completion_tokens_est
                    total_tokens = prompt_tokens + completion_tokens
                    is_estimate = True

            stats = LLMCallStats(
                elapsed_s=elapsed_s,
                request_bytes=len(data),
                response_bytes=len(raw_body),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                is_estimate=is_estimate,
            )
            if self.debug_logger:
                self.debug_logger.log("llm.response", content)
                self.debug_logger.log(
                    "llm.stats",
                    f"elapsed_s={elapsed_s:.3f} prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} total_tokens={total_tokens} estimate={is_estimate}",
                )
            return LLMCallResult(content=content, stats=stats)
        except Exception as exc:  # pragma: no cover - defensive parsing
            if self.debug_logger:
                self.debug_logger.log("llm.error", f"Unexpected response: {body}")
            raise RuntimeError(f"Unexpected response: {body}") from exc
