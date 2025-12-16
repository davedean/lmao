from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import ParseResult, urlparse, urlunparse
from urllib.request import Request, urlopen

from .config import resolve_default_config_path
from .llm import LLMClient


class OpenRouterModelSelectionError(RuntimeError):
    """Raised when the automatic free-model selection cannot pick a valid model."""


@dataclass(frozen=True)
class OpenRouterFreeModelPreferences:
    """User-configurable hints for the free-model selector."""

    default_model: Optional[str] = None
    blacklist: Sequence[str] = ()

    def normalized_blacklist(self) -> set[str]:
        return {entry.strip().lower() for entry in self.blacklist if entry and entry.strip()}


@dataclass(frozen=True)
class OpenRouterModelCandidate:
    """Metadata representing a free OpenRouter model."""

    model_id: str
    context_length: Optional[int]
    parameter_estimate: Optional[int]
    architecture: Optional[str]
    family: Optional[str]
    pricing_input: Optional[float]
    pricing_output: Optional[float]
    status: Optional[str]
    released_at: Optional[datetime]
    modalities: tuple[str, ...]

    @property
    def abbreviated_id(self) -> str:
        return self.model_id.split(":")[-1]

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> Optional["OpenRouterModelCandidate"]:
        model_id = payload.get("id") or payload.get("model_id")
        if not model_id:
            return None

        context_length = cls._parse_int(payload.get("context_length")) or cls._parse_int(
            payload.get("max_context")
        )
        parameter_estimate = cls._parse_int(payload.get("parameters")) or cls._parse_int(
            payload.get("parameter_count")
        )
        architecture = payload.get("architecture") or payload.get("family")
        family = payload.get("family")
        released_at = cls._parse_datetime(payload.get("released_at"))
        if not released_at:
            released_at = cls._parse_datetime(payload.get("created_at"))
        pricing_input = cls._parse_pricing(payload.get("pricing"), "input")
        pricing_output = cls._parse_pricing(payload.get("pricing"), "output")
        status = payload.get("status")
        modalities = cls._parse_modalities(payload) or ()
        return OpenRouterModelCandidate(
            model_id=model_id,
            context_length=context_length,
            parameter_estimate=parameter_estimate,
            architecture=architecture,
            family=family,
            pricing_input=pricing_input,
            pricing_output=pricing_output,
            status=status,
            released_at=released_at,
            modalities=tuple(modalities),
        )

    @classmethod
    def from_cache(cls, data: Mapping[str, Any]) -> Optional["OpenRouterModelCandidate"]:
        model_id = data.get("model_id")
        if not model_id:
            return None
        released_at = cls._parse_datetime(data.get("released_at"))
        modalities = tuple(data.get("modalities") or ())
        return OpenRouterModelCandidate(
            model_id=model_id,
            context_length=cls._parse_int(data.get("context_length")),
            parameter_estimate=cls._parse_int(data.get("parameter_estimate")),
            architecture=data.get("architecture"),
            family=data.get("family"),
            pricing_input=cls._parse_float(data.get("pricing_input")),
            pricing_output=cls._parse_float(data.get("pricing_output")),
            status=data.get("status"),
            released_at=released_at,
            modalities=modalities,
        )

    def to_cache(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "context_length": self.context_length,
            "parameter_estimate": self.parameter_estimate,
            "architecture": self.architecture,
            "family": self.family,
            "pricing_input": self.pricing_input,
            "pricing_output": self.pricing_output,
            "status": self.status,
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "modalities": list(self.modalities),
        }

    @staticmethod
    def _parse_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
                except ValueError:
                    return None
        return None

    @staticmethod
    def _parse_pricing(pricing: Any, channel: str) -> Optional[float]:
        if not pricing or not isinstance(pricing, Mapping):
            return None
        channel_info = pricing.get(channel)
        if channel_info is None:
            return None
        if isinstance(channel_info, Mapping):
            return OpenRouterModelCandidate._parse_float(channel_info.get("price"))
        return OpenRouterModelCandidate._parse_float(channel_info)

    @staticmethod
    def _parse_modalities(payload: Mapping[str, Any]) -> Optional[list[str]]:
        for key in ("modalities", "input_modalities"):
            raw = payload.get(key)
            if raw is None:
                continue
            if isinstance(raw, str):
                values = [entry.strip() for entry in raw.replace(",", " ").split() if entry.strip()]
                return values
            if isinstance(raw, Sequence):
                return [str(entry).strip() for entry in raw if entry]
        return None

    def accepts_text_input(self) -> bool:
        if not self.modalities:
            return True
        entries = {entry.lower() for entry in self.modalities}
        if "text" not in entries:
            return False
        for blocked in ("audio", "speech"):
            if blocked in entries:
                return False
        return True


def resolve_model_cache_path() -> Path:
    config_path = resolve_default_config_path()
    machine_dir = config_path.parent / "openrouter"
    machine_dir.mkdir(parents=True, exist_ok=True)
    return (machine_dir / "free_models.json").expanduser()


class OpenRouterModelDiscovery:
    """Fetches and caches metadata for OpenRouter's free models."""

    DEFAULT_TTL_SECONDS = 3600

    def __init__(
        self,
        *,
        models_endpoint: str,
        api_key: str,
        cache_path: Path,
        ttl_seconds: int | None = None,
        timeout_seconds: int = 10,
    ) -> None:
        self._models_endpoint = models_endpoint
        self._api_key = api_key
        self._cache_path = cache_path
        self._ttl = ttl_seconds or self.DEFAULT_TTL_SECONDS
        self._timeout = timeout_seconds

    def fetch_free_models(self) -> Sequence[OpenRouterModelCandidate]:
        cached = self._load_cache()
        if cached:
            return cached
        models = self._retrieve_models()
        self._write_cache(models)
        return models

    def _load_cache(self) -> Sequence[OpenRouterModelCandidate] | None:
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception:
            return None
        timestamp = raw.get("fetched_at")
        if not timestamp:
            return None
        try:
            fetched_at = datetime.fromisoformat(timestamp)
        except ValueError:
            return None
        if datetime.now() - fetched_at > timedelta(seconds=self._ttl):
            return None
        models: list[OpenRouterModelCandidate] = []
        for entry in raw.get("models", []):
            candidate = OpenRouterModelCandidate.from_cache(entry)
            if candidate:
                models.append(candidate)
        return models

    def _write_cache(self, models: Iterable[OpenRouterModelCandidate]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": datetime.now().isoformat(),
            "models": [model.to_cache() for model in models],
        }
        self._cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _retrieve_models(self) -> list[OpenRouterModelCandidate]:
        endpoint = self._models_endpoint
        if "free=true" not in endpoint:
            delimiter = "&" if "?" in endpoint else "?"
            endpoint = f"{endpoint}{delimiter}free=true"
        request = Request(endpoint, headers=self._build_headers())
        try:
            with urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise OpenRouterModelSelectionError(
                f"Failed to fetch OpenRouter models: {exc}"
            ) from exc
        except Exception as exc:
            raise OpenRouterModelSelectionError(
                f"Failed to parse OpenRouter models: {exc}"
            ) from exc

        models_data = payload.get("data") if isinstance(payload, Mapping) else payload
        if isinstance(models_data, Mapping):
            models_data = models_data.get("data")
        if not isinstance(models_data, list):
            raise OpenRouterModelSelectionError(
                "Unexpected response when fetching OpenRouter models."
            )
        candidates: list[OpenRouterModelCandidate] = []
        for raw_model in models_data:
            if not isinstance(raw_model, Mapping):
                continue
            candidate = OpenRouterModelCandidate.from_api(raw_model)
            if not candidate:
                continue
            if not self._is_free(candidate):
                continue
            if not candidate.accepts_text_input():
                continue
            candidates.append(candidate)
        if not candidates:
            raise OpenRouterModelSelectionError("No free OpenRouter models found.")
        return candidates

    def _is_free(self, candidate: OpenRouterModelCandidate) -> bool:
        input_price = candidate.pricing_input
        output_price = candidate.pricing_output
        if input_price is None and output_price is None:
            return False
        input_is_zero = input_price == 0.0 if input_price is not None else False
        output_is_zero = output_price == 0.0 if output_price is not None else False
        return input_is_zero and output_is_zero

    def _build_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}


class OpenRouterFreeModelSelector:
    """Scores, validates, and selects the best free OpenRouter model."""

    MAX_PARAMETER_ESTIMATE = 1_760_000_000_000
    MAX_CONTEXT_LENGTH = 32_000

    def __init__(
        self,
        *,
        discovery: OpenRouterModelDiscovery,
        preferences: OpenRouterFreeModelPreferences,
        completions_endpoint: str,
        api_key: str,
        openrouter_referer: Optional[str] = None,
        openrouter_title: Optional[str] = None,
        health_prompt: str = "Hello",
    ) -> None:
        self._discovery = discovery
        self._preferences = preferences
        self._completions_endpoint = completions_endpoint
        self._api_key = api_key
        self._openrouter_referer = openrouter_referer
        self._openrouter_title = openrouter_title
        self._health_prompt = health_prompt

    def select_model(self) -> OpenRouterModelCandidate:
        candidates = self._discovery.fetch_free_models()
        blacklist = self._preferences.normalized_blacklist()
        filtered = [
            candidate for candidate in candidates if candidate.model_id.lower() not in blacklist
        ]
        if not filtered:
            raise OpenRouterModelSelectionError("All free OpenRouter models are blacklisted.")

        default_candidate = self._find_default(filtered)
        ordered = self._order_candidates(filtered, default_candidate)
        validation_errors: list[str] = []
        for candidate in ordered:
            if not candidate.accepts_text_input():
                validation_errors.append(f"{candidate.model_id}: not text-input")
                continue
            valid, reason = self._validate_candidate(candidate)
            if valid:
                return candidate
            validation_errors.append(f"{candidate.model_id}: {reason}")

        raise OpenRouterModelSelectionError(
            "No working text-based free OpenRouter model found. "
            + "; ".join(validation_errors)
        )

    def _order_candidates(
        self,
        candidates: Sequence[OpenRouterModelCandidate],
        default_candidate: Optional[OpenRouterModelCandidate],
    ) -> list[OpenRouterModelCandidate]:
        seen: set[str] = set()
        ordered: list[OpenRouterModelCandidate] = []
        if default_candidate:
            ordered.append(default_candidate)
            seen.add(default_candidate.model_id)
        scored = sorted(candidates, key=self._score_candidate, reverse=True)
        for candidate in scored:
            if candidate.model_id in seen:
                continue
            ordered.append(candidate)
        return ordered

    def _validate_candidate(
        self, candidate: OpenRouterModelCandidate
    ) -> tuple[bool, str]:
        client = LLMClient(
            endpoint=self._completions_endpoint,
            model=candidate.model_id,
            provider="openrouter",
            api_key=self._api_key,
            openrouter_referer=self._openrouter_referer,
            openrouter_title=self._openrouter_title,
            temperature=0.1,
            max_tokens=16,
        )
        try:
            client.call([
                {"role": "user", "content": self._health_prompt},
            ])
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _find_default(
        self, candidates: Sequence[OpenRouterModelCandidate]
    ) -> Optional[OpenRouterModelCandidate]:
        default_id = self._preferences.default_model
        if not default_id:
            return None
        for candidate in candidates:
            if candidate.model_id.lower() == default_id.lower():
                return candidate
        return None

    def _score_candidate(self, candidate: OpenRouterModelCandidate) -> float:
        capability = self._capability_score(candidate)
        availability = self._availability_score(candidate)
        return 0.7 * capability + 0.3 * availability

    def _capability_score(self, candidate: OpenRouterModelCandidate) -> float:
        param_score = self._normalize(
            candidate.parameter_estimate, self.MAX_PARAMETER_ESTIMATE, default=0.5
        )
        context_score = self._normalize(
            candidate.context_length, self.MAX_CONTEXT_LENGTH, default=0.5
        )
        architecture_score = self._architecture_score(candidate)
        recency_score = self._recency_score(candidate)
        total = (
            0.4 * param_score
            + 0.3 * context_score
            + 0.2 * (architecture_score / 100)
            + 0.1 * recency_score
        )
        return total * 100

    def _availability_score(self, candidate: OpenRouterModelCandidate) -> float:
        base = 1.0
        if candidate.status and candidate.status.lower() in {
            "unavailable",
            "offline",
            "deprecated",
        }:
            base = 0.4
        return base * 100

    def _normalize(
        self, value: Optional[int], maximum: int, default: float = 0.5
    ) -> float:
        if value is None or value <= 0:
            return default
        return min(value / maximum, 1.0)

    def _architecture_score(self, candidate: OpenRouterModelCandidate) -> float:
        identifier = candidate.model_id.lower()
        if "gpt-4" in identifier or "gpt4" in identifier:
            return 100.0
        if "claude" in identifier:
            return 90.0
        if "gpt-3.5" in identifier or "gpt35" in identifier:
            return 70.0
        return 50.0

    def _recency_score(self, candidate: OpenRouterModelCandidate) -> float:
        if not candidate.released_at:
            return 0.5
        now = datetime.now()
        age = now - candidate.released_at
        if age <= timedelta(days=14):
            return 1.0
        if age <= timedelta(days=90):
            return 0.8
        if age <= timedelta(days=365):
            return 0.6
        return 0.4


def derive_models_endpoint(completions_endpoint: str) -> str:
    parsed = urlparse(completions_endpoint)
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")] + "/models"
    else:
        path = path + "/models"
    rebuilt = ParseResult(
        scheme=parsed.scheme,
        netloc=parsed.netloc,
        path=path,
        params="",
        query="",
        fragment="",
    )
    return urlunparse(rebuilt)
