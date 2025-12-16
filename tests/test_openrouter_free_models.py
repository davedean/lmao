import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
from unittest import TestCase
from unittest.mock import patch

from lmao.openrouter_free_models import (
    OpenRouterFreeModelPreferences,
    OpenRouterFreeModelSelector,
    OpenRouterModelCandidate,
    OpenRouterModelDiscovery,
    OpenRouterModelSelectionError,
)


class _DummyDiscovery:
    def __init__(self, candidates: Sequence[OpenRouterModelCandidate]) -> None:
        self._candidates = list(candidates)

    def fetch_free_models(self) -> Sequence[OpenRouterModelCandidate]:
        return list(self._candidates)


def _build_candidate(
    model_id: str,
    *,
    parameter_estimate: int = 1_000_000_000,
    context_length: int = 4096,
    architecture: str = "gpt-4",
    pricing_input: Optional[float] = 0.0,
    pricing_output: Optional[float] = 0.0,
    modalities: tuple[str, ...] = ("text",),
    tag_is_free: Optional[bool] = None,
) -> OpenRouterModelCandidate:
    return OpenRouterModelCandidate(
        model_id=model_id,
        context_length=context_length,
        parameter_estimate=parameter_estimate,
        architecture=architecture,
        family=None,
        pricing_input=pricing_input,
        pricing_output=pricing_output,
        status="available",
        released_at=datetime.now(),
        modalities=modalities,
        tag_is_free=_detect_free_tag(model_id) if tag_is_free is None else tag_is_free,
    )


def _detect_free_tag(model_id: str) -> bool:
    lower = model_id.lower()
    return ":free" in lower or lower.endswith(" free")


class OpenRouterFreeModelsTests(TestCase):
    def test_selector_prefers_default_and_respects_blacklist(self) -> None:
        candidates = [
            _build_candidate("openrouter/free-a"),
            _build_candidate("openrouter/free-b", parameter_estimate=2_000_000_000),
            _build_candidate("openrouter/free-c", parameter_estimate=3_000_000_000),
        ]
        with patch.object(
            OpenRouterFreeModelSelector, "_validate_candidate", return_value=(True, "")
        ):
            selector = OpenRouterFreeModelSelector(
                discovery=_DummyDiscovery(candidates),
                preferences=OpenRouterFreeModelPreferences(
                    default_model="openrouter/free-b",
                    blacklist=("openrouter/free-c",),
                ),
                completions_endpoint="https://openrouter.ai/api/v1/chat/completions",
                api_key="secret",
            )
            chosen = selector.select_model()
        self.assertEqual("openrouter/free-b", chosen.model_id)

    def test_selector_error_when_all_blacklisted(self) -> None:
        candidates = [_build_candidate("model-one")]
        selector = OpenRouterFreeModelSelector(
            discovery=_DummyDiscovery(candidates),
            preferences=OpenRouterFreeModelPreferences(blacklist=("model-one",)),
            completions_endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key="secret",
        )
        with self.assertRaises(OpenRouterModelSelectionError):
            selector.select_model()

    def test_selector_skips_audio_only_models(self) -> None:
        candidates = [
            _build_candidate("model-audio", modalities=("audio",)),
            _build_candidate("model-text"),
        ]
        with patch.object(
            OpenRouterFreeModelSelector, "_validate_candidate", return_value=(True, "")
        ):
            selector = OpenRouterFreeModelSelector(
                discovery=_DummyDiscovery(candidates),
                preferences=OpenRouterFreeModelPreferences(),
                completions_endpoint="https://openrouter.ai/api/v1/chat/completions",
                api_key="secret",
            )
            chosen = selector.select_model()
        self.assertEqual("model-text", chosen.model_id)

    def test_selector_retries_on_validation_failures(self) -> None:
        candidates = [
            _build_candidate("model-one"),
            _build_candidate("model-two"),
        ]
        side_effects = [(False, "timeout"), (True, "")]
        with patch.object(
            OpenRouterFreeModelSelector,
            "_validate_candidate",
            side_effect=side_effects,
        ):
            selector = OpenRouterFreeModelSelector(
                discovery=_DummyDiscovery(candidates),
                preferences=OpenRouterFreeModelPreferences(),
                completions_endpoint="https://openrouter.ai/api/v1/chat/completions",
                api_key="secret",
            )
            chosen = selector.select_model()
            self.assertEqual("model-two", chosen.model_id)

    def test_pricing_none_is_rejected(self) -> None:
        discovery = OpenRouterModelDiscovery(
            models_endpoint="https://openrouter.ai/api/v1/models",
            api_key="secret",
            cache_path=Path(tempfile.gettempdir()) / "openrouter-pricing-test.json",
        )
        candidate = _build_candidate("model-missing-pricing", pricing_input=None, pricing_output=None)
        self.assertFalse(discovery._is_free(candidate))

    def test_zero_pricing_without_free_tag_filtered(self) -> None:
        discovery = OpenRouterModelDiscovery(
            models_endpoint="https://openrouter.ai/api/v1/models",
            api_key="secret",
            cache_path=Path(tempfile.gettempdir()) / "openrouter-tag-test.json",
        )
        candidate = _build_candidate(
            "openai/gpt-4.1",
            pricing_input=0.0,
            pricing_output=0.0,
            modalities=("text",),
        )
        self.assertFalse(discovery._is_free(candidate))

    def test_discovery_caches_and_filters(self) -> None:
        payload = {
            "data": [
                {
                    "id": "free-model:free",
                    "pricing": {"input": 0, "output": 0},
                    "context_length": 4096,
                    "parameters": 1_000_000_000,
                },
                {
                    "id": "paid-model",
                    "pricing": {"input": 0.05, "output": 0.05},
                    "context_length": 2048,
                },
            ]
        }

        class FakeResponse:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, exc_type, exc_value, traceback) -> bool:
                return False

        cache_path = Path(tempfile.gettempdir()) / "openrouter_free_models_test.json"
        if cache_path.exists():
            cache_path.unlink()

        with patch(
            "lmao.openrouter_free_models.urlopen",
            return_value=FakeResponse(json.dumps(payload).encode("utf-8")),
        ) as mocked_urlopen:
            discovery = OpenRouterModelDiscovery(
                models_endpoint="https://openrouter.ai/api/v1/models",
                api_key="secret",
                cache_path=cache_path,
                ttl_seconds=3600,
            )
            try:
                first_batch = discovery.fetch_free_models()
                self.assertEqual(1, len(first_batch))
                self.assertEqual("free-model:free", first_batch[0].model_id)
                self.assertTrue(cache_path.exists())
                second_batch = discovery.fetch_free_models()
                self.assertEqual(1, len(second_batch))
                self.assertEqual(1, mocked_urlopen.call_count)
            finally:
                cache_path.unlink(missing_ok=True)
