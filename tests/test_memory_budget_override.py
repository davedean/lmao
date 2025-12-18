from unittest import TestCase

from lmao.llm import LLMClient
from lmao.memory import determine_prompt_budget


class MemoryBudgetOverrideTests(TestCase):
    def test_lmstudio_context_window_override_changes_budget(self) -> None:
        client = LLMClient(
            endpoint="http://example",
            model="qwen",
            provider="lmstudio",
            max_tokens=100,
            context_window_tokens=2000,
        )
        max_prompt, trigger, target = determine_prompt_budget(client)
        self.assertEqual(1900, max_prompt)
        self.assertEqual(int(1900 * 0.7), trigger)
        self.assertEqual(int(1900 * 0.6), target)

    def test_openrouter_context_window_override_changes_budget(self) -> None:
        client = LLMClient(
            endpoint="http://example",
            model="qwen",
            provider="openrouter",
            max_tokens=200,
            context_window_tokens=5000,
        )
        max_prompt, trigger, target = determine_prompt_budget(client)
        self.assertEqual(4800, max_prompt)
        self.assertEqual(int(4800 * 0.7), trigger)
        self.assertEqual(int(4800 * 0.6), target)
