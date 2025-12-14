import json
from unittest import TestCase

from lmao.llm import _parse_chat_completion, estimate_message_tokens, estimate_tokens


class LLMHelpersTests(TestCase):
    def test_estimate_tokens_basic(self) -> None:
        self.assertEqual(0, estimate_tokens(""))
        self.assertGreaterEqual(estimate_tokens("hello world"), 2)
        self.assertGreaterEqual(estimate_tokens("a,b"), 3)

    def test_estimate_message_tokens_counts_content(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hello"},
        ]
        self.assertGreater(estimate_message_tokens(messages), 0)

    def test_parse_chat_completion_extracts_content_and_usage(self) -> None:
        body = json.dumps(
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            }
        )
        content, usage = _parse_chat_completion(body)
        self.assertEqual("hi", content)
        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(10, usage["prompt_tokens"])

