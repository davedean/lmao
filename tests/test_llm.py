import json
from unittest import TestCase

from lmao.llm import _build_request_headers, _parse_chat_completion, estimate_message_tokens, estimate_tokens


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

    def test_parse_chat_completion_falls_back_to_text(self) -> None:
        body = json.dumps({"choices": [{"text": "hello"}]})
        content, usage = _parse_chat_completion(body)
        self.assertEqual("hello", content)
        self.assertIsNone(usage)

    def test_build_request_headers_openrouter_requires_key(self) -> None:
        with self.assertRaises(ValueError):
            _build_request_headers(
                provider="openrouter",
                api_key=None,
                openrouter_referer=None,
                openrouter_title=None,
            )

    def test_build_request_headers_openrouter_includes_expected_headers(self) -> None:
        headers = _build_request_headers(
            provider="openrouter",
            api_key="sk-test",
            openrouter_referer="https://example.com",
            openrouter_title="Example App",
        )
        self.assertEqual("application/json", headers["Content-Type"])
        self.assertEqual("Bearer sk-test", headers["Authorization"])
        self.assertEqual("https://example.com", headers["HTTP-Referer"])
        self.assertEqual("Example App", headers["X-Title"])
