import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

from lmao.llm import LLMCallResult, LLMCallStats
from lmao.loop import run_agent_turn


class _FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls = 0

    def call(self, messages):  # type: ignore[no-untyped-def]
        content = self._replies[self.calls]
        self.calls += 1
        stats = LLMCallStats(
            elapsed_s=0.01,
            request_bytes=1,
            response_bytes=1,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            is_estimate=True,
        )
        return LLMCallResult(content=content, stats=stats)


class ThinkOnlyContinuationTests(TestCase):
    def test_think_only_turn_triggers_followup_call(self) -> None:
        client = _FakeClient(
            replies=[
                '{"type":"assistant_turn","version":"1","steps":[{"type":"think","content":"plan"}]}',
                '{"type":"assistant_turn","version":"1","steps":[{"type":"message","purpose":"final","content":"ok"},{"type":"end"}]}',
            ]
        )
        base = Path(".").resolve()
        messages = [{"role": "system", "content": "sys"}]

        with redirect_stdout(io.StringIO()):
            next_turn, stats, ended = run_agent_turn(
                messages=messages,
                client=client,  # type: ignore[arg-type]
                turn=1,
                last_user="hi",
                base=base,
                extra_roots=[],
                skill_roots=[],
                max_tool_output=(0, 0),
                yolo_enabled=False,
                read_only=False,
                allowed_tools=[],
                plugin_tools={},
                show_stats=False,
                debug_logger=None,
            )

        self.assertTrue(ended)
        self.assertEqual(2, client.calls)
        self.assertEqual(3, next_turn)
        self.assertIsNotNone(stats)
