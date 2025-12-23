import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

from lmao.llm import LLMCallResult, LLMCallStats
from lmao.loop import run_agent_turn
from lmao.runtime_tools import RuntimeContext


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


class HeadlessImplicitInputRequestTests(TestCase):
    def test_headless_blocks_final_question_even_with_end(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()

        client = _FakeClient(
            [
                (
                    '{"type":"assistant_turn","version":"2","steps":['
                    '{"type":"message","purpose":"final","format":"markdown","content":"I can proceed. Would you like me to use option A or B?"},'
                    '{"type":"end","reason":"completed"}'
                    "]}"
                ),
                (
                    '{"type":"assistant_turn","version":"2","steps":['
                    '{"type":"message","purpose":"final","format":"markdown","content":"Chose option A by default. Done."},'
                    '{"type":"end","reason":"completed"}'
                    "]}"
                ),
            ]
        )

        runtime_ctx = RuntimeContext(
            client=client,
            plugin_tools={},
            base=base,
            extra_roots=(),
            skill_roots=(),
            yolo_enabled=False,
            read_only=False,
            headless=True,
            debug_logger=None,
        )

        with redirect_stdout(io.StringIO()):
            _, _, ended = run_agent_turn(
                messages=[{"role": "system", "content": "sys"}],
                client=client,  # type: ignore[arg-type]
                turn=1,
                last_user="headless test",
                base=base,
                extra_roots=(),
                skill_roots=(),
                max_tool_output=(0, 0),
                yolo_enabled=False,
                read_only=False,
                allowed_tools=[],
                plugin_tools={},
                runtime_tools={},
                runtime_context=runtime_ctx,
                show_stats=False,
                quiet=False,
                no_tools=False,
                debug_logger=None,
            )

        self.assertTrue(ended)
        self.assertEqual(2, client.calls)

    def test_headless_ignores_quoted_questions_in_tables(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()

        client = _FakeClient(
            [
                (
                    '{"type":"assistant_turn","version":"2","steps":['
                    '{"type":"message","purpose":"final","format":"markdown","content":"'
                    '| Sender | Message |\\n'
                    '| :--- | :--- |\\n'
                    '| **pr-agent** | \\"What would you like to know?\\" |\\n'
                    '\\nSummary: no action needed."},'
                    '{"type":"end","reason":"completed"}'
                    "]}"
                ),
            ]
        )

        runtime_ctx = RuntimeContext(
            client=client,
            plugin_tools={},
            base=base,
            extra_roots=(),
            skill_roots=(),
            yolo_enabled=False,
            read_only=False,
            headless=True,
            debug_logger=None,
        )

        with redirect_stdout(io.StringIO()):
            _, _, ended = run_agent_turn(
                messages=[{"role": "system", "content": "sys"}],
                client=client,  # type: ignore[arg-type]
                turn=1,
                last_user="headless test",
                base=base,
                extra_roots=(),
                skill_roots=(),
                max_tool_output=(0, 0),
                yolo_enabled=False,
                read_only=False,
                allowed_tools=[],
                plugin_tools={},
                runtime_tools={},
                runtime_context=runtime_ctx,
                show_stats=False,
                quiet=False,
                no_tools=False,
                debug_logger=None,
            )

        self.assertTrue(ended)
        self.assertEqual(1, client.calls)
