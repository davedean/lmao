import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

from lmao.llm import LLMCallResult, LLMCallStats
from lmao.loop import ACTION_REQUIRED_PREFIX, run_agent_turn
from lmao.runtime_tools import RuntimeContext
from lmao.task_list import TaskListManager


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


class HeadlessClarificationTests(TestCase):
    def test_clarification_rejected_in_headless_mode(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()

        client = _FakeClient(
            [
                '{"type":"assistant_turn","version":"1","steps":[{"type":"message","purpose":"clarification","content":"Need more info"}]}',
                '{"type":"assistant_turn","version":"1","steps":[{"type":"message","purpose":"final","content":"done"},{"type":"end"}]}',
            ]
        )

        task_manager = TaskListManager()
        task_manager.new_list()

        runtime_ctx = RuntimeContext(
            client=client,
            plugin_tools={},
            base=base,
            extra_roots=(),
            skill_roots=(),
            yolo_enabled=False,
            read_only=False,
            headless=True,
            task_manager=task_manager,
            debug_logger=None,
        )

        messages = [{"role": "system", "content": "sys"}]
        with redirect_stdout(io.StringIO()):
            next_turn, stats, ended = run_agent_turn(
                messages=messages,
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
                task_manager=task_manager,
                show_stats=False,
                debug_logger=None,
            )

        self.assertTrue(ended)
        self.assertEqual(2, client.calls)
        self.assertIsNotNone(stats)
        self.assertTrue(next_turn >= 2)
