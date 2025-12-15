import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

from lmao.llm import LLMCallResult, LLMCallStats
from lmao.loop import run_agent_turn
from lmao.plugins import discover_plugins
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


class WithheldMessageEndBlockTests(TestCase):
    def test_end_is_blocked_until_message_resent(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()

        tools_dir = Path(__file__).resolve().parent.parent / "lmao" / "tools"
        plugins = discover_plugins([tools_dir], base, allow_outside_base=True)

        task_manager = TaskListManager()
        task_manager.new_list()
        task_manager.add_task("tell joke")

        client = _FakeClient(
            replies=[
                # 1) Try to message while tasks incomplete -> withheld
                '{"type":"assistant_turn","version":"1","steps":[{"type":"message","purpose":"progress","content":"joke"}]}',
                # 2) Complete the task
                '{"type":"assistant_turn","version":"1","steps":[{"type":"tool_call","call":{"tool":"complete_task","target":"","args":"1"}}]}',
                # 3) Try to end without resending -> must be blocked
                '{"type":"assistant_turn","version":"1","steps":[{"type":"end","reason":"completed"}]}',
                # 4) Resend message then end -> allowed
                '{"type":"assistant_turn","version":"1","steps":[{"type":"message","purpose":"final","content":"joke"},{"type":"end"}]}',
            ]
        )

        allowed_tools = ["complete_task"]

        with redirect_stdout(io.StringIO()):
            next_turn, stats, ended = run_agent_turn(
                messages=[{"role": "system", "content": "sys"}],
                client=client,  # type: ignore[arg-type]
                turn=1,
                last_user="hi",
                base=base,
                extra_roots=[],
                skill_roots=[],
                max_tool_output=(0, 0),
                yolo_enabled=False,
                read_only=False,
                allowed_tools=allowed_tools,
                plugin_tools=plugins,
                task_manager=task_manager,
                show_stats=False,
                debug_logger=None,
            )

        self.assertTrue(ended)
        self.assertEqual(4, client.calls)
        self.assertIsNotNone(stats)
        self.assertGreaterEqual(next_turn, 2)

