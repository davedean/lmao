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


class EndRequiresSummaryTests(TestCase):
    def test_end_blocked_if_tool_ran_since_last_message(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()

        tools_dir = Path(__file__).resolve().parent.parent / "lmao" / "tools"
        plugins = discover_plugins([tools_dir], base, allow_outside_base=True)

        task_manager = TaskListManager()
        task_manager.new_list()

        (base / "a.txt").write_text("x\ny\nz\nline4\n", encoding="utf-8")

        client = _FakeClient(
            replies=[
                # 1) Tool call (read) only.
                '{"type":"assistant_turn","version":"1","steps":[{"type":"tool_call","call":{"tool":"read","target":"a.txt","args":"lines:4"}}]}',
                # 2) Try to end without final message -> blocked.
                '{"type":"assistant_turn","version":"1","steps":[{"type":"end"}]}',
                # 3) Provide final message then end.
                '{"type":"assistant_turn","version":"1","steps":[{"type":"message","purpose":"final","content":"done"},{"type":"end"}]}',
            ]
        )

        allowed_tools = ["read"]

        with redirect_stdout(io.StringIO()):
            _, _, ended = run_agent_turn(
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
        self.assertEqual(3, client.calls)

