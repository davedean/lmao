import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

from lmao.llm import LLMCallResult, LLMCallStats
from lmao.loop import run_loop


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


class RunLoopHeadlessContinueTests(TestCase):
    def test_headless_continues_until_end(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()

        client = _FakeClient(
            [
                '{"type":"assistant_turn","version":"1","steps":[{"type":"message","purpose":"progress","content":"status update"}]}',
                '{"type":"assistant_turn","version":"1","steps":[{"type":"message","purpose":"final","content":"done"},{"type":"end"}]}',
            ]
        )

        with redirect_stdout(io.StringIO()):
            run_loop(
                initial_prompt="do the thing",
                client=client,  # type: ignore[arg-type]
                workdir=base,
                max_tool_output=(0, 0),
                max_turns=5,
                silent_tools=True,
                yolo_enabled=False,
                read_only=False,
                show_stats=False,
                headless=True,
                multiline=False,
                plugin_dirs=[],
                debug_logger=None,
            )

        self.assertEqual(2, client.calls)

