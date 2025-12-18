import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

from lmao.llm import LLMCallResult, LLMCallStats
from lmao.loop import run_loop


class _CapturingClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0
        self.last_messages = None

    def call(self, messages):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.last_messages = messages
        stats = LLMCallStats(
            elapsed_s=0.01,
            request_bytes=1,
            response_bytes=1,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            is_estimate=True,
        )
        return LLMCallResult(content=self.reply, stats=stats)


class StartupPreludeTests(TestCase):
    def test_startup_prelude_includes_read_agents_and_skill_guide(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        (base / "AGENTS.md").write_text("repo instructions", encoding="utf-8")

        built_in_plugins_dir = Path(__file__).resolve().parents[1] / "lmao" / "tools"
        client = _CapturingClient(
            '{"type":"assistant_turn","version":"1","steps":[{"type":"message","purpose":"final","content":"done"},{"type":"end"}]}'
        )

        with redirect_stdout(io.StringIO()):
            run_loop(
                initial_prompt="do the thing",
                client=client,  # type: ignore[arg-type]
                workdir=base,
                max_tool_output=(0, 0),
                max_turns=3,
                silent_tools=True,
                yolo_enabled=False,
                read_only=False,
                show_stats=False,
                headless=True,
                multiline=False,
                plugin_dirs=[built_in_plugins_dir],
                debug_logger=None,
            )

        self.assertEqual(client.calls, 1)
        self.assertIsNotNone(client.last_messages)
        joined = "\n".join(msg.get("content", "") for msg in client.last_messages)  # type: ignore[union-attr]
        self.assertIn("Tool result for tool 'read_agents' on ''", joined)
        self.assertIn("repo instructions", joined)
        self.assertIn("Tool result for tool 'skill_guide' on ''", joined)

