import json
import tempfile
from pathlib import Path
from unittest import TestCase

from lmao.plugins import discover_plugins
from lmao.tools import ToolCall, run_tool


class AsyncToolsTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        tools_dir = Path(__file__).resolve().parent.parent / "lmao" / "tools"
        self.plugins = discover_plugins([tools_dir], self.base, allow_outside_base=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_async_tail_and_poll(self) -> None:
        log_path = self.base / "app.log"
        log_path.write_text("", encoding="utf-8")

        start = ToolCall(tool="async_tail", target="app.log", args="start_at=start")
        start_result = run_tool(
            start,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=True,
            plugin_tools=self.plugins,
        )
        start_payload = json.loads(start_result)
        self.assertTrue(start_payload["success"])
        job_id = start_payload["data"]["job_id"]

        log_path.write_text("line1\nline2\n", encoding="utf-8")

        poll = ToolCall(tool="async_poll", target="", args=f"{job_id} 0")
        poll_result = run_tool(
            poll,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=True,
            plugin_tools=self.plugins,
        )
        poll_payload = json.loads(poll_result)
        self.assertTrue(poll_payload["success"])
        events = poll_payload["data"]["events"]
        texts = [e["text"] for e in events if e["stream"] == "tail"]
        self.assertIn("line1", texts)
        self.assertIn("line2", texts)

    def test_async_poll_and_stop_accept_target_when_args_empty(self) -> None:
        log_path = self.base / "app.log"
        log_path.write_text("hello\n", encoding="utf-8")

        start = ToolCall(tool="async_tail", target="app.log", args="start_at=start")
        start_result = run_tool(
            start,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=True,
            plugin_tools=self.plugins,
        )
        job_id = json.loads(start_result)["data"]["job_id"]

        poll = ToolCall(tool="async_poll", target=job_id, args="")
        poll_result = run_tool(
            poll,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=True,
            plugin_tools=self.plugins,
        )
        poll_payload = json.loads(poll_result)
        self.assertTrue(poll_payload["success"])

        stop = ToolCall(tool="async_stop", target=job_id, args="")
        stop_result = run_tool(
            stop,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=False,
            plugin_tools=self.plugins,
        )
        stop_payload = json.loads(stop_result)
        self.assertFalse(stop_payload["success"])
        self.assertIn("not approved", stop_payload["error"])

    def test_async_poll_ignores_non_numeric_args_when_target_is_job_id(self) -> None:
        log_path = self.base / "app.log"
        log_path.write_text("hello\n", encoding="utf-8")

        start = ToolCall(tool="async_tail", target="app.log", args="start_at=start")
        start_result = run_tool(
            start,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=True,
            plugin_tools=self.plugins,
        )
        job_id = json.loads(start_result)["data"]["job_id"]

        poll = ToolCall(tool="async_poll", target=job_id, args="logs/test.log")
        poll_result = run_tool(
            poll,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=True,
            plugin_tools=self.plugins,
        )
        poll_payload = json.loads(poll_result)
        self.assertTrue(poll_payload["success"])
