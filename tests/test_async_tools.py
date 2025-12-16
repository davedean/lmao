import json
import tempfile
import time
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from lmao.plugins import discover_plugins
from lmao.tools import ToolCall, run_tool
from lmao.task_list import TaskListManager
from lmao.async_jobs import get_async_job_manager


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

    def test_async_tail_track_task_marks_complete_on_finish(self) -> None:
        tasks = TaskListManager()
        log_path = self.base / "app.log"
        log_path.write_text("", encoding="utf-8")

        start = ToolCall(tool="async_tail", target="app.log", args="start_at=end track_task=true")
        start_result = run_tool(
            start,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=True,
            plugin_tools=self.plugins,
            task_manager=tasks,
        )
        start_payload = json.loads(start_result)
        self.assertTrue(start_payload["success"])
        self.assertTrue(start_payload["data"]["task_added"])
        self.assertEqual(1, len(tasks.tasks))
        self.assertFalse(tasks.tasks[0].done)

        job_id = start_payload["data"]["job_id"]

        manager = get_async_job_manager()
        self.assertTrue(manager.stop(job_id))

        poll = ToolCall(tool="async_poll", target=job_id, args="0")
        poll_result = run_tool(
            poll,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=True,
            plugin_tools=self.plugins,
            task_manager=tasks,
        )
        poll_payload = json.loads(poll_result)
        self.assertTrue(poll_payload["success"])
        self.assertTrue(poll_payload["data"]["task_updated"])
        self.assertTrue(tasks.tasks[0].done)

    def test_async_bash_strips_trailing_track_task_token(self) -> None:
        tasks = TaskListManager()
        call = ToolCall(tool="async_bash", target="", args="python3 -c \"print('ok')\" track_task=true")
        with patch("builtins.input", return_value="y"):
            start_result = run_tool(
                call,
                base=self.base,
                extra_roots=[],
                skill_roots=[],
                yolo_enabled=False,
                read_only=False,
                plugin_tools=self.plugins,
                task_manager=tasks,
            )
        start_payload = json.loads(start_result)
        self.assertTrue(start_payload["success"])
        self.assertTrue(start_payload["data"]["task_added"])
        self.assertEqual("python3 -c \"print('ok')\"", start_payload["data"]["command"])

        job_id = start_payload["data"]["job_id"]
        for _ in range(50):
            poll = ToolCall(tool="async_poll", target=job_id, args="0")
            poll_result = run_tool(
                poll,
                base=self.base,
                extra_roots=[],
                skill_roots=[],
                yolo_enabled=False,
                read_only=True,
                plugin_tools=self.plugins,
                task_manager=tasks,
            )
            payload = json.loads(poll_result)
            self.assertTrue(payload["success"])
            status = payload["data"]["status"]
            if status in ("done", "error", "canceled"):
                self.assertIn(status, ("done", "error"))
                break
            time.sleep(0.01)
        else:
            self.fail("async_bash job did not finish in time")

    def test_async_bash_structured_args_and_meta(self) -> None:
        tasks = TaskListManager()
        call = ToolCall(
            tool="async_bash",
            target="",
            args={"command": "python3 -c \"print('ok')\""},
            meta={"track_task": True},
        )
        with patch("builtins.input", return_value="y"):
            start_result = run_tool(
                call,
                base=self.base,
                extra_roots=[],
                skill_roots=[],
                yolo_enabled=False,
                read_only=False,
                plugin_tools=self.plugins,
                task_manager=tasks,
            )
        start_payload = json.loads(start_result)
        self.assertTrue(start_payload["success"])
        self.assertTrue(start_payload["data"]["task_added"])
        self.assertEqual("python3 -c \"print('ok')\"", start_payload["data"]["command"])
