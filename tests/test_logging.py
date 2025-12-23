import json
import tempfile
from pathlib import Path
from unittest import TestCase

from lmao.debug_log import DebugLogger
from lmao.error_log import ErrorLogger
from lmao.text_utils import summarize_tool_output


class DebugLoggerTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_debug_logger_writes_jsonl(self) -> None:
        log_path = self.base / "debug.log"
        logger = DebugLogger(log_path)
        logger.log("tool.call", "turn=1 tool=read")
        entry = json.loads(log_path.read_text(encoding="utf-8").strip())
        self.assertEqual("tool.call", entry["event"])
        self.assertEqual("debug", entry["level"])
        self.assertEqual("tool.call", entry["message"])
        self.assertEqual("turn=1 tool=read", entry["data"]["detail"])
        self.assertIn("ts", entry)

    def test_debug_logger_accepts_structured_payload(self) -> None:
        log_path = self.base / "debug.log"
        logger = DebugLogger(log_path)
        logger.log("tool.call", {"message": "tool call", "data": {"tool": "read"}, "level": "info"})
        entry = json.loads(log_path.read_text(encoding="utf-8").strip())
        self.assertEqual("tool.call", entry["event"])
        self.assertEqual("info", entry["level"])
        self.assertEqual("tool call", entry["message"])
        self.assertEqual("read", entry["data"]["tool"])


class ErrorLoggerTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_error_logger_writes_jsonl(self) -> None:
        log_path = self.base / "error.log"
        logger = ErrorLogger(log_path)
        logger.log("tool.failure", {"tool": "read", "error": "missing target file path"})
        entry = json.loads(log_path.read_text(encoding="utf-8").strip())
        self.assertEqual("tool.failure", entry["event"])
        self.assertEqual("error", entry["level"])
        self.assertEqual("tool.failure", entry["message"])
        self.assertEqual("read", entry["data"]["tool"])


class ToolSummaryTests(TestCase):
    def test_tool_summary_success(self) -> None:
        output = json.dumps(
            {
                "tool": "read",
                "success": True,
                "data": {
                    "path": "README.md",
                    "content": "abc",
                    "lines": {"start": 1, "end": 2},
                    "truncated": False,
                    "limit_chars": 200000,
                },
            }
        )
        summary = summarize_tool_output(output, max_lines=1, max_chars=200)
        self.assertIn("ok", summary)
        self.assertIn("path=README.md", summary)
        self.assertIn("lines=1-2", summary)
        self.assertIn("truncated=false", summary)
        self.assertIn("limit_chars=200000", summary)
        self.assertNotIn("abc", summary)

    def test_tool_summary_error(self) -> None:
        output = json.dumps(
            {
                "tool": "read",
                "success": False,
                "error": "missing target file path",
            }
        )
        summary = summarize_tool_output(output, max_lines=1, max_chars=200)
        self.assertIn("error", summary)
        self.assertIn("missing target file path", summary)

    def test_tool_summary_unparseable(self) -> None:
        summary = summarize_tool_output("not json", max_lines=1, max_chars=200)
        self.assertEqual("unparseable tool output", summary)
