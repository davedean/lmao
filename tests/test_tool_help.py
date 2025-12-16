import json
import tempfile
from pathlib import Path
from unittest import TestCase

from lmao.plugins import discover_plugins
from lmao.tools import ToolCall, run_tool


class ToolHelpTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        tools_dir = Path(__file__).resolve().parent.parent / "lmao" / "tools"
        self.plugins = discover_plugins([tools_dir], self.base, allow_outside_base=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_tool_help_returns_usage_and_details(self) -> None:
        call = ToolCall(tool="tool_help", target="", args="async_tail")
        result = run_tool(
            call,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=True,
            plugin_tools=self.plugins,
        )
        payload = json.loads(result)
        self.assertTrue(payload["success"])
        data = payload["data"]
        self.assertEqual("async_tail", data["name"])
        self.assertIsInstance(data["usage"], list)
        self.assertIsInstance(data["details"], list)

