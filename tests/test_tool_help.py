import json
import tempfile
from pathlib import Path
from unittest import TestCase

from lmao.plugins import discover_plugins
from lmao.tools import ToolCall, run_tool


class ToolsGuideTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        tools_dir = Path(__file__).resolve().parent.parent / "lmao" / "tools"
        self.plugins = discover_plugins([tools_dir], self.base, allow_outside_base=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_tools_guide_returns_usage_and_details(self) -> None:
        call = ToolCall(tool="tools_guide", target="", args="async_tail")
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

    def test_tools_guide_multi_returns_mixed_results(self) -> None:
        call = ToolCall(tool="tools_guide", target="", args={"tools": ["async_tail", "not_a_tool"]})
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
        self.assertIn("tools", data)
        results = data["tools"]
        self.assertEqual(2, len(results))
        ok = next(item for item in results if item["name"] == "async_tail")
        bad = next(item for item in results if item["name"] == "not_a_tool")
        self.assertTrue(ok["success"])
        self.assertIn("data", ok)
        self.assertFalse(bad["success"])
        self.assertIn("error", bad)

    def test_tools_guide_unknown_list_only_is_concise(self) -> None:
        call = ToolCall(tool="tools_guide", target="", args={"tool": "not_a_tool", "list_only": True})
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
        self.assertFalse(payload["success"])
        error = payload["error"]
        self.assertIn("unknown tool", error)
        self.assertNotIn("known:", error)
