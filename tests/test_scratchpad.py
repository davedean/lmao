import json
import tempfile
from pathlib import Path
from unittest import TestCase

from lmao.plugins import discover_plugins
from lmao.tools import ToolCall, run_tool


class ScratchpadToolTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        tools_dir = Path(__file__).resolve().parent.parent / "lmao" / "tools"
        self.plugins = discover_plugins([tools_dir], self.base, allow_outside_base=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_scratchpad_write_and_read(self) -> None:
        write_call = ToolCall(tool="scratchpad", target="", args=json.dumps({"action": "write", "content": "Hello, world!"}))
        write_output = run_tool(write_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        write_payload = json.loads(write_output)
        self.assertTrue(write_payload["success"])
        self.assertEqual("Content written to scratchpad.", write_payload["data"]["message"])

        read_call = ToolCall(tool="scratchpad", target="", args=json.dumps({"action": "read"}))
        read_output = run_tool(read_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        read_payload = json.loads(read_output)
        self.assertTrue(read_payload["success"])
        self.assertEqual("Hello, world!", read_payload["data"]["content"])

    def test_scratchpad_clear(self) -> None:
        write_call = ToolCall(tool="scratchpad", target="", args=json.dumps({"action": "write", "content": "Test content"}))
        run_tool(write_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)

        clear_call = ToolCall(tool="scratchpad", target="", args=json.dumps({"action": "clear"}))
        clear_output = run_tool(clear_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        clear_payload = json.loads(clear_output)
        self.assertTrue(clear_payload["success"])
        self.assertEqual("Scratchpad cleared.", clear_payload["data"]["message"])

        read_call = ToolCall(tool="scratchpad", target="", args=json.dumps({"action": "read"}))
        read_output = run_tool(read_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        read_payload = json.loads(read_output)
        self.assertTrue(read_payload["success"])
        self.assertEqual("", read_payload["data"]["content"])

    def test_scratchpad_default_action_read(self) -> None:
        read_call = ToolCall(tool="scratchpad", target="", args=json.dumps({}))
        read_output = run_tool(read_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        read_payload = json.loads(read_output)
        self.assertTrue(read_payload["success"])
        self.assertEqual("", read_payload["data"]["content"])

    def test_scratchpad_invalid_action(self) -> None:
        invalid_call = ToolCall(tool="scratchpad", target="", args=json.dumps({"action": "invalid"}))
        invalid_output = run_tool(invalid_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        invalid_payload = json.loads(invalid_output)
        self.assertFalse(invalid_payload["success"])
        self.assertIn("Invalid action", invalid_payload["error"])

    def test_scratchpad_multiple_writes(self) -> None:
        write1_call = ToolCall(tool="scratchpad", target="", args=json.dumps({"action": "write", "content": "First content"}))
        run_tool(write1_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)

        write2_call = ToolCall(tool="scratchpad", target="", args=json.dumps({"action": "write", "content": "Second content"}))
        run_tool(write2_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)

        read_call = ToolCall(tool="scratchpad", target="", args=json.dumps({"action": "read"}))
        read_output = run_tool(read_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        read_payload = json.loads(read_output)
        self.assertTrue(read_payload["success"])
        self.assertEqual("Second content", read_payload["data"]["content"])

    def test_scratchpad_empty_write(self) -> None:
        write_call = ToolCall(tool="scratchpad", target="", args=json.dumps({"action": "write", "content": ""}))
        write_output = run_tool(write_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        write_payload = json.loads(write_output)
        self.assertTrue(write_payload["success"])
        self.assertEqual("Content written to scratchpad.", write_payload["data"]["message"])

        read_call = ToolCall(tool="scratchpad", target="", args=json.dumps({"action": "read"}))
        read_output = run_tool(read_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        read_payload = json.loads(read_output)
        self.assertTrue(read_payload["success"])
        self.assertEqual("", read_payload["data"]["content"])
