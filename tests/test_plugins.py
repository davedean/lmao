import json
import tempfile
from pathlib import Path
from unittest import TestCase

from lmao.plugins import PLUGIN_API_VERSION, discover_plugins
from lmao.tools import ToolCall, get_allowed_tools, run_tool


PLUGIN_TEMPLATE = """from lmao.plugins import PLUGIN_API_VERSION

PLUGIN = {{
    "name": "{name}",
    "description": "test plugin",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": {is_destructive},
    "allow_in_read_only": {allow_in_read_only},
    "allow_in_normal": {allow_in_normal},
    "allow_in_yolo": {allow_in_yolo},
}}


def run(target, args, base, extra_roots, skill_roots, task_manager=None, debug_logger=None):
    import json
    return json.dumps({{"tool": "{name}", "success": True, "data": {{"target": target, "args": args}}}})
"""


class PluginLoaderTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_plugin(
        self,
        name: str = "echo_plugin",
        destructive: bool = False,
        allow_in_read_only: bool = True,
        allow_in_normal: bool = True,
        allow_in_yolo: bool = True,
    ) -> Path:
        plugin_dir = self.base / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "tool.py").write_text(
            PLUGIN_TEMPLATE.format(
                name=name,
                is_destructive=str(destructive),
                allow_in_read_only=str(allow_in_read_only),
                allow_in_normal=str(allow_in_normal),
                allow_in_yolo=str(allow_in_yolo),
            ),
            encoding="utf-8",
        )
        return plugin_dir

    def test_loads_plugin_and_dispatches(self) -> None:
        plugin_dir = self._write_plugin()
        plugins = discover_plugins([plugin_dir], self.base)
        self.assertIn("echo_plugin", plugins)

        allowed = get_allowed_tools(read_only=False, yolo_enabled=False, plugins=plugins.values())
        self.assertIn("echo_plugin", allowed)

        call = ToolCall(tool="echo_plugin", target="t", args="payload")
        result = run_tool(
            call,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            plugin_tools=plugins,
        )
        payload = json.loads(result)
        self.assertTrue(payload["success"])
        self.assertEqual({"target": "t", "args": "payload"}, payload["data"])

    def test_read_only_skips_destructive_plugins(self) -> None:
        plugin_dir = self._write_plugin(name="mutator", destructive=True, allow_in_read_only=False)
        plugins = discover_plugins([plugin_dir], self.base)
        self.assertIn("mutator", plugins)

        allowed = get_allowed_tools(read_only=True, yolo_enabled=False, plugins=plugins.values())
        self.assertNotIn("mutator", allowed)

        call = ToolCall(tool="mutator", target="", args="")
        result = run_tool(
            call,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=True,
            plugin_tools=plugins,
        )
        payload = json.loads(result)
        self.assertFalse(payload["success"])
        self.assertIn("read-only", payload["error"])

    def test_plugin_modes(self) -> None:
        plugin_dir = self._write_plugin(name="yolo_only", allow_in_normal=False, allow_in_yolo=True)
        plugins = discover_plugins([plugin_dir], self.base)

        allowed_normal = get_allowed_tools(read_only=False, yolo_enabled=False, plugins=plugins.values())
        self.assertNotIn("yolo_only", allowed_normal)

        allowed_yolo = get_allowed_tools(read_only=False, yolo_enabled=True, plugins=plugins.values())
        self.assertIn("yolo_only", allowed_yolo)

        call = ToolCall(tool="yolo_only", target="", args="")
        result = run_tool(
            call,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            plugin_tools=plugins,
        )
        payload = json.loads(result)
        self.assertFalse(payload["success"])
        self.assertIn("not allowed in normal mode", payload["error"])
