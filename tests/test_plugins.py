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

    def _write_plugin(self, name: str = "echo_plugin", destructive: bool = False) -> Path:
        plugin_dir = self.base / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "tool.py").write_text(
            PLUGIN_TEMPLATE.format(name=name, is_destructive=str(destructive)),
            encoding="utf-8",
        )
        return plugin_dir

    def test_loads_plugin_and_dispatches(self) -> None:
        plugin_dir = self._write_plugin()
        plugins = discover_plugins([plugin_dir], self.base)
        self.assertIn("echo_plugin", plugins)

        allowed = get_allowed_tools(read_only=False, git_allowed=False, yolo_enabled=False, plugins=plugins.values())
        self.assertIn("echo_plugin", allowed)

        call = ToolCall(tool="echo_plugin", target="t", args="payload")
        result = run_tool(
            call,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            git_allowed=False,
            yolo_enabled=False,
            plugin_tools=plugins,
        )
        payload = json.loads(result)
        self.assertTrue(payload["success"])
        self.assertEqual({"target": "t", "args": "payload"}, payload["data"])

    def test_read_only_skips_destructive_plugins(self) -> None:
        plugin_dir = self._write_plugin(name="mutator", destructive=True)
        plugins = discover_plugins([plugin_dir], self.base)
        self.assertIn("mutator", plugins)

        allowed = get_allowed_tools(read_only=True, git_allowed=False, yolo_enabled=False, plugins=plugins.values())
        self.assertNotIn("mutator", allowed)

        call = ToolCall(tool="mutator", target="", args="")
        result = run_tool(
            call,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            git_allowed=False,
            yolo_enabled=False,
            read_only=True,
            plugin_tools=plugins,
        )
        payload = json.loads(result)
        self.assertFalse(payload["success"])
        self.assertIn("read-only", payload["error"])
