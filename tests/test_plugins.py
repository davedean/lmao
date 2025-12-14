import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

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
    "always_confirm": {always_confirm},
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
        always_confirm: bool = False,
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
                always_confirm=str(always_confirm),
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

    def test_always_confirm_plugins_prompt(self) -> None:
        plugin_dir = self._write_plugin(name="confirmme", always_confirm=True)
        plugins = discover_plugins([plugin_dir], self.base)

        allowed = get_allowed_tools(read_only=False, yolo_enabled=False, plugins=plugins.values())
        self.assertIn("confirmme", allowed)

        call = ToolCall(tool="confirmme", target="", args="")
        # Decline run
        with patch("builtins.input", return_value="n"):
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
        self.assertIn("not approved", payload["error"])

        # Approve run
        with patch("builtins.input", return_value="yes"):
            result_ok = run_tool(
                call,
                base=self.base,
                extra_roots=[],
                skill_roots=[],
                yolo_enabled=False,
                plugin_tools=plugins,
            )
        payload_ok = json.loads(result_ok)
        self.assertTrue(payload_ok["success"])


class BuiltinPluginTests(TestCase):
    def setUp(self) -> None:
        self.base = Path(tempfile.mkdtemp()).resolve()
        tools_dir = Path(__file__).resolve().parent.parent / "lmao" / "tools"
        self.plugins = discover_plugins([tools_dir], self.base, allow_outside_base=True)

    def tearDown(self) -> None:
        try:
            for child in self.base.glob("*"):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    for p in sorted(child.rglob("*"), reverse=True):
                        if p.is_file():
                            p.unlink()
                    child.rmdir()
            self.base.rmdir()
        except Exception:
            pass

    def test_git_plugins_respect_read_only(self) -> None:
        # No repo present: both should error, but still be allowed in normal/yolo
        allowed_normal = get_allowed_tools(read_only=False, yolo_enabled=False, plugins=self.plugins.values())
        self.assertIn("git_add", allowed_normal)
        self.assertIn("git_commit", allowed_normal)

        allowed_read_only = get_allowed_tools(read_only=True, yolo_enabled=False, plugins=self.plugins.values())
        self.assertNotIn("git_add", allowed_read_only)
        self.assertNotIn("git_commit", allowed_read_only)

    def test_all_core_plugins_discovered(self) -> None:
        core_names = {
            "read",
            "write",
            "mkdir",
            "move",
            "ls",
            "find",
            "grep",
            "list_skills",
            "add_task",
            "complete_task",
            "delete_task",
            "list_tasks",
            "git_add",
            "git_commit",
            "bash",
        }
        self.assertTrue(core_names.issubset(self.plugins.keys()))

    def test_bash_plugin_always_confirms(self) -> None:
        allowed = get_allowed_tools(read_only=False, yolo_enabled=False, plugins=self.plugins.values())
        self.assertIn("bash", allowed)

        call = ToolCall(tool="bash", target="", args="echo ok")
        with patch("builtins.input", return_value="n"):
            result = run_tool(
                call,
                base=self.base,
                extra_roots=[],
                skill_roots=[],
                yolo_enabled=False,
                plugin_tools=self.plugins,
            )
        payload = json.loads(result)
        self.assertFalse(payload["success"])
        self.assertIn("not approved", payload["error"])
