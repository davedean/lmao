import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from lmao.plugins import discover_plugins
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

META_PLUGIN_TEMPLATE = """from lmao.plugins import PLUGIN_API_VERSION

PLUGIN = {{
    "name": "{name}",
    "description": "test meta plugin",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "usage": "{{'tool':'{name}','target':'','args':''}}",
}}


def run(target, args, base, extra_roots, skill_roots, task_manager=None, debug_logger=None, meta=None):
    import json
    return json.dumps({{"tool": "{name}", "success": True, "data": {{"target": target, "args": args, "meta": meta}}}})
"""

MULTI_PLUGIN_TEMPLATE = """from lmao.plugins import PLUGIN_API_VERSION

PLUGINS = [
    {{
        "name": "{name_a}",
        "description": "test multi plugin a",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "usage": "{{'tool':'{name_a}','target':'','args':''}}",
    }},
    {{
        "name": "{name_b}",
        "description": "test multi plugin b",
        "api_version": PLUGIN_API_VERSION,
        "is_destructive": False,
        "allow_in_read_only": True,
        "allow_in_normal": True,
        "allow_in_yolo": True,
        "always_confirm": False,
        "usage": "{{'tool':'{name_b}','target':'','args':''}}",
    }},
]


def run(tool_name, target, args, base, extra_roots, skill_roots, task_manager=None, debug_logger=None):
    import json
    return json.dumps({{"tool": tool_name, "success": True, "data": {{"target": target, "args": args}}}})
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

    def _write_multi_plugin(self, name_a: str = "multi_a", name_b: str = "multi_b") -> Path:
        plugin_dir = self.base / "plugins" / "multi"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "tool.py").write_text(
            MULTI_PLUGIN_TEMPLATE.format(name_a=name_a, name_b=name_b),
            encoding="utf-8",
        )
        return plugin_dir

    def _write_meta_plugin(self, name: str = "meta_plugin") -> Path:
        plugin_dir = self.base / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "tool.py").write_text(
            META_PLUGIN_TEMPLATE.format(name=name),
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

    def test_always_confirm_plugins_auto_approve_in_yolo(self) -> None:
        plugin_dir = self._write_plugin(name="confirmme_yolo", always_confirm=True)
        plugins = discover_plugins([plugin_dir], self.base)

        allowed = get_allowed_tools(read_only=False, yolo_enabled=True, plugins=plugins.values())
        self.assertIn("confirmme_yolo", allowed)

        call = ToolCall(tool="confirmme_yolo", target="", args="")
        with patch("builtins.input", side_effect=AssertionError("input() should not be called in yolo mode")):
            result = run_tool(
                call,
                base=self.base,
                extra_roots=[],
                skill_roots=[],
                yolo_enabled=True,
                plugin_tools=plugins,
            )
        payload = json.loads(result)
        self.assertTrue(payload["success"])

    def test_loads_multi_tool_plugin(self) -> None:
        plugin_dir = self._write_multi_plugin(name_a="multi_a", name_b="multi_b")
        plugins = discover_plugins([plugin_dir], self.base)
        self.assertIn("multi_a", plugins)
        self.assertIn("multi_b", plugins)

        call_a = ToolCall(tool="multi_a", target="t", args="a")
        result_a = run_tool(
            call_a,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            plugin_tools=plugins,
        )
        payload_a = json.loads(result_a)
        self.assertTrue(payload_a["success"])
        self.assertEqual("multi_a", payload_a["tool"])

        call_b = ToolCall(tool="multi_b", target="t", args="b")
        result_b = run_tool(
            call_b,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            plugin_tools=plugins,
        )
        payload_b = json.loads(result_b)
        self.assertTrue(payload_b["success"])
        self.assertEqual("multi_b", payload_b["tool"])

    def test_dispatch_passes_meta_when_supported(self) -> None:
        plugin_dir = self._write_meta_plugin()
        plugins = discover_plugins([plugin_dir], self.base)
        call = ToolCall(tool="meta_plugin", target="t", args={"x": 1}, meta={"timeout_s": 1})
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
        self.assertEqual({"timeout_s": 1}, payload["data"]["meta"])


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
        self.assertIn("git_status", allowed_normal)
        self.assertIn("git_diff", allowed_normal)

        allowed_read_only = get_allowed_tools(read_only=True, yolo_enabled=False, plugins=self.plugins.values())
        self.assertNotIn("git_add", allowed_read_only)
        self.assertNotIn("git_commit", allowed_read_only)
        self.assertIn("git_status", allowed_read_only)
        self.assertIn("git_diff", allowed_read_only)

        for name in ("git_status", "git_diff"):
            call = ToolCall(tool=name, target="", args="")
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
            self.assertIn("not inside a git repository", payload["error"])

    def test_all_core_plugins_discovered(self) -> None:
        core_names = {
            "read",
            "write",
            "patch",
            "mkdir",
            "move",
            "ls",
            "find",
            "grep",
            "read_agents",
            "list_skills",
            "add_task",
            "complete_task",
            "delete_task",
            "list_tasks",
            "git_add",
            "git_commit",
            "git_status",
            "git_diff",
            "bash",
            "tool_help",
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

    def test_bash_plugin_runs_without_confirm_in_yolo(self) -> None:
        allowed = get_allowed_tools(read_only=False, yolo_enabled=True, plugins=self.plugins.values())
        self.assertIn("bash", allowed)

        call = ToolCall(tool="bash", target="", args="echo ok")
        with patch("builtins.input", side_effect=AssertionError("input() should not be called in yolo mode")):
            result = run_tool(
                call,
                base=self.base,
                extra_roots=[],
                skill_roots=[],
                yolo_enabled=True,
                plugin_tools=self.plugins,
            )
        payload = json.loads(result)
        self.assertTrue(payload["success"])
        self.assertEqual("ok", payload["data"]["stdout"])

    def test_builtin_plugins_include_usage_examples(self) -> None:
        missing = [name for name, plugin in self.plugins.items() if not plugin.usage_examples]
        self.assertEqual([], missing)
