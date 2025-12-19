import json
import tempfile
from pathlib import Path
from unittest import TestCase

from lmao.plugins import PluginTool, discover_plugins
from lmao.runtime_tools import RuntimeTool, build_runtime_tool_registry
from lmao.tool_dispatch import (
    get_allowed_tools,
    get_allowed_runtime_tools,
    runtime_tool_allowed_visibility,
    runtime_tool_visible_to_agent,
)
from lmao.runtime_tools import RuntimeContext


class ToolVisibilityTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_runtime_tool_visibility_filter(self) -> None:
        """Test that runtime tools can be hidden from agent."""
        visible_tool = RuntimeTool(
            name="visible_tool",
            description="A visible tool",
            visible_to_agent=True,
        )
        hidden_tool = RuntimeTool(
            name="hidden_tool",
            description="A hidden tool",
            visible_to_agent=False,
        )

        runtime_tools = {
            "visible_tool": visible_tool,
            "hidden_tool": hidden_tool,
        }

        # Test visibility check
        self.assertTrue(runtime_tool_visible_to_agent(visible_tool))
        self.assertFalse(runtime_tool_visible_to_agent(hidden_tool))

        # Test allowed filtering in normal mode
        allowed = get_allowed_runtime_tools(
            runtime_tools, read_only=False, yolo_enabled=False
        )
        self.assertIn("visible_tool", allowed)
        self.assertNotIn("hidden_tool", allowed)

        # Test allowed filtering in yolo mode (visibility still applies)
        allowed_yolo = get_allowed_runtime_tools(
            runtime_tools, read_only=False, yolo_enabled=True
        )
        self.assertIn("visible_tool", allowed_yolo)
        self.assertNotIn("hidden_tool", allowed_yolo)

    def test_plugin_tool_visibility_filter(self) -> None:
        """Test that plugin tools can be hidden from agent."""
        visible_plugin = PluginTool(
            name="visible_plugin",
            description="A visible plugin",
            input_schema=None,
            usage_examples=[],
            details=[],
            is_destructive=False,
            allow_in_read_only=True,
            allow_in_normal=True,
            allow_in_yolo=True,
            always_confirm=False,
            handler=lambda *_: "",
            path=self.base / "visible.py",
            visible_to_agent=True,
        )
        hidden_plugin = PluginTool(
            name="hidden_plugin",
            description="A hidden plugin",
            input_schema=None,
            usage_examples=[],
            details=[],
            is_destructive=False,
            allow_in_read_only=True,
            allow_in_normal=True,
            allow_in_yolo=True,
            always_confirm=False,
            handler=lambda *_: "",
            path=self.base / "hidden.py",
            visible_to_agent=False,
        )

        plugins = [visible_plugin, hidden_plugin]

        # Test normal mode
        allowed = get_allowed_tools(
            read_only=False, yolo_enabled=False, plugins=plugins
        )
        self.assertIn("visible_plugin", allowed)
        self.assertNotIn("hidden_plugin", allowed)

        # Test yolo mode
        allowed_yolo = get_allowed_tools(
            read_only=False, yolo_enabled=True, plugins=plugins
        )
        self.assertIn("visible_plugin", allowed_yolo)
        self.assertNotIn("hidden_plugin", allowed_yolo)

    def test_plugin_manifest_visibility_parsing(self) -> None:
        """Test that plugin manifests correctly parse visible_to_agent field."""
        plugin_dir = self.base / "plugins" / "test_visibility"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "tool.py").write_text(
            """from lmao.plugins import PLUGIN_API_VERSION

PLUGIN = {
    "name": "test_hidden",
    "description": "Test hidden plugin",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "visible_to_agent": False
}

def run(target, args, base, extra_roots, skill_roots, task_manager=None, debug_logger=None):
    return '{"tool":"test_hidden","success":true,"data":{}}'
""",
            encoding="utf-8",
        )

        plugins = discover_plugins([plugin_dir], self.base)
        self.assertIn("test_hidden", plugins)
        plugin = plugins["test_hidden"]
        self.assertFalse(plugin.visible_to_agent)

    def test_runtime_context_internal_tool_call(self) -> None:
        """Test that RuntimeContext can call hidden tools internally."""

        # Create a hidden runtime tool
        def hidden_handler(ctx, target, args, meta):
            return json.dumps(
                {"tool": "hidden_tool", "success": True, "data": {"called": True}}
            )

        hidden_tool = RuntimeTool(
            name="hidden_tool",
            description="A hidden tool for internal use",
            visible_to_agent=False,
            handler=hidden_handler,
        )

        # Create runtime context
        from lmao.llm import LLMClient

        runtime_ctx = RuntimeContext(
            client=LLMClient(endpoint="http://test", model="test"),  # Mock client
            plugin_tools={},
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=False,
        )

        # Mock the tool registry to include our hidden tool
        custom_runtime_tools = {"hidden_tool": hidden_tool}

        # Test internal call works
        result = runtime_ctx.call_tool_internal(
            "hidden_tool", target="", args="", runtime_tools=custom_runtime_tools
        )
        print(f"Internal tool call result: {result}")
        payload = json.loads(result)
        self.assertTrue(payload["success"])
        self.assertTrue(payload["data"]["called"])

    def test_combined_visibility_workflow(self) -> None:
        """Test end-to-end workflow with mixed visible/hidden tools."""
        # Create test plugins with different visibility
        visible_plugin = PluginTool(
            name="visible_plugin",
            description="Visible plugin",
            input_schema=None,
            usage_examples=[],
            details=[],
            is_destructive=False,
            allow_in_read_only=True,
            allow_in_normal=True,
            allow_in_yolo=True,
            always_confirm=False,
            handler=lambda *_: '{"tool":"visible_plugin","success":true}',
            path=self.base / "visible.py",
            visible_to_agent=True,
        )
        hidden_plugin = PluginTool(
            name="hidden_plugin",
            description="Hidden plugin",
            input_schema=None,
            usage_examples=[],
            details=[],
            is_destructive=False,
            allow_in_read_only=True,
            allow_in_normal=True,
            allow_in_yolo=True,
            always_confirm=False,
            handler=lambda *_: '{"tool":"hidden_plugin","success":true}',
            path=self.base / "hidden.py",
            visible_to_agent=False,
        )

        plugins = [visible_plugin, hidden_plugin]

        # Create runtime tools with different visibility
        visible_runtime = RuntimeTool(
            name="visible_runtime",
            description="Visible runtime tool",
            visible_to_agent=True,
        )
        hidden_runtime = RuntimeTool(
            name="hidden_runtime",
            description="Hidden runtime tool",
            visible_to_agent=False,
        )

        runtime_tools = {
            "visible_runtime": visible_runtime,
            "hidden_runtime": hidden_runtime,
        }

        # Test that only visible tools are included in allowed lists
        allowed_plugins = get_allowed_tools(
            read_only=False, yolo_enabled=False, plugins=plugins
        )
        allowed_runtime = get_allowed_runtime_tools(
            runtime_tools, read_only=False, yolo_enabled=False
        )

        self.assertEqual(allowed_plugins, ["visible_plugin"])
        self.assertEqual(allowed_runtime, ["visible_runtime"])

        # Test that all tools still exist for internal calls
        self.assertEqual(len(plugins), 2)
        self.assertEqual(len(runtime_tools), 2)
