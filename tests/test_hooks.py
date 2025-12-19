import json
import tempfile
from pathlib import Path
from unittest import TestCase

from lmao.hooks import (
    HookContext,
    HookRegistry,
    ToolHookContext,
    ToolHookTypes,
)
from lmao.plugins import PluginTool, discover_plugin_hooks
from lmao.runtime_tools import RuntimeContext
from lmao.tool_parsing import ToolCall
from lmao.tool_dispatch import run_tool


class HookRegistryTests(TestCase):
    def test_priority_ordering(self) -> None:
        registry = HookRegistry()
        events = []

        def low(ctx: HookContext) -> None:
            events.append("low")

        def high(ctx: HookContext) -> None:
            events.append("high")

        registry.register("test", low, priority=0)
        registry.register("test", high, priority=10)
        registry.execute_hooks("test", HookContext(hook_type="test", runtime_state={}))
        self.assertEqual(events, ["high", "low"])

    def test_cancel_stops_execution(self) -> None:
        registry = HookRegistry()
        events = []

        def stop(ctx: HookContext) -> HookContext:
            ctx.cancel("stop")
            events.append("stop")
            return ctx

        def never(ctx: HookContext) -> None:
            events.append("never")

        registry.register("test", stop, priority=0)
        registry.register("test", never, priority=-1)
        result = registry.execute_hooks("test", HookContext(hook_type="test", runtime_state={}))
        self.assertTrue(result.should_cancel)
        self.assertEqual(events, ["stop"])


class ToolHookIntegrationTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_post_transform_hook_overrides_result(self) -> None:
        registry = HookRegistry()

        def handler(target, args, base, extra_roots, skill_roots, task_manager=None, debug_logger=None):
            return json.dumps({"tool": "echo", "success": True, "data": {"value": args}})

        plugin = PluginTool(
            name="echo",
            description="test",
            input_schema=None,
            usage_examples=[],
            details=[],
            is_destructive=False,
            allow_in_read_only=True,
            allow_in_normal=True,
            allow_in_yolo=True,
            always_confirm=False,
            handler=handler,
            path=self.base / "tool.py",
        )

        def transform(ctx: ToolHookContext) -> ToolHookContext:
            ctx.tool_result = json.dumps({"tool": "echo", "success": True, "data": {"value": "hooked"}})
            return ctx

        registry.register(ToolHookTypes.POST_RESULT_TRANSFORM, transform, priority=0)
        runtime_ctx = RuntimeContext(
            client=None,  # unused for tool dispatch
            plugin_tools={"echo": plugin},
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=False,
            hook_registry=registry,
        )

        result = run_tool(
            ToolCall(tool="echo", target="", args="input"),
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            plugin_tools={"echo": plugin},
            runtime_context=runtime_ctx,
        )
        payload = json.loads(result)
        self.assertEqual(payload["data"]["value"], "hooked")


class PluginHookDiscoveryTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_discovers_manifest_hooks(self) -> None:
        plugin_dir = self.base / "plugins" / "hooked"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "tool.py").write_text(
            """from lmao.plugins import PLUGIN_API_VERSION

PLUGIN = {
    "name": "hooked",
    "description": "hook test",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "hooks": {
        "pre_tool_execution": ["hook_fn"]
    }
}

def run(target, args, base, extra_roots, skill_roots, task_manager=None, debug_logger=None):
    return ""

def hook_fn(context):
    return None
""",
            encoding="utf-8",
        )
        registry = HookRegistry()
        discover_plugin_hooks([plugin_dir], self.base, registry)
        self.assertIn("pre_tool_execution", registry.get_hook_types())
