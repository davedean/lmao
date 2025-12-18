from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from lmao.llm import LLMCallResult, LLMCallStats
from lmao.runtime_tools import RuntimeContext, build_runtime_tool_registry
from lmao.subagents import subagent_run_tool
from lmao.tool_dispatch import run_tool
from lmao.tool_parsing import ToolCall


class _FakeClient:
    def __init__(self, content: str) -> None:
        self._content = content

    def call(self, messages):  # type: ignore[no-untyped-def]
        stats = LLMCallStats(
            elapsed_s=0.01,
            request_bytes=0,
            response_bytes=len(self._content),
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            is_estimate=True,
        )
        return LLMCallResult(content=self._content, stats=stats)


class TestSubagents(unittest.TestCase):
    def test_subagent_run_tool_requires_objective(self) -> None:
        runtime_ctx = RuntimeContext(
            client=_FakeClient(""),
            plugin_tools={},
            base=SimpleNamespace(),
            extra_roots=(),
            skill_roots=(),
            yolo_enabled=False,
            read_only=False,
            debug_logger=None,
        )
        raw = subagent_run_tool(runtime_ctx, "", {"context": "x"}, None)
        payload = json.loads(raw)
        self.assertFalse(payload["success"])

    def test_subagent_run_tool_happy_path(self) -> None:
        content = json.dumps(
            {
                "type": "assistant_turn",
                "version": "2",
                "steps": [
                    {"type": "message", "purpose": "final", "content": "done"},
                    {"type": "end", "reason": "completed"},
                ],
            }
        )
        runtime_ctx = RuntimeContext(
            client=_FakeClient(content),
            plugin_tools={},
            base=SimpleNamespace(),
            extra_roots=(),
            skill_roots=(),
            yolo_enabled=False,
            read_only=False,
            debug_logger=None,
        )
        raw = subagent_run_tool(runtime_ctx, "", {"objective": "test", "max_turns": 2}, None)
        payload = json.loads(raw)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["result"]["status"], "ok")
        self.assertEqual(payload["data"]["result"]["summary"], "done")

    def test_subagent_run_treats_final_message_as_done(self) -> None:
        content = json.dumps(
            {
                "type": "assistant_turn",
                "version": "2",
                "steps": [
                    {"type": "message", "purpose": "final", "content": "done"},
                ],
            }
        )
        runtime_ctx = RuntimeContext(
            client=_FakeClient(content),
            plugin_tools={},
            base=SimpleNamespace(),
            extra_roots=(),
            skill_roots=(),
            yolo_enabled=False,
            read_only=False,
            debug_logger=None,
        )
        raw = subagent_run_tool(runtime_ctx, "", {"objective": "test", "max_turns": 2}, None)
        payload = json.loads(raw)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["result"]["status"], "ok")

    def test_subagent_run_tool_accepts_target_as_objective(self) -> None:
        content = json.dumps(
            {
                "type": "assistant_turn",
                "version": "2",
                "steps": [
                    {"type": "message", "purpose": "final", "content": "done"},
                    {"type": "end"},
                ],
            }
        )
        runtime_ctx = RuntimeContext(
            client=_FakeClient(content),
            plugin_tools={},
            base=SimpleNamespace(),
            extra_roots=(),
            skill_roots=(),
            yolo_enabled=False,
            read_only=False,
            debug_logger=None,
        )
        raw = subagent_run_tool(runtime_ctx, "objective from target", "", None)
        payload = json.loads(raw)
        self.assertTrue(payload["success"])

    def test_runtime_dispatch_calls_subagent_run(self) -> None:
        content = json.dumps(
            {
                "type": "assistant_turn",
                "version": "2",
                "steps": [
                    {"type": "message", "purpose": "final", "content": "ok"},
                    {"type": "end"},
                ],
            }
        )
        plugin_tools = {}
        runtime_tools = build_runtime_tool_registry()
        runtime_ctx = RuntimeContext(
            client=_FakeClient(content),
            plugin_tools=plugin_tools,
            base=SimpleNamespace(),
            extra_roots=(),
            skill_roots=(),
            yolo_enabled=False,
            read_only=False,
            debug_logger=None,
        )
        call = ToolCall(tool="subagent_run", target="", args={"objective": "x"}, meta=None)
        raw = run_tool(
            call,
            base=SimpleNamespace(),
            extra_roots=(),
            skill_roots=(),
            yolo_enabled=False,
            read_only=False,
            plugin_tools=plugin_tools,
            runtime_tools=runtime_tools,
            runtime_context=runtime_ctx,
            debug_logger=None,
        )
        payload = json.loads(raw)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["tool"], "subagent_run")
