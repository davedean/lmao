import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from lmao.hooks import (
    HookRegistry,
    ErrorHookContext,
    ErrorHookTypes,
    AgentHookContext,
    AgentHookTypes,
    ProtocolHookContext,
    ProtocolHookTypes,
    HookResult,
)
from lmao.runtime_tools import RuntimeContext, RuntimeTool
from lmao.llm import LLMClient


class GovernanceHooksTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        self.hook_registry = HookRegistry()
        self.runtime_context = RuntimeContext(
            client=LLMClient(endpoint="http://test", model="test"),
            plugin_tools={},
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            read_only=False,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_protocol_parse_error_hook(self) -> None:
        """Test protocol parse error hook handling."""
        recovery_triggered = False

        def recovery_hook(context) -> HookResult:
            nonlocal recovery_triggered
            recovery_triggered = True
            if context.recovery_attempts > 2:
                return HookResult(
                    success=True,
                    data={
                        "action": "fail_conversation",
                        "message": "Too many protocol errors",
                    },
                    should_cancel=True,
                )
            return HookResult(
                success=True,
                data={
                    "action": "insert_user_message",
                    "message": "Please fix your JSON format",
                },
            )

        self.hook_registry.register(
            ErrorHookTypes.ON_PROTOCOL_PARSE_ERROR,
            recovery_hook,
            priority=10,
        )

        # Test recovery scenario
        error_context = ErrorHookContext(
            hook_type=ErrorHookTypes.ON_PROTOCOL_PARSE_ERROR,
            runtime_state={"invalid_replies": 1},
            error_type="protocol_parse_error",
            error_message="Invalid JSON",
            recovery_attempts=1,
        )

        result = self.hook_registry.execute_hooks(
            ErrorHookTypes.ON_PROTOCOL_PARSE_ERROR, error_context
        )

        self.assertTrue(recovery_triggered)
        self.assertEqual(result.data["action"], "insert_user_message")
        self.assertIn("Please fix your JSON format", result.data["message"])

    def test_empty_reply_hook(self) -> None:
        """Test empty reply hook handling."""
        recovery_triggered = False

        def empty_reply_hook(context) -> HookResult:
            nonlocal recovery_triggered
            recovery_triggered = True
            if context.recovery_attempts >= 4:
                return HookResult(
                    success=True,
                    data={
                        "action": "insert_assistant_message_and_end",
                        "message": "Auto-generated fallback response",
                    },
                    should_cancel=True,
                )
            return HookResult(
                success=True,
                data={
                    "action": "insert_user_message",
                    "message": "Please provide a non-empty response",
                },
            )

        self.hook_registry.register(
            ErrorHookTypes.ON_LLM_EMPTY_REPLY_ERROR,
            empty_reply_hook,
            priority=10,
        )

        # Test empty recovery scenario
        error_context = ErrorHookContext(
            hook_type=ErrorHookTypes.ON_LLM_EMPTY_REPLY_ERROR,
            runtime_state={"empty_replies": 3},
            error_type="empty_reply_error",
            error_message="Empty reply",
            recovery_attempts=3,
        )

        result = self.hook_registry.execute_hooks(
            ErrorHookTypes.ON_LLM_EMPTY_REPLY_ERROR, error_context
        )

        self.assertTrue(recovery_triggered)
        self.assertEqual(result.data["action"], "insert_user_message")

    def test_agent_startup_hook(self) -> None:
        """Test agent startup hook handling."""
        startup_executed = False
        policy_called = False
        skills_guide_called = False

        # Mock the call_tool_internal method for testing
        def mock_call_internal(self, tool_name, target="", args=None, **kwargs):
            if tool_name == "policy":
                nonlocal policy_called
                policy_called = True
                return '{"tool":"policy","success":true,"data":{"rules":[]}}'
            elif tool_name == "skills_guide":
                nonlocal skills_guide_called
                skills_guide_called = True
                return '{"tool":"skills_guide","success":true,"data":{"skills":[]}}'
            return f'{{"tool":"{tool_name}","success":true}}'

        def startup_hook(context) -> HookResult:
            nonlocal startup_executed
            startup_executed = True

            # Execute policy tool internally
            if context.runtime_state.get("current_user_input"):
                runtime_context = context.runtime_state.get("runtime_context")
                if runtime_context is None:
                    return HookResult(success=False, data={"startup_handled": False})
                policy_result = runtime_context.call_tool_internal(
                    "policy", args={"truncate": True}
                )

                # Check if skills guide should be included
                prompt = context.runtime_state.get("initial_prompt", "")
                if "skill" in prompt.lower():
                    skills_result = runtime_context.call_tool_internal("skills_guide")

            return HookResult(
                success=True,
                data={
                    "startup_handled": True,
                    "policy_executed": True,
                    "skills_guide_executed": "skill" in prompt.lower(),
                },
            )

        self.hook_registry.register(
            AgentHookTypes.AGENT_STARTUP,
            startup_hook,
            priority=10,
        )

        with patch.object(RuntimeContext, "call_tool_internal", new=mock_call_internal):
            # Test startup scenario
            startup_context = AgentHookContext(
                hook_type=AgentHookTypes.AGENT_STARTUP,
                runtime_state={
                    "workdir": str(self.base),
                    "initial_prompt": "Create a new skill",
                    "current_user_input": "Create a new skill",
                    "runtime_context": self.runtime_context,
                },
                agent_type="primary",
            )

            result = self.hook_registry.execute_hooks(
                AgentHookTypes.AGENT_STARTUP, startup_context
            )

        self.assertTrue(startup_executed)
        self.assertTrue(policy_called)
        self.assertTrue(skills_guide_called)
        self.assertTrue(result.data["startup_handled"])

    def test_headless_guardrail_hook(self) -> None:
        """Test headless guardrail hook handling."""
        guardrail_triggered = False

        def guardrail_hook(context) -> HookResult:
            nonlocal guardrail_triggered
            guardrail_triggered = True

            user_messages = context.runtime_state.get("user_messages", [])
            has_input_request = any("?" in msg for msg in user_messages)

            if context.runtime_state.get("headless_run") and has_input_request:
                return HookResult(
                    success=True,
                    data={
                        "action": "insert_user_message",
                        "message": "Headless mode: proceed autonomously without asking questions",
                    },
                )

            return HookResult(success=True, data={})

        self.hook_registry.register(
            ProtocolHookTypes.POST_MESSAGE_VALIDATION,
            guardrail_hook,
            priority=10,
        )

        # Test headless scenario with input request
        validation_context = ProtocolHookContext(
            hook_type=ProtocolHookTypes.POST_MESSAGE_VALIDATION,
            runtime_state={
                "headless_run": True,
                "user_messages": ["What do you think about this?"],
            },
            parsed_message={
                "steps": [{"type": "message", "content": "What do you think?"}]
            },
            validation_result=True,
        )

        result = self.hook_registry.execute_hooks(
            ProtocolHookTypes.POST_MESSAGE_VALIDATION, validation_context
        )

        self.assertTrue(guardrail_triggered)
        self.assertEqual(result.data["action"], "insert_user_message")
        self.assertIn("Headless mode", result.data["message"])

    def test_post_message_parsing_hook(self) -> None:
        """Test post-message parsing hook handling."""
        parsing_hook_triggered = False

        def parsing_hook(context) -> HookResult:
            nonlocal parsing_hook_triggered
            parsing_hook_triggered = True

            # Check for think-only or progress-only scenarios
            thinks = context.runtime_state.get("thinks", 0)
            user_messages = context.runtime_state.get("user_messages", 0)
            tool_calls = context.runtime_state.get("tool_call_payloads", 0)
            has_end = context.runtime_state.get("has_end", False)

            if thinks > 0 and user_messages == 0 and tool_calls == 0 and not has_end:
                return HookResult(
                    success=True,
                    data={
                        "action": "insert_user_message",
                        "message": "Think-only turns are not allowed. Please call a tool.",
                    },
                )

            if user_messages > 0 and tool_calls == 0 and not has_end:
                return HookResult(
                    success=True,
                    data={
                        "action": "insert_user_message",
                        "message": "Progress-only turns are not allowed. Please call a tool or end.",
                    },
                )

            return HookResult(success=True, data={})

        self.hook_registry.register(
            ProtocolHookTypes.POST_MESSAGE_PARSING,
            parsing_hook,
            priority=10,
        )

        # Test think-only scenario
        think_context = ProtocolHookContext(
            hook_type=ProtocolHookTypes.POST_MESSAGE_PARSING,
            runtime_state={
                "thinks": 1,
                "user_messages": 0,
                "tool_call_payloads": 0,
                "has_end": False,
            },
            parsed_message={"steps": [{"type": "think", "content": "Thinking..."}]},
            validation_result=True,
        )

        result = self.hook_registry.execute_hooks(
            ProtocolHookTypes.POST_MESSAGE_PARSING, think_context
        )

        self.assertTrue(parsing_hook_triggered)
        self.assertEqual(result.data["action"], "insert_user_message")
        self.assertIn("Think-only turns are not allowed", result.data["message"])

    def test_hook_priority_execution(self) -> None:
        """Test that hooks execute in priority order."""
        execution_order = []

        def low_priority_hook(context):
            execution_order.append("low")
            return HookResult(success=True)

        def high_priority_hook(context):
            execution_order.append("high")
            return HookResult(success=True)

        self.hook_registry.register("test", low_priority_hook, priority=0)
        self.hook_registry.register("test", high_priority_hook, priority=10)

        self.hook_registry.execute_hooks(
            "test",
            AgentHookContext(
                hook_type="test",
                runtime_state={},
                agent_type="test",
            ),
        )

        self.assertEqual(execution_order, ["high", "low"])

    def test_hook_cancellation_stops_execution(self) -> None:
        """Test that hook cancellation stops further hook execution."""
        executed_hooks = []

        def canceling_hook(context):
            executed_hooks.append("canceling")
            context.cancel("Test cancellation")
            return context

        def should_not_execute_hook(context):
            executed_hooks.append("should_not_execute")
            return HookResult(success=True)

        self.hook_registry.register("test", canceling_hook, priority=10)
        self.hook_registry.register("test", should_not_execute_hook, priority=0)

        result = self.hook_registry.execute_hooks(
            "test",
            AgentHookContext(
                hook_type="test",
                runtime_state={},
                agent_type="test",
            ),
        )

        self.assertEqual(executed_hooks, ["canceling"])
        self.assertTrue(result.should_cancel)

    def test_hook_data_accumulation(self) -> None:
        """Test that hook data is properly accumulated."""

        def hook1(context):
            return HookResult(success=True, data={"value1": "from_hook1"})

        def hook2(context):
            return HookResult(success=True, data={"value2": "from_hook2"})

        self.hook_registry.register("test", hook1)
        self.hook_registry.register("test", hook2)

        result = self.hook_registry.execute_hooks(
            "test",
            AgentHookContext(
                hook_type="test",
                runtime_state={},
                agent_type="test",
            ),
        )

        self.assertEqual(result.data["value1"], "from_hook1")
        self.assertEqual(result.data["value2"], "from_hook2")
