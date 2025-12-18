import json

from unittest import TestCase

from lmao.memory import (
    MAX_TOOL_RESULT_PROMPT_CHARS,
    MemoryState,
    TRUNCATION_MARKER,
    aggressive_compact_messages,
    compact_messages_if_needed,
    sanitize_assistant_reply,
    should_pin_agents_tool_result,
    truncate_tool_result_for_prompt,
)


class MemoryCompactionTests(TestCase):
    def test_sanitize_removes_think_steps(self) -> None:
        reply = {
            "type": "assistant_turn",
            "version": "2",
            "steps": [
                {"type": "think", "content": "plan"},
                {"type": "message", "purpose": "progress", "content": "ok"},
            ],
        }
        sanitized = sanitize_assistant_reply(json.dumps(reply, ensure_ascii=False), allowed_tools=[])
        self.assertIn("message", sanitized)
        self.assertNotIn("think", sanitized)

    def test_tool_result_truncation_respects_pin(self) -> None:
        payload = "x" * (MAX_TOOL_RESULT_PROMPT_CHARS + 100)
        trimmed, flagged = truncate_tool_result_for_prompt(payload, is_pinned=False)
        self.assertTrue(flagged)
        self.assertTrue(trimmed.endswith(TRUNCATION_MARKER))
        pinned, pinned_flag = truncate_tool_result_for_prompt(payload, is_pinned=True)
        self.assertEqual(pinned, payload)
        self.assertFalse(pinned_flag)
        self.assertTrue(should_pin_agents_tool_result("read_agents", ""))

    def test_compaction_keeps_system_user_and_pinned_results(self) -> None:
        system = {"role": "system", "content": "sys"}
        user_message = {"role": "user", "content": "original request"}
        assistant = {"role": "assistant", "content": "thinking"}
        pinned_tool = {
            "role": "user",
            "content": "Tool result for tool 'read' on 'AGENTS.md':\n...",
        }
        extra_assistant = {"role": "assistant", "content": "later"}
        messages = [system, user_message, assistant, pinned_tool, extra_assistant]
        state = MemoryState()
        state.last_user_message = user_message
        state.pinned_message_ids.add(id(pinned_tool))

        compact_messages_if_needed(
            messages,
            last_user_message=state.last_user_message,
            pinned_message_ids=state.pinned_message_ids,
            trigger_tokens=1,
            target_tokens=1,
            debug_logger=None,
        )

        self.assertEqual(messages[0], system)
        self.assertIn(user_message, messages)
        self.assertIn(pinned_tool, messages)
        self.assertNotIn(assistant, messages)
        self.assertNotIn(extra_assistant, messages)

    def test_aggressive_compaction_retain_pinned_and_last_user(self) -> None:
        system = {"role": "system", "content": "sys"}
        user_message = {"role": "user", "content": "follow-up"}
        pinned_tool = {
            "role": "user",
            "content": "Tool result for tool 'read_agents' on 'AGENTS.md':\n...",
        }
        other = {"role": "assistant", "content": "old"}
        messages = [system, user_message, pinned_tool, other]
        state = MemoryState()
        state.last_user_message = user_message
        state.pinned_message_ids.add(id(pinned_tool))

        aggressive_compact_messages(
            messages,
            last_user_message=state.last_user_message,
            pinned_message_ids=state.pinned_message_ids,
            debug_logger=None,
        )

        self.assertEqual(messages, [system, pinned_tool, user_message])

    def test_compaction_preserves_latest_tool_result(self) -> None:
        system = {"role": "system", "content": "sys"}
        pinned_tool = {
            "role": "user",
            "content": "Tool result for tool 'read_agents' on 'AGENTS.md':\n...",
        }
        user_message = {"role": "user", "content": "use the list tool"}
        assistant_call = {
            "role": "assistant",
            "content": '{"type":"assistant_turn","version":"2","steps":[{"type":"tool_call","call":{"tool":"list_skills","target":"","args":{}}}]}',
        }
        latest_tool_result = {
            "role": "user",
            "content": "Tool result for tool 'list_skills' on '':\n{\"tool\":\"list_skills\",\"success\":true,\"data\":[]}",
        }
        messages = [system, pinned_tool, user_message, assistant_call, latest_tool_result]
        state = MemoryState()
        state.last_user_message = user_message
        state.pinned_message_ids.add(id(pinned_tool))

        compact_messages_if_needed(
            messages,
            last_user_message=state.last_user_message,
            pinned_message_ids=state.pinned_message_ids,
            trigger_tokens=1,
            target_tokens=1,
            debug_logger=None,
        )

        self.assertEqual(messages[0], system)
        self.assertIn(user_message, messages)
        self.assertIn(pinned_tool, messages)
        self.assertIn(latest_tool_result, messages)
