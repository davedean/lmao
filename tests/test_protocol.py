import json
from unittest import TestCase

from lmao.protocol import ProtocolError, parse_assistant_turn


class ProtocolParsingTests(TestCase):
    def test_parses_message_and_end(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "1",
                "steps": [
                    {"type": "think", "content": "plan"},
                    {"type": "message", "format": "markdown", "purpose": "final", "content": "done"},
                    {"type": "end", "reason": "completed"},
                ],
            }
        )
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("assistant_turn", turn.type)
        self.assertEqual("1", turn.version)
        self.assertEqual(3, len(turn.steps))

    def test_rejects_invalid_message_purpose(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "1",
                "steps": [{"type": "message", "purpose": "random", "content": "hi"}],
            }
        )
        with self.assertRaises(ProtocolError):
            parse_assistant_turn(raw, allowed_tools=["read"])

    def test_rejects_multiple_tool_calls(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "1",
                "steps": [
                    {"type": "tool_call", "call": {"tool": "read", "target": "a", "args": ""}},
                    {"type": "tool_call", "call": {"tool": "read", "target": "b", "args": ""}},
                ],
            }
        )
        with self.assertRaises(ProtocolError):
            parse_assistant_turn(raw, allowed_tools=["read"])

    def test_rejects_disallowed_tool(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "1",
                "steps": [{"type": "tool_call", "call": {"tool": "write", "target": "a", "args": "x"}}],
            }
        )
        with self.assertRaises(ProtocolError):
            parse_assistant_turn(raw, allowed_tools=["read"])

    def test_allows_multiple_task_tool_steps(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "1",
                "steps": [
                    {"type": "add_task", "args": {"task": "one"}},
                    {"type": "add_task", "args": {"task": "two"}},
                    {"type": "list_tasks"},
                ],
            }
        )
        turn = parse_assistant_turn(raw, allowed_tools=["add_task", "list_tasks"])
        self.assertEqual(3, len(turn.steps))

    def test_accepts_extra_data_with_valid_prefix(self) -> None:
        raw = (
            '{"type":"assistant_turn","version":"1","steps":[{"type":"message","content":"hi"}]}'
            + "\n"
            + '{"tool":"ls"}'
        )
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("assistant_turn", turn.type)
