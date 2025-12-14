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
                    {"type": "message", "format": "markdown", "content": "done"},
                    {"type": "end", "reason": "completed"},
                ],
            }
        )
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("assistant_turn", turn.type)
        self.assertEqual("1", turn.version)
        self.assertEqual(3, len(turn.steps))

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

