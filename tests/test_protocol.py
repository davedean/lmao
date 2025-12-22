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

    def test_accepts_version_alias_v2(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "v2",
                "steps": [{"type": "message", "content": "hi"}, {"type": "end"}],
            }
        )
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("2", turn.version)

    def test_accepts_fenced_json(self) -> None:
        raw_obj = {
            "type": "assistant_turn",
            "version": "1",
            "steps": [{"type": "message", "content": "hi"}],
        }
        raw = f"```json\n{json.dumps(raw_obj)}\n```"
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("assistant_turn", turn.type)

    def test_accepts_preamble_text_then_json(self) -> None:
        raw_obj = {
            "type": "assistant_turn",
            "version": "1",
            "steps": [{"type": "message", "content": "hi"}],
        }
        raw = f"Sure, here you go:\n{json.dumps(raw_obj)}"
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("assistant_turn", turn.type)

    def test_skips_non_protocol_json_before_assistant_turn(self) -> None:
        raw_obj = {
            "type": "assistant_turn",
            "version": "1",
            "steps": [{"type": "message", "content": "hi"}],
        }
        raw = f'{{"foo": 1}}\n{json.dumps(raw_obj)}'
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("assistant_turn", turn.type)

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

    def test_coerces_common_message_purpose_aliases(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "1",
                "steps": [{"type": "message", "purpose": "success", "content": "hi"}],
            }
        )
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("assistant_turn", turn.type)

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
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual(2, len(turn.steps))

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

    def test_distinguishes_unknown_tool(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "1",
                "steps": [{"type": "tool_call", "call": {"tool": "ghost", "target": "a", "args": ""}}],
            }
        )
        with self.assertRaises(ProtocolError) as exc:
            parse_assistant_turn(
                raw,
                allowed_tools=["read"],
                known_tools=["read", "write"],
            )
        self.assertIn("not found", str(exc.exception))

    def test_distinguishes_disallowed_tool(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "1",
                "steps": [{"type": "tool_call", "call": {"tool": "write", "target": "a", "args": ""}}],
            }
        )
        with self.assertRaises(ProtocolError) as exc:
            parse_assistant_turn(
                raw,
                allowed_tools=["read"],
                known_tools=["read", "write"],
            )
        self.assertIn("not allowed", str(exc.exception))

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

    def test_accepts_python_dict_repr(self) -> None:
        raw = "{'type':'assistant_turn','version':'1','steps':[{'type':'message','content':'hi'}]}"
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("assistant_turn", turn.type)

    def test_defaults_version_when_missing(self) -> None:
        raw = '{"type":"assistant_turn","steps":[{"type":"message","content":"hi"}]}'
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("assistant_turn", turn.type)
        self.assertEqual(1, len(turn.steps))

    def test_wraps_single_tool_call_step(self) -> None:
        raw = "{'type':'tool_call','call':{'tool':'read','target':'a.txt','args':'lines:1-2'}}"
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("assistant_turn", turn.type)
        self.assertEqual(1, len(turn.steps))

    def test_parses_v2_tool_call_with_structured_args_and_meta(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "2",
                "steps": [
                    {
                        "type": "tool_call",
                        "call": {
                            "tool": "read",
                            "target": "a.txt",
                            "args": {"lines": "1-2"},
                            "meta": {"timeout_s": 5},
                        },
                    }
                ],
            }
        )
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("2", turn.version)
        step = turn.steps[0]
        self.assertEqual("tool_call", step.type)
        self.assertEqual({"lines": "1-2"}, getattr(step, "call").args)
        self.assertEqual({"timeout_s": 5}, getattr(step, "call").meta)

    def test_rejects_v2_non_object_meta(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "2",
                "steps": [
                    {"type": "tool_call", "call": {"tool": "read", "target": "a", "args": {}, "meta": "nope"}}
                ],
            }
        )
        with self.assertRaises(ProtocolError):
            parse_assistant_turn(raw, allowed_tools=["read"])

    def test_allows_direct_tool_step_alias_in_v2(self) -> None:
        raw = json.dumps(
            {
                "type": "assistant_turn",
                "version": "2",
                "steps": [{"type": "write", "target": "a.txt", "args": {"content": "hello"}}],
            }
        )
        turn = parse_assistant_turn(raw, allowed_tools=["write"])
        self.assertEqual("assistant_turn", turn.type)
        self.assertEqual(1, len(turn.steps))
        step = turn.steps[0]
        self.assertEqual("tool_call", step.type)
        self.assertEqual("write", getattr(step, "call").tool)

    def test_wraps_single_direct_tool_step_dict(self) -> None:
        raw = json.dumps({"type": "read", "target": "a.txt", "args": "lines:1-2"})
        turn = parse_assistant_turn(raw, allowed_tools=["read"])
        self.assertEqual("assistant_turn", turn.type)
        self.assertEqual(1, len(turn.steps))
        step = turn.steps[0]
        self.assertEqual("tool_call", step.type)
        self.assertEqual("read", getattr(step, "call").tool)
