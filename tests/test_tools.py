import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase

from lmao.plugins import discover_plugins
from lmao.tools import ToolCall, parse_tool_calls, run_tool, safe_target_path
from lmao.task_list import TaskListManager


class ToolCallParsingTests(TestCase):
    def test_parses_fenced_json(self) -> None:
        raw = "```json\n{\"tool\": \"read\", \"target\": \"file.txt\", \"args\": \"\"}\n```"
        call = ToolCall.from_raw_message(raw)
        self.assertIsNotNone(call)
        self.assertEqual("read", call.tool)
        self.assertEqual("file.txt", call.target)

    def test_rejects_unknown_tool(self) -> None:
        raw = "{\"tool\": \"rm -rf\", \"target\": \"./\", \"args\": \"\"}"
        call = ToolCall.from_raw_message(raw)
        self.assertIsNone(call)

    def test_parses_json_after_text(self) -> None:
        raw = "Let's do this\n{\"tool\": \"ls\", \"target\": \".\", \"args\": \"\"}"
        call = ToolCall.from_raw_message(raw)
        self.assertIsNotNone(call)
        self.assertEqual("ls", call.tool)

    def test_parses_multiple_tool_calls(self) -> None:
        raw = """{"tool":"ls","target":".","args":""}
{"tool":"find","target":".","args":""}"""
        calls = parse_tool_calls(raw)
        self.assertEqual(2, len(calls))
        self.assertEqual("ls", calls[0].tool)
        self.assertEqual("find", calls[1].tool)


class ToolSafetyTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        tools_dir = Path(__file__).resolve().parent.parent / "lmao" / "tools"
        self.plugins = discover_plugins([tools_dir], self.base, allow_outside_base=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_safe_target_path_blocks_escape(self) -> None:
        with self.assertRaises(ValueError):
            safe_target_path("../outside", self.base, extra_roots=[])

    def test_safe_target_path_allows_slash_as_repo_root(self) -> None:
        resolved = safe_target_path("/", self.base, extra_roots=[])
        self.assertEqual(self.base, resolved)

    def test_run_tool_read_with_line_range(self) -> None:
        target = self.base / "notes.txt"
        target.write_text("a\nb\nc\nd\n", encoding="utf-8")
        call = ToolCall(tool="read", target="notes.txt", args="lines:2-3")
        output = run_tool(call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        payload = json.loads(output)
        self.assertTrue(payload["success"])
        self.assertEqual({"start": 2, "end": 3}, payload["data"]["lines"])
        self.assertEqual("b\nc", payload["data"]["content"])

    def test_write_blocks_top_level_skill_file(self) -> None:
        skills_root = self.base / "skills"
        skills_root.mkdir()
        call = ToolCall(tool="write", target="skills/loose.md", args="demo")
        result = run_tool(call, base=self.base, extra_roots=[], skill_roots=[skills_root], yolo_enabled=False, plugin_tools=self.plugins)
        payload = json.loads(result)
        self.assertFalse(payload["success"])
        self.assertIn("skills/<skill-name>", payload["error"])

    def test_move_blocks_top_level_skill_file(self) -> None:
        skills_root = self.base / "skills"
        skills_root.mkdir()
        source = self.base / "notes.txt"
        source.write_text("demo", encoding="utf-8")
        call = ToolCall(tool="move", target="notes.txt", args="skills/loose")
        result = run_tool(call, base=self.base, extra_roots=[], skill_roots=[skills_root], yolo_enabled=False, plugin_tools=self.plugins)
        payload = json.loads(result)
        self.assertFalse(payload["success"])
        self.assertIn("skills/<skill-name>", payload["error"])

    def test_list_skills_includes_user_and_repo(self) -> None:
        repo_skill = self.base / "skills" / "demo"
        repo_skill.mkdir(parents=True)
        (repo_skill / "SKILL.md").write_text("demo", encoding="utf-8")

        user_root = self.base / "user_skills"
        user_skill = user_root / "personal"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("personal", encoding="utf-8")

        call = ToolCall(tool="list_skills", target="", args="")
        output = run_tool(
            call,
            base=self.base,
            extra_roots=[user_root],
            skill_roots=[self.base / "skills", user_root],
            yolo_enabled=False,
            plugin_tools=self.plugins,
        )
        payload = json.loads(output)
        self.assertTrue(payload["success"])
        paths = {entry["path"] for entry in payload["data"]}
        self.assertIn(str(repo_skill), paths)
        self.assertIn(str(user_skill), paths)

    def test_task_tools_flow(self) -> None:
        manager = TaskListManager()
        add = ToolCall(tool="add_task", target="", args="do something")
        complete = ToolCall(tool="complete_task", target="", args="1")
        list_call = ToolCall(tool="list_tasks", target="", args="")

        add_out = run_tool(add, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins, task_manager=manager)
        add_payload = json.loads(add_out)
        self.assertTrue(add_payload["success"])

        list_out = run_tool(list_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins, task_manager=manager)
        list_payload = json.loads(list_out)
        self.assertTrue(list_payload["success"])
        self.assertIn("[ ] 1 do something", list_payload["data"]["render"])

        complete_out = run_tool(complete, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins, task_manager=manager)
        complete_payload = json.loads(complete_out)
        self.assertTrue(complete_payload["success"])

        list_out_done = run_tool(list_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins, task_manager=manager)
        list_payload_done = json.loads(list_out_done)
        self.assertTrue(list_payload_done["success"])
        self.assertIn("[x] 1 do something", list_payload_done["data"]["render"])

    def test_task_tools_accept_target_as_payload(self) -> None:
        manager = TaskListManager()
        add = ToolCall(tool="add_task", target="task via target", args="")
        list_call = ToolCall(tool="list_tasks", target="", args="")
        add_out = run_tool(add, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins, task_manager=manager)
        add_payload = json.loads(add_out)
        self.assertTrue(add_payload["success"])
        list_out = run_tool(list_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins, task_manager=manager)
        list_payload = json.loads(list_out)
        self.assertIn("task via target", list_payload["data"]["render"])

    def test_find_quotes_paths_with_spaces(self) -> None:
        spaced_dir = self.base / "my folder"
        spaced_dir.mkdir()
        (spaced_dir / "file name.txt").write_text("hi", encoding="utf-8")
        call = ToolCall(tool="find", target=".", args="")
        output = run_tool(call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        payload = json.loads(output)
        paths = {entry["path"] for entry in payload["data"]["results"]}
        self.assertIn("my folder/", paths)
        self.assertIn("my folder/file name.txt", paths)

    def test_add_task_strips_numbering_and_newlines(self) -> None:
        manager = TaskListManager()
        add = ToolCall(tool="add_task", target="", args="1. do something\nmultiline")
        add_out = run_tool(add, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins, task_manager=manager)
        add_payload = json.loads(add_out)
        self.assertTrue(add_payload["success"])
        list_call = ToolCall(tool="list_tasks", target="", args="")
        list_out = run_tool(list_call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins, task_manager=manager)
        list_payload = json.loads(list_out)
        self.assertIn("[ ] 1 do something multiline", list_payload["data"]["render"])

    def test_read_only_blocks_destructive_tools(self) -> None:
        target = self.base / "notes.txt"
        call = ToolCall(tool="write", target=str(target), args="content")
        output = run_tool(call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, read_only=True, plugin_tools=self.plugins)
        payload = json.loads(output)
        self.assertFalse(payload["success"])
        self.assertIn("read-only", payload["error"])
