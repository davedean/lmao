import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase

from lmao.plugins import discover_plugins
from lmao.tools import ToolCall, parse_tool_calls, run_tool, safe_target_path


class ToolCallParsingTests(TestCase):
    def test_parses_fenced_json(self) -> None:
        raw = "```json\n{\"tool\": \"read\", \"target\": \"file.txt\", \"args\": \"\"}\n```"
        call = ToolCall.from_raw_message(raw, allowed_tools=["read"])
        self.assertIsNotNone(call)
        self.assertEqual("read", call.tool)
        self.assertEqual("file.txt", call.target)

    def test_rejects_unknown_tool(self) -> None:
        raw = "{\"tool\": \"rm -rf\", \"target\": \"./\", \"args\": \"\"}"
        call = ToolCall.from_raw_message(raw, allowed_tools=["read", "ls"])
        self.assertIsNone(call)

    def test_parses_json_after_text(self) -> None:
        raw = "Let's do this\n{\"tool\": \"ls\", \"target\": \".\", \"args\": \"\"}"
        call = ToolCall.from_raw_message(raw, allowed_tools=["ls"])
        self.assertIsNotNone(call)
        self.assertEqual("ls", call.tool)

    def test_parses_multiple_tool_calls(self) -> None:
        raw = """{"tool":"ls","target":".","args":""}
{"tool":"find","target":".","args":""}"""
        calls = parse_tool_calls(raw, allowed_tools=["ls", "find"])
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

    def test_safe_target_path_allows_repo_root_prefixed_paths(self) -> None:
        resolved = safe_target_path("/skills", self.base, extra_roots=[])
        self.assertEqual(self.base / "skills", resolved)

    def test_safe_target_path_allows_true_absolute_paths_inside_base(self) -> None:
        target = self.base / "logs" / "test.log"
        target.parent.mkdir(parents=True)
        target.write_text("x", encoding="utf-8")
        resolved = safe_target_path(str(target), self.base, extra_roots=[])
        self.assertEqual(target.resolve(), resolved)

    def test_safe_target_path_blocks_escape_even_with_leading_slash(self) -> None:
        with self.assertRaises(ValueError):
            safe_target_path("/../outside", self.base, extra_roots=[])

    def test_safe_target_path_blocks_symlink_escape(self) -> None:
        outside_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(outside_tmp.cleanup)
        outside = Path(outside_tmp.name).resolve()

        escape = self.base / "escape"
        try:
            os.symlink(str(outside), str(escape))
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink not supported: {exc}")

        with self.assertRaises(ValueError):
            safe_target_path("escape/secret.txt", self.base, extra_roots=[])

    def test_run_tool_read_with_line_range(self) -> None:
        target = self.base / "notes.txt"
        target.write_text("a\nb\nc\nd\n", encoding="utf-8")
        call = ToolCall(tool="read", target="notes.txt", args="lines:2-3")
        output = run_tool(call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        payload = json.loads(output)
        self.assertTrue(payload["success"])
        self.assertEqual({"start": 2, "end": 3}, payload["data"]["lines"])
        self.assertEqual("b\nc", payload["data"]["content"])

    def test_patch_replaces_line_range(self) -> None:
        target = self.base / "hello.txt"
        target.write_text("a\nb\nc\nd\n", encoding="utf-8")
        args = json.dumps({"range": "lines:2-3", "content": "B\nC\n"})
        call = ToolCall(tool="patch", target="hello.txt", args=args)
        output = run_tool(call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        payload = json.loads(output)
        self.assertTrue(payload["success"])
        self.assertEqual({"start": 2, "end": 3}, payload["data"]["range"])
        self.assertEqual("a\nB\nC\nd\n", target.read_text(encoding="utf-8"))

    def test_read_only_blocks_patch(self) -> None:
        target = self.base / "hello.txt"
        target.write_text("a\n", encoding="utf-8")
        call = ToolCall(tool="patch", target="hello.txt", args=json.dumps({"range": "lines:1-1", "content": "b\n"}))
        output = run_tool(call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, read_only=True, plugin_tools=self.plugins)
        payload = json.loads(output)
        self.assertFalse(payload["success"])
        self.assertIn("read-only", payload["error"])

    def test_write_blocks_top_level_skill_file(self) -> None:
        skills_root = self.base / "skills"
        skills_root.mkdir()
        call = ToolCall(tool="write", target="skills/loose.md", args="demo")
        result = run_tool(call, base=self.base, extra_roots=[], skill_roots=[skills_root], yolo_enabled=False, plugin_tools=self.plugins)
        payload = json.loads(result)
        self.assertFalse(payload["success"])
        self.assertIn("skills/<skill-name>", payload["error"])

    def test_write_requires_non_empty_target(self) -> None:
        call = ToolCall(tool="write", target="", args="demo")
        result = run_tool(call, base=self.base, extra_roots=[], skill_roots=[], yolo_enabled=False, plugin_tools=self.plugins)
        payload = json.loads(result)
        self.assertFalse(payload["success"])
        self.assertIn("missing target", payload["error"])

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
