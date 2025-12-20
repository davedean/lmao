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
        self.assertEqual("missing target file path", payload["error"])

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


class ToolBehaviorTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        tools_dir = Path(__file__).resolve().parent.parent / "lmao" / "tools"
        self.plugins = discover_plugins([tools_dir], self.base, allow_outside_base=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, tool: str, target: str, args) -> dict:
        call = ToolCall(tool=tool, target=target, args=args)
        output = run_tool(
            call,
            base=self.base,
            extra_roots=[],
            skill_roots=[self.base / "skills"],
            yolo_enabled=True,
            plugin_tools=self.plugins,
        )
        return json.loads(output)

    def test_grep_skips_dotfiles_and_truncates(self) -> None:
        visible = self.base / "visible.txt"
        lines = ["match"] * 201
        visible.write_text("\n".join(lines), encoding="utf-8")
        hidden = self.base / ".hidden"
        hidden.write_text("match", encoding="utf-8")
        payload = self._run("grep", ".", {"pattern": "match"})
        self.assertTrue(payload["success"])
        data = payload["data"]
        self.assertTrue(data.get("truncated"))
        self.assertEqual(200, len(data["matches"]))
        for match in data["matches"]:
            self.assertNotIn(".hidden", match["path"])

    def test_grep_skips_large_files(self) -> None:
        big = self.base / "big.txt"
        big.write_text("x" * 2_000_001 + "\nneedle\n", encoding="utf-8")
        small = self.base / "small.txt"
        small.write_text("needle", encoding="utf-8")
        payload = self._run("grep", ".", {"pattern": "needle"})
        self.assertTrue(payload["success"])
        matches = payload["data"]["matches"]
        paths = {match["path"] for match in matches}
        self.assertIn("small.txt", paths)
        self.assertNotIn("big.txt", paths)

    def test_find_include_dotfiles(self) -> None:
        (self.base / "visible.txt").write_text("x", encoding="utf-8")
        (self.base / ".hidden").write_text("x", encoding="utf-8")
        payload = self._run("find", ".", {"include_dotfiles": True})
        self.assertTrue(payload["success"])
        names = {entry["name"] for entry in payload["data"]["results"]}
        self.assertIn(".hidden", names)
        self.assertIn("visible.txt", names)

    def test_find_truncates(self) -> None:
        for idx in range(3):
            (self.base / f"file{idx}.txt").write_text("x", encoding="utf-8")
        payload = self._run("find", ".", {"max_entries": 1})
        self.assertTrue(payload["success"])
        self.assertEqual(1, len(payload["data"]["results"]))
        self.assertTrue(payload["data"].get("truncated"))

    def test_mkdir_requires_target(self) -> None:
        payload = self._run("mkdir", "", "")
        self.assertFalse(payload["success"])
        self.assertEqual("missing target path", payload["error"])

    def test_ls_uses_args_path(self) -> None:
        target = self.base / "note.txt"
        target.write_text("hi", encoding="utf-8")
        payload = self._run("ls", "", {"path": "note.txt"})
        self.assertTrue(payload["success"])
        self.assertEqual(1, len(payload["data"]["entries"]))
        self.assertEqual("note.txt", payload["data"]["entries"][0]["name"])

    def test_ls_missing_path_errors(self) -> None:
        payload = self._run("ls", "missing.txt", "")
        self.assertFalse(payload["success"])
        self.assertIn("not found", payload["error"])

    def test_mkdir_uses_args_path(self) -> None:
        payload = self._run("mkdir", "", {"path": "newdir"})
        self.assertTrue(payload["success"])
        self.assertTrue((self.base / "newdir").exists())

    def test_mkdir_blocks_escape(self) -> None:
        call = ToolCall(tool="mkdir", target="../outside", args="")
        output = run_tool(
            call,
            base=self.base,
            extra_roots=[],
            skill_roots=[self.base / "skills"],
            yolo_enabled=False,
            plugin_tools=self.plugins,
        )
        payload = json.loads(output)
        self.assertFalse(payload["success"])
        self.assertIn("escapes", payload["error"])

    def test_write_decodes_escaped_text(self) -> None:
        payload = self._run("write", "escaped.txt", "line1\\nline2")
        self.assertTrue(payload["success"])
        content = (self.base / "escaped.txt").read_text(encoding="utf-8")
        self.assertEqual("line1\nline2", content)

    def test_write_rejects_directory_path(self) -> None:
        payload = self._run("write", "somedir/", "x")
        self.assertFalse(payload["success"])
        self.assertIn("mkdir", payload["error"])

    def test_write_allows_skill_child_path(self) -> None:
        skill_dir = self.base / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        payload = self._run("write", "skills/demo/notes.txt", "ok")
        self.assertTrue(payload["success"])
        self.assertTrue((skill_dir / "notes.txt").exists())

    def test_read_accepts_args_path_and_range(self) -> None:
        target = self.base / "notes.txt"
        target.write_text("a\nb\nc\n", encoding="utf-8")
        payload = self._run("read", "", {"path": "notes.txt", "start": 2, "end": 3})
        self.assertTrue(payload["success"])
        self.assertEqual("b\nc", payload["data"]["content"])
        self.assertEqual({"start": 2, "end": 3}, payload["data"]["lines"])

    def test_read_requires_target(self) -> None:
        payload = self._run("read", "", "")
        self.assertFalse(payload["success"])
        self.assertEqual("missing target file path", payload["error"])

    def test_read_truncates_large_content(self) -> None:
        target = self.base / "big.txt"
        target.write_text("a" * 200_010, encoding="utf-8")
        payload = self._run("read", "big.txt", "")
        self.assertTrue(payload["success"])
        self.assertTrue(payload["data"].get("truncated"))
        self.assertEqual(200_000, payload["data"]["limit_chars"])
        self.assertEqual(200_000, len(payload["data"]["content"]))

    def test_move_success_and_destination_exists(self) -> None:
        source = self.base / "from.txt"
        source.write_text("hi", encoding="utf-8")
        payload = self._run("move", "from.txt", "to.txt")
        self.assertTrue(payload["success"])
        self.assertTrue((self.base / "to.txt").exists())
        (self.base / "from.txt").write_text("again", encoding="utf-8")
        payload = self._run("move", "from.txt", "to.txt")
        self.assertFalse(payload["success"])
        self.assertIn("already exists", payload["error"])

    def test_move_requires_destination(self) -> None:
        source = self.base / "from.txt"
        source.write_text("hi", encoding="utf-8")
        payload = self._run("move", "from.txt", {"dest": ""})
        self.assertFalse(payload["success"])
        self.assertEqual("missing target path", payload["error"])

    def test_move_directory(self) -> None:
        source_dir = self.base / "dir"
        source_dir.mkdir()
        payload = self._run("move", "dir", "dir_new")
        self.assertTrue(payload["success"])
        self.assertTrue((self.base / "dir_new").exists())

    def test_patch_start_end_dict_and_invalid_args(self) -> None:
        target = self.base / "patch.txt"
        target.write_text("a\nb\nc\n", encoding="utf-8")
        payload = self._run("patch", "patch.txt", {"start": 2, "end": 2, "content": "B\n"})
        self.assertTrue(payload["success"])
        self.assertEqual("a\nB\nc\n", target.read_text(encoding="utf-8"))
        payload = self._run("patch", "patch.txt", "nope")
        self.assertFalse(payload["success"])
        self.assertIn("invalid patch args", payload["error"])

    def test_patch_requires_target(self) -> None:
        payload = self._run("patch", "", {"range": "lines:1-1", "content": "x"})
        self.assertFalse(payload["success"])
        self.assertEqual("missing target path", payload["error"])

    def test_patch_rejects_directory_path(self) -> None:
        directory = self.base / "dir"
        directory.mkdir()
        payload = self._run("patch", "dir/", {"range": "lines:1-1", "content": "x"})
        self.assertFalse(payload["success"])
        self.assertIn("directory", payload["error"])

    def test_tools_guide_single_and_multi(self) -> None:
        payload = self._run("tools_guide", "", {"tool": "read"})
        self.assertTrue(payload["success"])
        self.assertEqual("read", payload["data"]["name"])
        payload = self._run("tools_guide", "", {"tools": ["read", "not_a_tool"]})
        self.assertTrue(payload["success"])
        results = payload["data"]["tools"]
        self.assertEqual(2, len(results))
        errors = [item for item in results if not item["success"]]
        self.assertEqual(1, len(errors))

    def test_tools_guide_unknown(self) -> None:
        payload = self._run("tools_guide", "", {"tool": "not_a_tool", "list_only": True})
        self.assertFalse(payload["success"])
        self.assertIn("unknown tool", payload["error"])

    def test_tools_list_pattern_filter(self) -> None:
        payload = self._run("tools_list", "", {"pattern": "read"})
        self.assertTrue(payload["success"])
        names = {tool["name"] for tool in payload["data"]["tools"]}
        self.assertIn("read", names)
        self.assertEqual("read", payload["data"]["pattern"])

    def test_tools_list_no_match(self) -> None:
        payload = self._run("tools_list", "", {"pattern": "definitely-not-a-tool"})
        self.assertTrue(payload["success"])
        self.assertEqual([], payload["data"]["tools"])
        self.assertEqual("definitely-not-a-tool", payload["data"]["pattern"])

    def test_policy_reads_agents_excerpt(self) -> None:
        agents = self.base / "AGENTS.md"
        agents.write_text("x" * 2500, encoding="utf-8")
        payload = self._run("policy", "", {"offset": 0, "limit": 100})
        self.assertTrue(payload["success"])
        self.assertTrue(payload["data"]["has_more"])
        self.assertEqual(100, payload["data"]["limit_chars"])
        self.assertEqual(100, len(payload["data"]["content"]))

    def test_policy_truncate_false(self) -> None:
        agents = self.base / "AGENTS.md"
        agents.write_text("hello", encoding="utf-8")
        payload = self._run("policy", "", {"truncate": False})
        self.assertTrue(payload["success"])
        self.assertFalse(payload["data"]["has_more"])
        self.assertEqual("hello", payload["data"]["content"])

    def test_policy_missing_agents(self) -> None:
        other_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(other_tmp.cleanup)
        other_base = Path(other_tmp.name).resolve()
        call = ToolCall(tool="policy", target="", args="")
        output = run_tool(
            call,
            base=other_base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=True,
            plugin_tools=self.plugins,
        )
        payload = json.loads(output)
        self.assertFalse(payload["success"])
        self.assertIn("no AGENTS.md", payload["error"])
