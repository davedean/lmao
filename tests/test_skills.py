import tempfile
from pathlib import Path
from unittest import TestCase

from lmao.skills import list_skill_info, list_skill_paths, validate_skill_write_target


class SkillGuardsTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_blocks_top_level_skill_file(self) -> None:
        skill_root = self.tmpdir / "skills"
        target = skill_root / "loose.md"
        error = validate_skill_write_target(target, [skill_root])
        self.assertIsNotNone(error)
        self.assertIn("skills/<skill-name>", error)

    def test_blocks_top_level_skill_file_without_extension(self) -> None:
        skill_root = self.tmpdir / "skills"
        target = skill_root / "loose"
        error = validate_skill_write_target(target, [skill_root])
        self.assertIsNotNone(error)
        self.assertIn("skills/<skill-name>", error)

    def test_allows_nested_skill_file(self) -> None:
        skill_root = self.tmpdir / "skills"
        target = skill_root / "demo" / "SKILL.md"
        error = validate_skill_write_target(target, [skill_root])
        self.assertIsNone(error)

    def test_lists_skill_paths(self) -> None:
        skill_root = Path(self.tmpdir) / "skills"
        other_root = Path(self.tmpdir) / "user_skills"
        for root, name in ((skill_root, "demo"), (other_root, "personal")):
            (root / name).mkdir(parents=True, exist_ok=True)
            (root / name / "SKILL.md").write_text("frontmatter", encoding="utf-8")
        paths = list_skill_paths([skill_root, other_root])
        self.assertTrue(any(path.endswith("/demo") for path in paths))
        self.assertTrue(any(path.endswith("/personal") for path in paths))

    def test_lists_skill_info(self) -> None:
        skill_root = Path(self.tmpdir) / "skills"
        (skill_root / "alpha").mkdir(parents=True, exist_ok=True)
        (skill_root / "alpha" / "SKILL.md").write_text("frontmatter", encoding="utf-8")
        info = list_skill_info([skill_root])
        self.assertEqual([("alpha", skill_root / "alpha")], info)
