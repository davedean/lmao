import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from lmao import context


class ContextDiscoveryTests(TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_gather_context_reads_repo_agents(self) -> None:
        repo_root = self.base / "repo"
        repo_root.mkdir()
        (repo_root / ".git").mkdir()
        nested = repo_root / "nested"
        nested.mkdir()
        agents_file = repo_root / "AGENTS.md"
        agents_file.write_text("Repo instructions", encoding="utf-8")

        with patch("pathlib.Path.home", return_value=self.base):
            notes = context.gather_context(nested)

        self.assertEqual(repo_root, notes.repo_root)
        self.assertEqual(agents_file, notes.nearest_agents)
        self.assertIn("Repo instructions", notes.repo_notes)
        self.assertEqual("", notes.user_notes)
        self.assertIsNone(notes.user_skills)
