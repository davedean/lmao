import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from lmao.cli import main
from lmao.config import ConfigLoadResult, UserConfig


class CLIHeadlessTests(TestCase):
    @patch("lmao.cli.run_loop")
    @patch("lmao.cli.LLMClient")
    def test_headless_without_prompt_errors(self, client_cls, run_loop):
        config_result = ConfigLoadResult(Path("lmao.conf"), UserConfig(), None, True)
        with patch("sys.argv", ["lmao", "--headless"]), patch(
            "lmao.cli.load_user_config",
            return_value=config_result,
        ):
            with self.assertRaises(SystemExit) as exc:
                main()
            self.assertEqual(2, exc.exception.code)
        run_loop.assert_not_called()

    @patch("lmao.cli.run_loop")
    @patch("lmao.cli.LLMClient")
    def test_config_headless_default_prompt_used(self, client_cls, run_loop):
        config = UserConfig(default_prompt="stored prompt", headless=True)
        config_result = ConfigLoadResult(Path("lmao.conf"), config, None, True)
        with patch("sys.argv", ["lmao"]), patch(
            "lmao.cli.load_user_config",
            return_value=config_result,
        ):
            main()
        run_loop.assert_called_once()
        kwargs = run_loop.call_args[1]
        self.assertEqual("stored prompt", kwargs["initial_prompt"])
        self.assertTrue(kwargs["headless"])
