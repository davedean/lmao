import contextlib
import io
import warnings
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from lmao.cli import main
from lmao.config import ConfigLoadResult, UserConfig


class CLIModeOptionsTests(TestCase):
    @patch("lmao.cli.run_loop")
    @patch("lmao.cli.LLMClient")
    def test_conflicting_mode_and_yolo_flags_error(self, client_cls, run_loop) -> None:
        config_result = ConfigLoadResult(Path("lmao.conf"), UserConfig(), None, True)
        with patch("sys.argv", ["lmao", "--mode", "yolo", "--yolo"]), patch(
            "lmao.cli.load_user_config",
            return_value=config_result,
        ):
            with self.assertRaises(SystemExit) as exc, contextlib.redirect_stderr(
                io.StringIO()
            ):
                main()
            self.assertEqual(2, exc.exception.code)
        run_loop.assert_not_called()

    @patch("lmao.cli.run_loop")
    @patch("lmao.cli.LLMClient")
    def test_yolo_deprecated_warns_and_runs(self, client_cls, run_loop) -> None:
        config_result = ConfigLoadResult(Path("lmao.conf"), UserConfig(), None, True)
        with patch("sys.argv", ["lmao", "--yolo", "--headless", "prompt"]), patch(
            "lmao.cli.load_user_config",
            return_value=config_result,
        ), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            main()
        run_loop.assert_called_once()
        self.assertTrue(
            any(
                issubclass(record.category, DeprecationWarning)
                and "--yolo is deprecated" in str(record.message)
                for record in captured
            )
        )
    @patch("lmao.cli.run_loop")
    @patch("lmao.cli.LLMClient")
    def test_free_flag_rejects_non_openrouter(self, client_cls, run_loop) -> None:
        config_result = ConfigLoadResult(Path("lmao.conf"), UserConfig(), None, True)
        with patch("sys.argv", ["lmao", "--provider", "lmstudio", "--free"]), patch(
            "lmao.cli.load_user_config",
            return_value=config_result,
        ):
            with self.assertRaises(SystemExit) as exc, contextlib.redirect_stderr(
                io.StringIO()
            ):
                main()
            self.assertEqual(2, exc.exception.code)
        run_loop.assert_not_called()

    @patch("lmao.cli.run_loop")
    @patch("lmao.cli.LLMClient")
    def test_quiet_and_no_tools_flags(self, client_cls, run_loop) -> None:
        config_result = ConfigLoadResult(Path("lmao.conf"), UserConfig(), None, True)
        with patch(
            "sys.argv",
            ["lmao", "--quiet", "--no-tools", "--headless", "prompt"],
        ), patch(
            "lmao.cli.load_user_config",
            return_value=config_result,
        ):
            main()
        run_loop.assert_called_once()
        kwargs = run_loop.call_args[1]
        self.assertTrue(kwargs["quiet"])
        self.assertTrue(kwargs["no_tools"])
