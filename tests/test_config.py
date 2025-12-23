import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from lmao.config import (
    DEFAULT_CONFIG_TEMPLATE,
    load_user_config,
    resolve_default_config_path,
    resolve_openrouter_api_key,
    resolve_openrouter_headers,
    resolve_provider_settings,
    UserConfig,
    write_default_config,
)


class ConfigModuleTests(TestCase):
    def test_load_config_parses_values(self) -> None:
        content = """
[core]
provider = openrouter
mode = yolo
multiline = true
silent_tools = false
no_stats = true
quiet = true
no_tools = true
max_turns = 5
workdir = ~/code

[policy]
truncate = false
truncate_chars = 9000

[generation]
temperature = 0.7
top_p = 0.9
max_tokens = 200

[tool_output]
max_tool_lines = 5
max_tool_chars = 250

[lmstudio]
endpoint = http://lm.example
model = qwen3-4b-test
context_window_tokens = 12345

[openrouter]
endpoint = https://or.example/api
model = openrouter/model
context_window_tokens = 23456
http_referer = https://referer.example
app_title = Example Title
api_key_env = CUSTOM_API_KEY
free_default_model = openai/gpt-4o-mini
free_blacklist = gpt-free-1, gpt-free-2

[debug]
log_path = /tmp/lmao-debug.log
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lmao.conf"
            path.write_text(content, encoding="utf-8")
            result = load_user_config(path)
            self.assertTrue(result.loaded)
            cfg = result.config
        self.assertEqual("openrouter", cfg.provider)
        self.assertEqual("yolo", cfg.mode)
        self.assertTrue(cfg.multiline)
        self.assertFalse(cfg.silent_tools)
        self.assertTrue(cfg.no_stats)
        self.assertTrue(cfg.quiet)
        self.assertTrue(cfg.no_tools)
        self.assertEqual(5, cfg.max_turns)
        self.assertEqual("~/code", cfg.workdir)
        self.assertAlmostEqual(0.7, cfg.temperature)
        self.assertAlmostEqual(0.9, cfg.top_p)
        self.assertEqual(200, cfg.max_tokens)
        self.assertEqual(5, cfg.max_tool_lines)
        self.assertEqual(250, cfg.max_tool_chars)
        self.assertEqual("http://lm.example", cfg.lmstudio_endpoint)
        self.assertEqual("qwen3-4b-test", cfg.lmstudio_model)
        self.assertEqual(12345, cfg.lmstudio_context_window_tokens)
        self.assertEqual("/tmp/lmao-debug.log", cfg.debug_log_path)
        self.assertEqual("https://or.example/api", cfg.openrouter_endpoint)
        self.assertEqual("openrouter/model", cfg.openrouter_model)
        self.assertEqual(23456, cfg.openrouter_context_window_tokens)
        self.assertEqual("https://referer.example", cfg.openrouter_http_referer)
        self.assertEqual("Example Title", cfg.openrouter_app_title)
        self.assertIsNone(cfg.openrouter_api_key)
        self.assertEqual("CUSTOM_API_KEY", cfg.openrouter_api_key_env)
        self.assertEqual("openai/gpt-4o-mini", cfg.openrouter_free_default_model)
        self.assertEqual(("gpt-free-1", "gpt-free-2"), cfg.openrouter_free_blacklist)
        self.assertFalse(cfg.policy_truncate)
        self.assertEqual(9000, cfg.policy_truncate_chars)

    def test_resolve_provider_settings_precedence(self) -> None:
        env = {
            "LM_STUDIO_URL": "http://env-lm",
            "LM_STUDIO_MODEL": "env-model",
            "OPENROUTER_MODEL": "env-or-model",
        }
        config = UserConfig(
            lmstudio_endpoint="http://cfg-lm",
            lmstudio_model="cfg-model",
            openrouter_endpoint="https://cfg-openrouter",
            openrouter_model="cfg-openrouter-model",
        )
        settings = resolve_provider_settings(
            provider="lmstudio",
            cli_endpoint="http://cli-lm",
            cli_model="cli-model",
            config=config,
            env=env,
            lmstudio_default_endpoint="http://default-lm",
            lmstudio_default_model="default-model",
            openrouter_default_endpoint="https://default-openrouter",
        )
        self.assertEqual("http://cli-lm", settings.endpoint)
        self.assertEqual("cli-model", settings.model)

        settings = resolve_provider_settings(
            provider="lmstudio",
            cli_endpoint=None,
            cli_model=None,
            config=config,
            env=env,
            lmstudio_default_endpoint="http://default-lm",
            lmstudio_default_model="default-model",
            openrouter_default_endpoint="https://default-openrouter",
        )
        self.assertEqual("http://env-lm", settings.endpoint)
        self.assertEqual("env-model", settings.model)

        settings = resolve_provider_settings(
            provider="openrouter",
            cli_endpoint=None,
            cli_model="cli-or-model",
            config=config,
            env=env,
            lmstudio_default_endpoint="http://default-lm",
            lmstudio_default_model="default-model",
            openrouter_default_endpoint="https://default-openrouter",
        )
        self.assertEqual("https://cfg-openrouter", settings.endpoint)
        self.assertEqual("cli-or-model", settings.model)

        settings = resolve_provider_settings(
            provider="openrouter",
            cli_endpoint=None,
            cli_model=None,
            config=config,
            env=env,
            lmstudio_default_endpoint="http://default-lm",
            lmstudio_default_model="default-model",
            openrouter_default_endpoint="https://default-openrouter",
        )
        self.assertEqual("https://cfg-openrouter", settings.endpoint)
        self.assertEqual("env-or-model", settings.model)

        with self.assertRaises(ValueError):
            resolve_provider_settings(
                provider="openrouter",
                cli_endpoint=None,
                cli_model=None,
                config=UserConfig(),
                env={},
                lmstudio_default_endpoint="http://default-lm",
                lmstudio_default_model="default-model",
                openrouter_default_endpoint="https://default-openrouter",
            )

    def test_resolve_openrouter_headers_and_api_key(self) -> None:
        config = UserConfig(
            openrouter_http_referer="https://cfg-ref",
            openrouter_app_title="cfg-title",
            openrouter_api_key_env="CUSTOM_KEY",
        )
        env = {
            "OPENROUTER_HTTP_REFERER": "https://env-ref",
            "OPENROUTER_APP_TITLE": "env-title",
            "CUSTOM_KEY": "secret",
        }
        referer, title = resolve_openrouter_headers(
            cli_referer=None, cli_title=None, config=config, env=env
        )
        self.assertEqual("https://env-ref", referer)
        self.assertEqual("env-title", title)
        api_key = resolve_openrouter_api_key(cli_value=None, config=config, env=env)
        self.assertEqual("secret", api_key)
        api_key_cli = resolve_openrouter_api_key(
            cli_value="overridden", config=config, env=env
        )
        self.assertEqual("overridden", api_key_cli)

        # Test that direct config API key takes precedence over env var
        config_with_direct_key = UserConfig(openrouter_api_key="config-key")
        api_key_direct = resolve_openrouter_api_key(
            cli_value=None,
            config=config_with_direct_key,
            env={"OPENROUTER_API_KEY": "env-key"},
        )
        self.assertEqual("config-key", api_key_direct)

    def test_resolve_default_config_path_prefers_xdg(self) -> None:
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/tmp/xdg"}, clear=False):
            path = resolve_default_config_path()
            self.assertEqual(Path("/tmp/xdg/agents/lmao.conf"), path)

    def test_write_default_config_creates_template_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agents" / "lmao.conf"
            self.assertTrue(write_default_config(path))
            self.assertEqual(DEFAULT_CONFIG_TEMPLATE, path.read_text(encoding="utf-8"))
            self.assertFalse(write_default_config(path))
