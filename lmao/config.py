from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, TypeVar

from .llm import ProviderName

T = TypeVar("T")


@dataclass(frozen=True)
class UserConfig:
    provider: Optional[ProviderName] = None
    mode: Optional[str] = None
    default_prompt: Optional[str] = None
    headless: Optional[bool] = None
    multiline: Optional[bool] = None
    silent_tools: Optional[bool] = None
    no_stats: Optional[bool] = None
    max_turns: Optional[int] = None
    max_tool_lines: Optional[int] = None
    max_tool_chars: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    lmstudio_endpoint: Optional[str] = None
    lmstudio_model: Optional[str] = None
    lmstudio_context_window_tokens: Optional[int] = None
    openrouter_endpoint: Optional[str] = None
    openrouter_model: Optional[str] = None
    openrouter_context_window_tokens: Optional[int] = None
    openrouter_http_referer: Optional[str] = None
    openrouter_app_title: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_api_key_env: Optional[str] = None
    openrouter_free_default_model: Optional[str] = None
    openrouter_free_blacklist: tuple[str, ...] = ()
    workdir: Optional[str] = None
    debug_log_path: Optional[str] = None


@dataclass(frozen=True)
class ProviderSettings:
    endpoint: str
    model: str


@dataclass(frozen=True)
class ConfigLoadResult:
    path: Path
    config: UserConfig
    error: Optional[str]
    loaded: bool


DEFAULT_CONFIG_TEMPLATE = """[core]
provider = lmstudio
mode = normal
default_prompt =
headless = false
multiline = false
silent_tools = false
no_stats = false
max_turns =
workdir =

[lmstudio]
endpoint = http://localhost:1234/v1/chat/completions
model = qwen3-4b-instruct
context_window_tokens =

[openrouter]
endpoint = https://openrouter.ai/api/v1/chat/completions
model =
context_window_tokens =
http_referer =
app_title =
; API key can be set directly or via env var (api_key takes precedence)
api_key =
api_key_env = OPENROUTER_API_KEY
free_default_model =
free_blacklist =

[generation]
temperature = 0.2
top_p =
max_tokens =

[tool_output]
max_tool_lines = 8
max_tool_chars = 400

[debug]
log_path =
"""


def resolve_default_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        base = Path(config_home)
    elif os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / ".config"
    else:
        base = Path.home() / ".config"
    return (base / "agents" / "lmao.conf").expanduser()


def pick_first_non_none(values: Iterable[Optional[T]], default: Optional[T]) -> Optional[T]:
    for value in values:
        if value is not None:
            return value
    return default


def resolve_provider_settings(
    provider: ProviderName,
    *,
    cli_endpoint: Optional[str],
    cli_model: Optional[str],
    config: UserConfig,
    env: Optional[Mapping[str, str]] = None,
    lmstudio_default_endpoint: str,
    lmstudio_default_model: str,
    openrouter_default_endpoint: str,
) -> ProviderSettings:
    environment = env or os.environ
    env_lmstudio_endpoint = environment.get("LM_STUDIO_URL")
    env_lmstudio_model = environment.get("LM_STUDIO_MODEL")
    env_openrouter_model = environment.get("OPENROUTER_MODEL")

    if provider == "lmstudio":
        endpoint = pick_first_non_none(
            (
                cli_endpoint,
                env_lmstudio_endpoint,
                config.lmstudio_endpoint,
            ),
            lmstudio_default_endpoint,
        )
        assert endpoint is not None
        model = pick_first_non_none(
            (
                cli_model,
                env_lmstudio_model,
                config.lmstudio_model,
            ),
            lmstudio_default_model,
        )
        assert model is not None
        return ProviderSettings(endpoint=endpoint, model=model)

    endpoint = pick_first_non_none(
        (
            cli_endpoint,
            config.openrouter_endpoint,
        ),
        openrouter_default_endpoint,
    )
    assert endpoint is not None
    model = pick_first_non_none(
        (
            cli_model,
            env_openrouter_model,
            config.openrouter_model,
        ),
        None,
    )
    if model is None:
        raise ValueError(
            "Missing --model for provider openrouter (e.g. openai/gpt-4o-mini)."
        )
    return ProviderSettings(endpoint=endpoint, model=model)


def resolve_openrouter_headers(
    *,
    cli_referer: Optional[str],
    cli_title: Optional[str],
    config: UserConfig,
    env: Optional[Mapping[str, str]] = None,
) -> tuple[Optional[str], Optional[str]]:
    environment = env or os.environ
    referer = pick_first_non_none(
        (
            cli_referer,
            environment.get("OPENROUTER_HTTP_REFERER"),
            config.openrouter_http_referer,
        ),
        None,
    )
    title = pick_first_non_none(
        (
            cli_title,
            environment.get("OPENROUTER_APP_TITLE"),
            config.openrouter_app_title,
        ),
        None,
    )
    return referer, title


def resolve_openrouter_api_key(
    *,
    cli_value: Optional[str],
    config: UserConfig,
    env: Optional[Mapping[str, str]] = None,
    default_env_var: str = "OPENROUTER_API_KEY",
) -> Optional[str]:
    if cli_value:
        return cli_value
    if config.openrouter_api_key:
        return config.openrouter_api_key
    environment = env or os.environ
    env_var = config.openrouter_api_key_env or default_env_var
    return environment.get(env_var)


def load_user_config(path: Path) -> ConfigLoadResult:
    parser = configparser.ConfigParser()
    try:
        with path.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except FileNotFoundError:
        return ConfigLoadResult(
            path=path, config=UserConfig(), error=None, loaded=False
        )
    except Exception as exc:  # pragma: no cover - unexpected I/O error
        return ConfigLoadResult(
            path=path,
            config=UserConfig(),
            error=f"Failed to read config file: {exc}",
            loaded=False,
        )

    try:
        config = UserConfig(
            provider=_read_provider_name(parser, "core", "provider"),
            mode=_read_string(parser, "core", "mode"),
            default_prompt=_read_string(parser, "core", "default_prompt"),
            headless=_read_bool(parser, "core", "headless"),
            multiline=_read_bool(parser, "core", "multiline"),
            silent_tools=_read_bool(parser, "core", "silent_tools"),
            no_stats=_read_bool(parser, "core", "no_stats"),
            max_turns=_read_int(parser, "core", "max_turns"),
            workdir=_read_string(parser, "core", "workdir"),
            temperature=_read_float(parser, "generation", "temperature"),
            top_p=_read_float(parser, "generation", "top_p"),
            max_tokens=_read_int(parser, "generation", "max_tokens"),
            max_tool_lines=_read_int(parser, "tool_output", "max_tool_lines"),
            max_tool_chars=_read_int(parser, "tool_output", "max_tool_chars"),
            lmstudio_endpoint=_read_string(parser, "lmstudio", "endpoint"),
            lmstudio_model=_read_string(parser, "lmstudio", "model"),
            lmstudio_context_window_tokens=_read_int(parser, "lmstudio", "context_window_tokens"),
            openrouter_endpoint=_read_string(parser, "openrouter", "endpoint"),
            openrouter_model=_read_string(parser, "openrouter", "model"),
            openrouter_context_window_tokens=_read_int(
                parser, "openrouter", "context_window_tokens"
            ),
            openrouter_http_referer=_read_string(parser, "openrouter", "http_referer"),
            openrouter_app_title=_read_string(parser, "openrouter", "app_title"),
            openrouter_api_key=_read_string(parser, "openrouter", "api_key"),
            openrouter_api_key_env=_read_string(parser, "openrouter", "api_key_env"),
            openrouter_free_default_model=_read_string(
                parser, "openrouter", "free_default_model"
            ),
            openrouter_free_blacklist=_read_list(
                parser, "openrouter", "free_blacklist"
            ),
            debug_log_path=_read_string(parser, "debug", "log_path"),
        )
    except ValueError as exc:
        return ConfigLoadResult(
            path=path,
            config=UserConfig(),
            error=str(exc),
            loaded=False,
        )
    return ConfigLoadResult(path=path, config=config, error=None, loaded=True)


def write_default_config(path: Path) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    return True


def _read_string(
    parser: configparser.ConfigParser, section: str, option: str
) -> Optional[str]:
    if not parser.has_section(section) or not parser.has_option(section, option):
        return None
    raw = parser.get(section, option)
    if raw is None:
        return None
    value = raw.strip()
    return value if value else None


def _read_provider_name(
    parser: configparser.ConfigParser, section: str, option: str
) -> Optional[ProviderName]:
    value = _read_string(parser, section, option)
    if value is None:
        return None
    normalized = value.lower()
    if normalized == "lmstudio":
        return "lmstudio"
    if normalized == "openrouter":
        return "openrouter"
    raise ValueError(f"Invalid provider value for [{section}] {option}: {value}")


def _read_list(
    parser: configparser.ConfigParser, section: str, option: str
) -> tuple[str, ...]:
    raw = _read_string(parser, section, option)
    if raw is None:
        return ()
    entries = []
    for line in raw.replace(",", "\n").splitlines():
        value = line.strip()
        if value:
            entries.append(value)
    return tuple(entries)


def _read_bool(
    parser: configparser.ConfigParser, section: str, option: str
) -> Optional[bool]:
    value = _read_string(parser, section, option)
    if value is None:
        return None
    lower = value.lower()
    if lower in {"1", "true", "yes", "on"}:
        return True
    if lower in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean for [{section}] {option}: {value}")


def _read_int(
    parser: configparser.ConfigParser, section: str, option: str
) -> Optional[int]:
    value = _read_string(parser, section, option)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for [{section}] {option}: {value}") from exc


def _read_float(
    parser: configparser.ConfigParser, section: str, option: str
) -> Optional[float]:
    value = _read_string(parser, section, option)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for [{section}] {option}: {value}") from exc
