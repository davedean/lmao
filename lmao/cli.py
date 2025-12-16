from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, cast

from .config import (
    ConfigLoadResult,
    ProviderSettings,
    UserConfig,
    load_user_config,
    pick_first_non_none,
    resolve_default_config_path,
    resolve_openrouter_api_key,
    resolve_openrouter_headers,
    resolve_provider_settings,
    write_default_config,
)
from .debug_log import DebugLogger
from .llm import LLMClient, ProviderName
from .loop import run_loop
from .openrouter_free_models import (
    OpenRouterFreeModelPreferences,
    OpenRouterModelDiscovery,
    OpenRouterFreeModelSelector,
    OpenRouterModelSelectionError,
    derive_models_endpoint,
    resolve_model_cache_path,
)

LMSTUDIO_DEFAULT_ENDPOINT = os.environ.get(
    "LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions"
)
LMSTUDIO_DEFAULT_MODEL = os.environ.get("LM_STUDIO_MODEL", "qwen")
OPENROUTER_DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def read_prompt(args: argparse.Namespace) -> Optional[str]:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LMAO: tiny LM Studio agent operator with file tools")
    default_config_path = resolve_default_config_path()
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to per-user config file (default: "
            f"{default_config_path}, set --no-config to skip)"
        ),
    )
    parser.add_argument("--no-config", action="store_true", help="Ignore per-user configs")
    parser.add_argument(
        "--config-init",
        action="store_true",
        help="Create the default config file at the resolved path and exit (no overwrite).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved settings (secrets withheld) and exit",
    )
    parser.add_argument("prompt", nargs="?", help="Initial user prompt (optional; will prompt if omitted)")
    parser.add_argument(
        "--provider",
        choices=["lmstudio", "openrouter"],
        default=None,
        help="Model provider (default: lmstudio unless overridden in config)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=(
            "Chat completions endpoint URL; default depends on --provider "
            f"(lmstudio: LM_STUDIO_URL or {LMSTUDIO_DEFAULT_ENDPOINT}, openrouter: {OPENROUTER_DEFAULT_ENDPOINT})"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name; default depends on --provider (lmstudio: LM_STUDIO_MODEL or qwen, openrouter: config/env override required)",
    )
    parser.add_argument(
        "--free",
        action="store_true",
        help="Automatically select a free OpenRouter model (implies --model free when using openrouter)",
    )
    parser.add_argument("--api-key", default=None, help="API key (OpenRouter only; overrides env)")
    parser.add_argument(
        "--openrouter-referer",
        default=None,
        help="OpenRouter HTTP-Referer header (default: OPENROUTER_HTTP_REFERER or config)",
    )
    parser.add_argument(
        "--openrouter-title",
        default=None,
        help="OpenRouter X-Title header (default: OPENROUTER_APP_TITLE or config)",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: 0.2)")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top_p")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for responses")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Working directory for tools (default: current directory; config can override)",
    )
    parser.add_argument(
        "--max-tool-lines",
        type=int,
        default=None,
        help="Max lines to show from tool output summaries (default: 8)",
    )
    parser.add_argument(
        "--max-tool-chars",
        type=int,
        default=None,
        help="Max chars to show from tool output summaries (default: 400)",
    )
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum conversation turns before stopping")
    parser.add_argument("--silent-tools", action="store_true", help="Do not print tool outputs to console")
    parser.add_argument("--no-stats", action="store_true", help="Disable per-turn stats in console output")
    parser.add_argument("--prompt-file", type=str, default=None, help="Read initial prompt from file")
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Read user prompts as multiline until EOF (Ctrl-D; Windows: Ctrl-Z + Enter)",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "yolo", "ro", "readonly", "read-only"],
        default=None,
        help="Safety mode: normal/yolo/readonly (default: normal; config can set a different default)",
    )
    parser.add_argument("--yolo", action="store_true", help="Legacy: enable risky bash tool (use --mode yolo instead)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging to debug.log")
    parser.add_argument("--read-only", action="store_true", help="Legacy: disable destructive plugins (use --mode readonly instead)")
    return parser


def _config_summary(
    *,
    provider: ProviderName,
    provider_settings: tuple[str, str],
    config_path: Path,
    config_result: ConfigLoadResult,
    args: argparse.Namespace,
    temperature: float,
    top_p: Optional[float],
    max_tokens: Optional[int],
    max_tool_lines: int,
    max_tool_chars: int,
    max_turns: Optional[int],
    silent_tools: bool,
    no_stats: bool,
    multiline: bool,
    mode: str,
    read_only: bool,
    yolo_enabled: bool,
    workdir: Path,
    openrouter_referer: Optional[str],
    openrouter_title: Optional[str],
    api_key_present: bool,
) -> str:
    endpoint, model = provider_settings
    summary = {
        "provider": provider,
        "endpoint": endpoint,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "max_tool_lines": max_tool_lines,
        "max_tool_chars": max_tool_chars,
        "max_turns": max_turns,
        "silent_tools": silent_tools,
        "no_stats": no_stats,
        "multiline": multiline,
        "mode": mode,
        "read_only": read_only,
        "yolo_enabled": yolo_enabled,
        "workdir": str(workdir),
        "config_path": str(config_path),
        "config_loaded": config_result.loaded,
        "config_disabled": args.no_config,
        "config_error": config_result.error,
        "openrouter_referer": openrouter_referer,
        "openrouter_title": openrouter_title,
        "openrouter_api_key_present": api_key_present,
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    default_config_path = resolve_default_config_path()
    config_path = (
        Path(args.config).expanduser().resolve(strict=False) if args.config else default_config_path
    )
    if args.config_init:
        try:
            created = write_default_config(config_path)
        except OSError as exc:
            print(f"error writing config file: {exc}", file=sys.stderr)
            sys.exit(1)
        if created:
            print(f"Created default config at {config_path}")
        else:
            print(f"Config already exists at {config_path}; not overwriting.")
        return

    if args.no_config:
        config_result = ConfigLoadResult(path=config_path, config=UserConfig(), error=None, loaded=False)
    else:
        config_result = load_user_config(config_path)
    config = config_result.config

    provider_value = args.provider or config.provider
    provider = provider_value.lower() if provider_value else "lmstudio"
    provider_name = cast(ProviderName, provider)

    if args.free and provider_name != "openrouter":
        parser.error("--free is only valid with the openrouter provider")
    if args.free and args.model and args.model.lower() != "free":
        parser.error("--free cannot be combined with a concrete --model value")

    cli_model = args.model
    if provider_name == "openrouter" and args.free and not cli_model:
        cli_model = "free"
    try:
        provider_settings = resolve_provider_settings(
            provider=provider_name,
            cli_endpoint=args.endpoint,
            cli_model=cli_model,
            config=config,
            env=os.environ,
            lmstudio_default_endpoint=LMSTUDIO_DEFAULT_ENDPOINT,
            lmstudio_default_model=LMSTUDIO_DEFAULT_MODEL,
            openrouter_default_endpoint=OPENROUTER_DEFAULT_ENDPOINT,
        )
    except ValueError as exc:
        parser.error(str(exc))

    temperature = pick_first_non_none((args.temperature, config.temperature), 0.2)
    top_p = pick_first_non_none((args.top_p, config.top_p), None)
    max_tokens = pick_first_non_none((args.max_tokens, config.max_tokens), None)
    max_tool_lines = pick_first_non_none((args.max_tool_lines, config.max_tool_lines), 8)
    max_tool_chars = pick_first_non_none((args.max_tool_chars, config.max_tool_chars), 400)
    max_turns = args.max_turns if args.max_turns is not None else config.max_turns

    silent_tools = args.silent_tools or bool(config.silent_tools)
    no_stats = args.no_stats or bool(config.no_stats)
    multiline = args.multiline or bool(config.multiline)

    mode_value = args.mode or config.mode or "normal"
    mode = mode_value.lower()
    mode_read_only = mode in ("ro", "readonly", "read-only")
    mode_yolo = mode == "yolo"
    if mode != "normal" and (args.read_only or args.yolo):
        parser.error("Use --mode without --read-only/--yolo; the new flag replaces the legacy ones.")
    read_only = mode_read_only or (mode == "normal" and args.read_only)
    yolo_enabled = mode_yolo or (mode == "normal" and args.yolo)
    if read_only and yolo_enabled:
        parser.error("Cannot enable both read-only and yolo modes.")

    workdir_value = args.workdir or config.workdir
    base_dir = (
        Path(workdir_value).expanduser().resolve(strict=False)
        if workdir_value
        else Path.cwd().resolve(strict=False)
    )

    debug_logger = DebugLogger(base_dir / "debug.log") if args.debug else None
    if debug_logger and config_result.error:
        debug_logger.log("config.load_error", config_result.error)

    openrouter_referer, openrouter_title = resolve_openrouter_headers(
        cli_referer=args.openrouter_referer,
        cli_title=args.openrouter_title,
        config=config,
        env=os.environ,
    )

    api_key = resolve_openrouter_api_key(
        cli_value=args.api_key,
        config=config,
        env=os.environ,
    )

    free_selection_requested = (
        provider_name == "openrouter"
        and provider_settings.model
        and provider_settings.model.lower() == "free"
    )
    if free_selection_requested:
        if not api_key:
            parser.error("OpenRouter API key is required for automatic free model selection.")
        preferences = OpenRouterFreeModelPreferences(
            default_model=config.openrouter_free_default_model,
            blacklist=config.openrouter_free_blacklist,
        )
        discovery = OpenRouterModelDiscovery(
            models_endpoint=derive_models_endpoint(provider_settings.endpoint),
            api_key=api_key,
            cache_path=resolve_model_cache_path(),
        )
        selector = OpenRouterFreeModelSelector(
            discovery=discovery,
            preferences=preferences,
            completions_endpoint=provider_settings.endpoint,
            api_key=api_key,
            openrouter_referer=openrouter_referer,
            openrouter_title=openrouter_title,
        )
        try:
            chosen_model = selector.select_model()
        except OpenRouterModelSelectionError as exc:
            parser.error(str(exc))
        provider_settings = ProviderSettings(
            endpoint=provider_settings.endpoint,
            model=chosen_model.model_id,
        )
        print(
            f"[info] Automatically selected OpenRouter free model: {chosen_model.model_id}",
            file=sys.stderr,
        )

    if args.print_config:
        print(
            _config_summary(
                provider=provider_name,
                provider_settings=(provider_settings.endpoint, provider_settings.model),
                config_path=config_path,
                config_result=config_result,
                args=args,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_tool_lines=max_tool_lines,
                max_tool_chars=max_tool_chars,
                max_turns=max_turns,
                silent_tools=silent_tools,
                no_stats=no_stats,
                multiline=multiline,
                mode=mode,
                read_only=read_only,
                yolo_enabled=yolo_enabled,
                workdir=base_dir,
                openrouter_referer=openrouter_referer,
                openrouter_title=openrouter_title,
                api_key_present=bool(api_key),
            )
        )
        return

    initial_prompt = read_prompt(args)

    client = LLMClient(
        endpoint=provider_settings.endpoint,
        model=provider_settings.model,
        provider=provider_name,
        api_key=api_key,
        openrouter_referer=openrouter_referer,
        openrouter_title=openrouter_title,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        debug_logger=debug_logger,
    )

    try:
        built_in_plugins_dir = Path(__file__).resolve().parent / "tools"
        run_loop(
            initial_prompt=initial_prompt,
            client=client,
            workdir=base_dir,
            max_tool_output=(max_tool_lines, max_tool_chars),
            max_turns=max_turns,
            silent_tools=silent_tools,
            yolo_enabled=yolo_enabled,
            read_only=read_only,
            show_stats=not no_stats,
            multiline=multiline,
            plugin_dirs=[built_in_plugins_dir],
            debug_logger=debug_logger,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
