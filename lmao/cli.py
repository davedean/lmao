from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, cast

from .debug_log import DebugLogger
from .llm import LLMClient, ProviderName
from .loop import run_loop

LMSTUDIO_DEFAULT_ENDPOINT = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
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
    parser.add_argument("prompt", nargs="?", help="Initial user prompt for the agent (optional; will prompt if omitted)")
    parser.add_argument("--provider", choices=["lmstudio", "openrouter"], default="lmstudio", help="Model provider (default: lmstudio)")
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
        help=f"Model name; default depends on --provider (lmstudio: LM_STUDIO_MODEL or {LMSTUDIO_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (OpenRouter: overrides OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--openrouter-referer",
        default=os.environ.get("OPENROUTER_HTTP_REFERER"),
        help="OpenRouter HTTP-Referer header value (default: OPENROUTER_HTTP_REFERER)",
    )
    parser.add_argument(
        "--openrouter-title",
        default=os.environ.get("OPENROUTER_APP_TITLE"),
        help="OpenRouter X-Title header value (default: OPENROUTER_APP_TITLE)",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top_p")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for responses")
    parser.add_argument("--workdir", default=None, help="Working directory for tools (default: current working directory)")
    parser.add_argument("--max-tool-lines", type=int, default=8, help="Max lines to show from tool output in console summaries (0 for none)")
    parser.add_argument("--max-tool-chars", type=int, default=400, help="Max chars to show from tool output in console summaries (0 for none)")
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum conversation turns before stopping")
    parser.add_argument("--silent-tools", action="store_true", help="Do not print tool outputs to console (still sent to model)")
    parser.add_argument("--no-stats", action="store_true", help="Disable per-turn stats (tokens, latency, bytes) in the console prompt/output")
    parser.add_argument("--prompt-file", type=str, default=None, help="Read initial prompt from file")
    parser.add_argument("--mode", choices=["normal", "yolo", "ro", "readonly", "read-only"], default="normal", help="Safety mode: read-only disables destructive tools/plugins; yolo enables risky plugins; default: normal")
    parser.add_argument("--yolo", action="store_true", help="Legacy: enable unsafe 'bash' tool with per-command confirmation (use --mode yolo instead)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging to debug.log in the working directory")
    parser.add_argument("--read-only", action="store_true", help="Legacy: disable destructive tools and plugins that disallow read-only (use --mode readonly instead)")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    base_dir = Path(args.workdir).expanduser().resolve() if args.workdir else Path.cwd()
    debug_logger = DebugLogger(base_dir / "debug.log") if args.debug else None

    provider = str(args.provider or "lmstudio").lower()
    provider_name = cast(ProviderName, provider)
    endpoint = args.endpoint
    if not endpoint:
        endpoint = OPENROUTER_DEFAULT_ENDPOINT if provider == "openrouter" else LMSTUDIO_DEFAULT_ENDPOINT
    model = args.model
    if not model:
        if provider == "lmstudio":
            model = LMSTUDIO_DEFAULT_MODEL
        else:
            model = os.environ.get("OPENROUTER_MODEL")
            if not model:
                parser.error("Missing --model for provider openrouter (e.g. openai/gpt-4o-mini).")
    api_key = args.api_key
    if not api_key and provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")

    if debug_logger:
        debug_logger.log("cli.start", f"provider={provider} endpoint={endpoint} model={model} workdir={base_dir}")
        if args.prompt_file:
            debug_logger.log("prompt.file", f"path={Path(args.prompt_file).expanduser().resolve()}")
    initial_prompt = read_prompt(args)

    mode = (args.mode or "normal").lower()
    mode_read_only = mode in ("ro", "readonly", "read-only")
    mode_yolo = mode == "yolo"
    if mode != "normal" and (args.read_only or args.yolo):
        parser.error("Use --mode without --read-only/--yolo; the new flag replaces the legacy ones.")
    read_only = mode_read_only or (mode == "normal" and args.read_only)
    yolo_enabled = mode_yolo or (mode == "normal" and args.yolo)
    if read_only and yolo_enabled:
        parser.error("Cannot enable both read-only and yolo modes.")

    client = LLMClient(
        endpoint=endpoint,
        model=model,
        provider=provider_name,
        api_key=api_key,
        openrouter_referer=args.openrouter_referer,
        openrouter_title=args.openrouter_title,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        debug_logger=debug_logger,
    )

    try:
        # Locate the built-in tools directory next to the installed lmao package.
        built_in_plugins_dir = Path(__file__).resolve().parent / "tools"
        run_loop(
            initial_prompt=initial_prompt,
            client=client,
            workdir=base_dir,
            max_tool_output=(args.max_tool_lines, args.max_tool_chars),
            max_turns=args.max_turns,
            silent_tools=args.silent_tools,
            yolo_enabled=yolo_enabled,
            read_only=read_only,
            show_stats=not args.no_stats,
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
