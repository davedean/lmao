from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from .debug_log import DebugLogger
from .llm import LLMClient
from .loop import run_loop

DEFAULT_ENDPOINT = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("LM_STUDIO_MODEL", "qwen")


def read_prompt(args: argparse.Namespace) -> Optional[str]:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LMAO: tiny LM Studio agent operator with file tools")
    parser.add_argument("prompt", nargs="?", help="Initial user prompt for the agent (optional; will prompt if omitted)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="LM Studio chat completions endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to send to LM Studio")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top_p")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for responses")
    parser.add_argument("--workdir", default=None, help="Working directory for tools (default: current working directory)")
    parser.add_argument("--max-tool-lines", type=int, default=8, help="Max lines to show from tool output in console summaries (0 for none)")
    parser.add_argument("--max-tool-chars", type=int, default=400, help="Max chars to show from tool output in console summaries (0 for none)")
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum conversation turns before stopping")
    parser.add_argument("--silent-tools", action="store_true", help="Do not print tool outputs to console (still sent to model)")
    parser.add_argument("--prompt-file", type=str, default=None, help="Read initial prompt from file")
    parser.add_argument("--yolo", action="store_true", help="Enable unsafe 'bash' tool with per-command confirmation (off by default)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging to debug.log in the working directory")
    parser.add_argument("--read-only", action="store_true", help="Disable destructive tools (writes/moves/git/bash); for inspection-only sessions")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    base_dir = Path(args.workdir).expanduser().resolve() if args.workdir else Path.cwd()
    debug_logger = DebugLogger(base_dir / "debug.log") if args.debug else None
    if debug_logger:
        debug_logger.log("cli.start", f"endpoint={args.endpoint} model={args.model} workdir={base_dir}")
        if args.prompt_file:
            debug_logger.log("prompt.file", f"path={Path(args.prompt_file).expanduser().resolve()}")
    initial_prompt = read_prompt(args)
    client = LLMClient(
        endpoint=args.endpoint,
        model=args.model,
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
            yolo_enabled=args.yolo,
            read_only=args.read_only,
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
