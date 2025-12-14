# Repository Guidelines

## Project Structure & Module Organization
- `agent_loop.py`: single entrypoint that runs the LMAO (LM Studio Agent Operator) loop, tool definitions, AGENTS discovery, and safety checks. Skill-related helpers expect skills under `skills/<name>/SKILL.md` with YAML frontmatter and supporting files in the same folder.
- `.github/workflows/ci.yml`: CI compiles the code with `python -m compileall lmao` as a syntax sanity check; adjust the target if modules move.
- `README.md`: quickstart usage, CLI flags, and safety notes. `.gitignore` tracks local artifacts.

## Build, Test, and Development Commands
- `python -m lmao --help` — show CLI flags and defaults.
- `python -m lmao "describe this repo" --endpoint http://localhost:1234/v1/chat/completions --model qwen3-4b-instruct` — sample run; change prompt/endpoint/model as needed.
- `python -m compileall lmao` — mirrors the CI syntax check to catch parse errors early.

## Coding Style & Naming Conventions
- Python 3.10+; standard library only. Use 4-space indentation, type hints, and small, single-purpose helpers.
- Functions and variables use `snake_case`; constants use `UPPER_SNAKE_CASE` (e.g., `ALLOWED_TOOLS`). Keep argparse flags hyphenated and aligned with README examples.
- When editing the skill guide or tool prompt blocks, keep messaging concise and avoid introducing dependencies or unsafe defaults.

## Testing Guidelines
- No formal test suite yet. Run the compile check above and a local smoke test with a reachable LM Studio endpoint to verify tool calls and safety rails.
- For manual tests, prefer short prompts, `--max-turns` to bound conversations, and `--silent-tools` when you only need model output.

## Commit & Pull Request Guidelines
- Commit messages are short and imperative/present-tense (`tidying`, `added risky bash tool`). Include scope if multiple areas change.
- In PRs, describe the behavior change, note any CLI flag additions, and include example commands or transcripts for new flows. Link issues when relevant and call out safety-sensitive changes (`--allow-git`, `--yolo`, path handling).

## Security & Configuration Tips
- Git mutations stay disabled unless `--allow-git` is set. The optional `--yolo` flag enables the `bash` tool with per-command confirmation; use sparingly.
- Path safety blocks escapes outside the working directory; when adding tools, preserve those checks. User-specific notes and skills can live in `~/.config/agents/`, but repo-local `AGENTS.md` takes precedence.
