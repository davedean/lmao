# Repository Guidelines

## Project Structure & Module Organization
- `agent_loop.py`: single entrypoint that runs the LMAO (LM Studio Agent Operator) loop; CLI lives in `lmao/cli.py`, runtime loop in `lmao/loop.py`.
- Tools and guards live in `lmao/tools.py`; plugin loading lives in `lmao/plugins.py` and auto-loads `tool.py` files under `lmao/tools/**`.
- Context/safety scaffolding is in `lmao/context.py` (AGENTS discovery up to the repo root, user-level notes at `~/.config/agents/AGENTS.md`, skills under both repo `./skills` and optional `~/.config/agents/skills`).
- Skills belong in `skills/<name>/SKILL.md` with YAML frontmatter and supporting files in the same folder.
- `.github/workflows/ci.yml`: CI compiles the code with `python -m compileall lmao` as a syntax sanity check.
- Tests live in `tests/` (unittest-based).

## Runtime Behavior & Tools
- Default tools: read, write, mkdir, move, ls, find, grep, list_skills, plus plugins (task list tools add/complete/delete/list, git_add/git_commit, bash). Tool outputs are JSON (`success` + `data`/`error`). Do not call `list_skills` unless the user explicitly asks about skills or you need a skill path.
- Task list is always present and seeded with “create a plan to respond”; use task tools to keep it updated before replying.
- Path safety keeps targets under the working directory; writes into skill roots must stay inside `skills/<name>/`.
- Plugins: any `tool.py` under `lmao/tools/**` is loaded; plugin manifest controls if it is allowed in read-only, normal, or yolo modes, and whether it needs per-use confirmation. Core file tools, task tools, git add/commit, and bash all ship as plugins under `lmao/tools/*`; git/bash are available in normal/yolo but blocked in read-only. Bash always asks for confirmation.
- Read-only mode disables write/mkdir/move and any plugin that opts out of read-only (most destructive ones).
- Headless mode (`--headless` or `headless = true` in `lmao.conf`) runs the loop without interactive prompts; provide a prompt via CLI, `--prompt-file`, or `default_prompt` in the config, and the model will be instructed to avoid clarification requests and finish with a final summary.

## Build, Test, and Development Commands
- `python -m lmao --help` — show CLI flags and defaults.
- `python -m lmao "describe this repo" --endpoint http://localhost:1234/v1/chat/completions --model qwen3-4b-instruct` — sample run; change prompt/endpoint/model as needed.
- `python -m compileall lmao` — mirrors the CI syntax check to catch parse errors early.
- `python -m unittest discover tests` — quick unit tests (standard library only).
- Manual smoke: run against a reachable LM Studio endpoint; use `--silent-tools` when you only need model output.

## Coding Style & Naming Conventions
- Python 3.10+; standard library only. Use 4-space indentation, type hints, and small, single-purpose helpers.
- Functions and variables use `snake_case`; constants use `UPPER_SNAKE_CASE` (e.g., `ALLOWED_TOOLS`). Keep argparse flags hyphenated and aligned with README examples.
- When editing the skill guide or tool prompt blocks, keep messaging concise and avoid introducing dependencies or unsafe defaults.

## Testing Guidelines
- CI only runs `python -m compileall lmao`; locally you can add `python -m unittest discover tests` for quick coverage.
- For manual tests, prefer short prompts, `--max-turns` to bound conversations, and `--silent-tools` when you only need model output.

## Commit & Pull Request Guidelines
- Commit messages are short and imperative/present-tense (`tidying`, `added risky bash tool`). Include scope if multiple areas change.
- In PRs, describe the behavior change, note any CLI flag additions, and include example commands or transcripts for new flows. Link issues when relevant and call out safety-sensitive changes (`--yolo`, path handling).

## Security & Configuration Tips
- Git and bash plugins are loaded by default; they are blocked in read-only mode. The optional `--yolo` flag is intended for riskier flows; bash always prompts for confirmation (other plugins can opt into per-call confirmation via `always_confirm`).
- Path safety blocks escapes outside the working directory; when adding tools, preserve those checks. User-specific notes and skills can live in `~/.config/agents/`, but repo-local `AGENTS.md` takes precedence.
- Moves refuse to overwrite existing destinations. User-level AGENTS/skills are allowed roots but the nearest repo AGENTS wins on conflicts.
