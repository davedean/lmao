# LMAO (LM Studio Agent Operator)

Tiny Python loop that lets a local LM Studio model act as a file-editing agent with tools for inspecting and modifying a repo.

## Quickstart
- Install LM Studio and start the Qwen endpoint (default assumed: `http://localhost:1234/v1/chat/completions`).
- Confirm flags and defaults: `python -m lmao --help`.
- Run from any working directory (tools are sandboxed to that directory by default):
  ```bash
  python -m lmao "describe this repo" \
    --endpoint http://localhost:1234/v1/chat/completions \
    --model qwen3-4b-instruct
  ```
- Or point at a file prompt:
  ```bash
  python -m lmao --prompt-file ./prompt.txt
  ```

## Behavior & Tools
- Default tools: `read`, `write`, `mkdir`, `move`, `ls`, `find`, `grep`, `list_skills`, plus task-list helpers (`add_task`, `complete_task`, `delete_task`, `list_tasks`). Git tools (`git_add`, `git_commit`) are available only with `--allow-git`. Tool outputs are JSON (`success` + `data`/`error`) to keep paths with spaces unambiguous.
- Optional tool: `bash` is disabled by default; enable with `--yolo` and each command will ask for interactive confirmation.
- Path safety: all tool paths are constrained to the working directory. User skill folders under `~/.config/agents/skills` are also allowed when present.
- Task lists: each run starts with an active list (seeded with “create a plan to respond”). The agent is expected to keep the list in sync while it works instead of pausing for confirmation.

## Skills & AGENTS
- Skills live in `skills/<skill-name>/SKILL.md` with YAML frontmatter; supporting files stay in the same folder. User-specific skills can live in `~/.config/agents/skills/<skill-name>/SKILL.md`.
- AGENTS discovery walks up from the working directory to the repo root to find the nearest `AGENTS.md`. User-level notes in `~/.config/agents/AGENTS.md` are appended; the nearest repo AGENTS takes precedence on conflicts.

## CLI Flags (excerpt)
- Core: `--endpoint`, `--model`, `--temperature`, `--top-p`, `--max-tokens`, `--workdir`
- Safety: `--allow-git` (git tools), `--yolo` (unsafe bash with per-command approval)
- Loop control: `--max-turns`, `--silent-tools`, `--max-tool-lines`, `--max-tool-chars`
- Prompting: `--prompt-file` (seed long prompts), optional interactive prompt when the positional prompt is omitted
- Debugging: `--debug` writes verbose loop/tool/model traces to `debug.log` in the working directory

Environment defaults: `LM_STUDIO_URL`, `LM_STUDIO_MODEL` populate `--endpoint`/`--model`.

## Safety Notes
- Git mutations are off by default; require `--allow-git` and a git repo.
- Bash is gated by `--yolo` and requires interactive approval per command.
- Moves refuse to overwrite existing destinations.
- Path resolution blocks `..`/absolute escapes outside allowed roots.

## Development
- Dependencies: standard library only (Python 3.10+ recommended).
- CI (see `.github/workflows/ci.yml`) runs `python -m compileall lmao` as a quick sanity check.
