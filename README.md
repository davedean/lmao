# LMAO (LM Studio Agent Operator)

Tiny Python loop that lets a local LM Studio model act as a file-editing agent with tools for inspecting and modifying a repo.

## Why
- Built to use a small local model for quick tasks like note processing or tiny coding fixes.
- Keeps work private while staying fast and capable on everyday prompts.

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
- Default tools: `read`, `write`, `mkdir`, `move`, `ls`, `find`, `grep`, `list_skills`, plus task-list helpers (`add_task`, `complete_task`, `delete_task`, `list_tasks`). Git tools (`git_add`, `git_commit`) ship as plugins under `lmao/tools/git-*` and are available in normal/yolo modes (blocked in `--mode readonly`). Tool outputs are JSON (`success` + `data`/`error`) to keep paths with spaces unambiguous.
- Assistant protocol: the model must respond with a single JSON object each turn (`{"type":"assistant_turn","version":"1","steps":[...]}`), with step types `think`, `tool_call`, `message`, and `end`. The loop retries if the JSON is invalid and blocks `end` until the task list is complete.
- Optional tool: `bash` ships as a plugin and prompts for confirmation on every command; it is available unless read-only mode is set.
- Read-only mode: pass `--mode readonly` (or legacy `--read-only`) to disable destructive tools (`write`, `mkdir`, `move`) and any plugin that opts out of read-only (git/bash by default) for inspection-only runs.
- Pluggable tools: shipped plugins under `lmao/tools` are loaded automatically (see `lmao/tools/demo-plugin/tool.py` for a minimal echo example). Additional plugin directories are not yet supported via CLI.
- Path safety: all tool paths are constrained to the working directory. User skill folders under `~/.config/agents/skills` are also allowed when present.
- Task lists: each run starts with an active list (empty by default). If the model adds tasks, it is expected to keep the list in sync while it works instead of pausing for confirmation.

## Skills & AGENTS
- Skills live in `skills/<skill-name>/SKILL.md` with YAML frontmatter; supporting files stay in the same folder. User-specific skills can live in `~/.config/agents/skills/<skill-name>/SKILL.md`.
- AGENTS discovery walks up from the working directory to the repo root to find the nearest `AGENTS.md`. User-level notes in `~/.config/agents/AGENTS.md` are appended; the nearest repo AGENTS takes precedence on conflicts.

## Plugin Tools
- Shipped plugins under `lmao/tools` are loaded automatically. Each plugin file should be named `tool.py`.
- Convention: keep plugins under `lmao/tools/<plugin-name>/tool.py` to avoid mixing with skills.
- Manifest shape: `PLUGIN = {"name": "my_tool", "description": "...", "api_version": PLUGIN_API_VERSION, "is_destructive": bool, "allow_in_read_only": bool, "allow_in_normal": bool, "allow_in_yolo": bool, "always_confirm": bool, "input_schema": "<freeform>"}` plus a callable `run(target, args, base, extra_roots, skill_roots, task_manager=None, debug_logger=None) -> str` that returns the usual JSON payload.
- Gating: plugin allow-listing follows the allow flags above. Read-only mode blocks plugins unless `allow_in_read_only` is true. Yolo mode includes plugins allowed in normal plus any that require yolo (`allow_in_normal=False, allow_in_yolo=True`). Set `always_confirm: true` to prompt the user on every call (used by bash). They must not escape the working directory; use `safe_target_path` if you manipulate paths.
- Examples: `lmao/tools/demo-plugin/tool.py` echoes the incoming target/args; task tools, git add/commit, and bash also live under `lmao/tools/*`.

## CLI Flags (excerpt)
- Core: `--endpoint`, `--model`, `--temperature`, `--top-p`, `--max-tokens`, `--workdir`
- Safety: `--mode yolo` (opt into risky flows/plugins) or `--mode readonly` (disable writes/moves/git/bash and plugins that opt out of read-only); legacy `--yolo`/`--read-only` remain for compatibility
- Extensibility: user-specified plugin directories are planned but not yet supported; current runs load the shipped plugins from `lmao/tools/` (inside the installed package).
- Loop control: `--max-turns`, `--silent-tools`, `--max-tool-lines`, `--max-tool-chars`
- Output: `--no-stats` (hide token/latency/bytes stats in the prompt/output)
- Prompting: `--prompt-file` (seed long prompts), optional interactive prompt when the positional prompt is omitted
- Debugging: `--debug` writes verbose loop/tool/model traces to `debug.log` in the working directory

Environment defaults: `LM_STUDIO_URL`, `LM_STUDIO_MODEL` populate `--endpoint`/`--model`.

## Safety Notes
- Git and bash plugins are available by default but blocked in `--read-only`; they still require a git repo. Bash always asks for confirmation; other plugins can opt in with `always_confirm`.
- Read-only mode blocks destructive tools regardless of other flags; plugins that disallow read-only are hidden.
- Moves refuse to overwrite existing destinations.
- Path resolution blocks `..`/absolute escapes outside allowed roots.

## Development
- Dependencies: standard library only (Python 3.10+ recommended).
- CI (see `.github/workflows/ci.yml`) runs `python -m compileall lmao` and `python -m unittest discover -s tests`.
