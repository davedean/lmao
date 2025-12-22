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
- OpenRouter (remote) example:
  ```bash
  export OPENROUTER_API_KEY="..."
  python -m lmao "describe this repo" \
    --provider openrouter \
    --model openai/gpt-4o-mini
  ```
- Or point at a file prompt:
  ```bash
  python -m lmao --prompt-file ./prompt.txt
  ```

## Docker Usage

For containerized deployment, see [DOCKER.md](./DOCKER.md) for complete Docker setup and usage instructions.

## Persistent configuration
Per-user defaults can live in `${XDG_CONFIG_HOME:-~/.config}/agents/lmao.conf` (`%APPDATA%\agents\lmao.conf` on Windows). When running, flag values come first, then environment variables, then the config file, and finally the bundled defaults. Use `--config PATH` to change the config file location, `--no-config` to ignore stored preferences for a single run, or `--print-config` to show the resolved settings (secrets are redacted) before exiting. Run `python -m lmao --config-init` to write the default `lmao.conf` at the resolved path when it does not already exist.

The INI-style file holds repeatable preferences such as provider endpoints, sampling parameters, loop limits, and tool-output truncation. Store OpenRouter metadata here too (referer/title) but keep secrets in the environment; you can name the env var that holds the API key with `api_key_env` instead of storing the key itself. When the automatic free-model selector is enabled, you can also keep lightweight preferences here—`free_default_model` is tried before ranking, and `free_blacklist` lists comma- or newline-separated model IDs (or glob patterns like `allenai/olmo-3*-32b-think:free`) that should never be picked automatically.

Example `lmao.conf`:
```ini
[core]
provider = lmstudio
mode = normal
multiline = false
silent_tools = false
no_stats = false
max_turns =
workdir =

[policy]
; Startup AGENTS.md excerpt in the initial prompt.
truncate = true
truncate_chars = 2000

[lmstudio]
endpoint = http://localhost:1234/v1/chat/completions
model = qwen3-4b-instruct

[openrouter]
endpoint = https://openrouter.ai/api/v1/chat/completions
model =
http_referer =
app_title =
; Prefer env var for secrets:
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
```
Model discovery metadata (cache + health tracking) is kept in `~/.config/agents/openrouter/free_models.json` so the user-facing `lmao.conf` stays focused on preferences.

-## Behavior & Tools
- Default tools: `read`, `write`, `mkdir`, `move`, `ls`, `find`, `grep`, `list_skills`. Git tools (`git_add`, `git_commit`) ship as plugins under `lmao/tools/git-*` and are available in normal/yolo modes (blocked in `--mode readonly`). Tool outputs are JSON (`success` + `data`/`error`) to keep paths with spaces unambiguous.
- Repo instructions (AGENTS): the loop calls `policy` once at startup to include the nearest `AGENTS.md` excerpt; configure `[policy]` in `lmao.conf` or call `policy` with `offset`/`limit`/`truncate=false` for more.
- Assistant protocol: the model must respond with a single JSON object each turn (`{"type":"assistant_turn","version":"1","steps":[...]}`), with step types `think`, `tool_call`, `message`, and `end`. The loop retries if the JSON is invalid.
- Optional tool: `bash` ships as a plugin and prompts for confirmation on every command; it is available unless read-only mode is set.
- Read-only mode: pass `--mode readonly` (or legacy `--read-only`) to disable destructive tools (`write`, `mkdir`, `move`) and any plugin that opts out of read-only (git/bash by default) for inspection-only runs.
- Pluggable tools: shipped plugins under `lmao/tools` are loaded automatically (see `lmao/tools/demo-plugin/tool.py` for a minimal echo example). Additional plugin directories are not yet supported via CLI.
- Path safety: all tool paths are constrained to the working directory. User skill folders under `~/.config/agents/skills` are also allowed when present.
- Headless mode: `--headless` (or `headless = true` in `lmao.conf`) runs without interactive prompts. Provide a prompt via the positional argument, `--prompt-file`, or `default_prompt` in the config; the loop enforces that the agent skips clarification requests and emits a final summary before ending.

## Skills & AGENTS
- Skills live in `skills/<skill-name>/SKILL.md` with YAML frontmatter; supporting files stay in the same folder. User-specific skills can live in `~/.config/agents/skills/<skill-name>/SKILL.md`.
- AGENTS discovery walks up from the working directory to the repo root to find the nearest `AGENTS.md`. User-level notes in `~/.config/agents/AGENTS.md` are appended; the nearest repo AGENTS takes precedence on conflicts.

## Plugin Tools
- Shipped plugins under `lmao/tools` are loaded automatically. Each plugin file should be named `tool.py`.
- Convention: keep plugins under `lmao/tools/<plugin-name>/tool.py` to avoid mixing with skills.
- Manifest shape: `PLUGIN = {"name": "my_tool", "description": "...", "api_version": PLUGIN_API_VERSION, "is_destructive": bool, "allow_in_read_only": bool, "allow_in_normal": bool, "allow_in_yolo": bool, "always_confirm": bool, "input_schema": "<freeform>"}` plus a callable `run(target, args, base, extra_roots, skill_roots, task_manager=None, debug_logger=None) -> str` that returns the usual JSON payload.
- Gating: Read-only mode blocks plugins unless `allow_in_read_only` is true. Normal mode includes plugins with `allow_in_normal`. Yolo mode allows all discovered plugins (read-only restrictions still apply). Set `always_confirm: true` to prompt the user on every call in non-yolo modes (used by bash). They must not escape the working directory; use `safe_target_path` if you manipulate paths.
- Examples: `lmao/tools/demo-plugin/tool.py` echoes the incoming target/args; git add/commit and bash also live under `lmao/tools/*`.

## CLI Flags (excerpt)
- Core: `--provider`, `--endpoint`, `--model`, `--temperature`, `--top-p`, `--max-tokens`, `--workdir`
- OpenRouter free models: `--free` (equivalent to `--model free`) enables automatic selection of a free tier model, obeying `free_default_model`/`free_blacklist` preferences and selecting from a high-scoring shortlist with weighted randomness. (Only valid with `--provider openrouter`.)
- Safety: `--mode yolo` (opt into risky flows/plugins) or `--mode readonly` (disable writes/moves/git/bash and plugins that opt out of read-only). `--yolo` remains as a deprecated alias.
- Extensibility: user-specified plugin directories are planned but not yet supported; current runs load the shipped plugins from `lmao/tools/` (inside the installed package).
- Loop control: `--max-turns`, `--silent-tools`, `--max-tool-lines`, `--max-tool-chars`
- Output: `--no-stats` (hide token/latency/bytes stats in the prompt/output)
- Prompting: `--prompt-file` (seed long prompts), optional interactive prompt when the positional prompt is omitted
- Headless runtime: `--headless` keeps the loop from prompting; pair it with an explicit prompt (or `default_prompt` in `lmao.conf`) so the agent can finish without clarification requests.
- Debugging: `--debug` writes verbose loop/tool/model traces to `debug.log` in the working directory (override the destination with `[debug]` `log_path`, relative paths are resolved under the working directory)
- Error logging: `--error-log PATH` writes tool failure records to a JSONL file (default: `error.log` in the working directory; configurable via `[errors]` `log_path`).
- Config: `--config PATH`, `--no-config`, `--print-config` (print the resolved defaults without secrets), `--config-init` (write the default config if missing)

Config precedence:
- CLI flags override environment variables, which override `lmao.conf`.

Environment defaults:
- LM Studio: `LM_STUDIO_URL`, `LM_STUDIO_MODEL`
- OpenRouter: `OPENROUTER_API_KEY` (required); `OPENROUTER_MODEL` can be a concrete ID or `free`, plus optional `OPENROUTER_HTTP_REFERER`, `OPENROUTER_APP_TITLE`

## Safety Notes
- Git and bash plugins are available by default but blocked in `--read-only`; they still require a git repo. Bash always asks for confirmation; other plugins can opt in with `always_confirm`.
- Read-only mode blocks destructive tools regardless of other flags; plugins that disallow read-only are hidden.
- Moves refuse to overwrite existing destinations.
- Path resolution blocks `..`/absolute escapes outside allowed roots.

## Development
- Dependencies: standard library only (Python 3.10+ recommended).
- CI (see `.github/workflows/ci.yml`) runs `mypy`, `python -m compileall lmao`, and `python -m unittest discover -s tests`.
