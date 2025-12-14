# LMAO (LM Studio Agent Operator)

Tiny Python loop that lets a local LM Studio model act as a file-editing agent with tools for inspecting and modifying a repo.

## Quickstart
- Install LM Studio and start the Qwen endpoint (default assumed: `http://localhost:1234/v1/chat/completions`).
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

## Key Features
- Tools: `read`, `write`, `mkdir`, `move`, `find`, `ls`, `grep`, `git_add`, `git_commit` (git tools gated by `--allow-git`); tool outputs are JSON (`success` + `data`/`error`) to keep paths with spaces unambiguous.
- Path safety: operations are constrained to the working directory plus user skills (`~/.config/agents/skills`) when present.
- Skills: each skill lives in `skills/<skill-name>/SKILL.md` with YAML frontmatter; supporting files are allowed inside the skill folder.
- AGENTS discovery: finds the nearest `AGENTS.md` from the task path up to repo root; user-level `~/.config/agents/AGENTS.md` is appended; nearest repo AGENTS takes precedence when conflicting.
- CLI ergonomics: prompt file support, configurable sampling (`temperature`, `top_p`, `max_tokens`), tool output summarization, max turns, silent tool-output mode, and optional debug logging to `debug.log`.

## CLI Flags (excerpt)
- `--endpoint`, `--model`, `--temperature`, `--top-p`, `--max-tokens`
- `--workdir` (default: current directory)
- `--allow-git` (enable git add/commit)
- `--max-tool-lines`, `--max-tool-chars`, `--max-turns`, `--silent-tools`
- `--prompt-file` (seed long prompts from a file)
- `--debug` (write verbose loop/tool/model traces to `debug.log` in the working directory)

Environment variables: `LM_STUDIO_URL`, `LM_STUDIO_MODEL` act as defaults for endpoint/model.

## Safety Notes
- Git mutations are off by default; require `--allow-git` and a git repo.
- Moves refuse to overwrite existing destinations.
- Path resolution blocks `..`/absolute escapes outside allowed roots.

## Development
- Dependencies: standard library only (Python 3.10+ recommended).
- CI (see `.github/workflows/ci.yml`) runs `python -m compileall lmao` as a quick sanity check.
