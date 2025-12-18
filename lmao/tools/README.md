# Plugin guide

Built-in plugins live under `lmao/tools/<plugin-name>/tool.py` and are auto-loaded at runtime (alongside any extra plugin dirs passed to the loop). Each plugin file must expose:

- `PLUGIN` manifest with keys:
  - `name` (string, <=64 chars, letters/numbers/[_-])
  - `description` (string)
  - `api_version` (`PLUGIN_API_VERSION`)
  - `is_destructive` (bool)
  - `allow_in_read_only` (bool; defaults to `not is_destructive`)
  - `allow_in_normal` (bool; defaults to `True`)
  - `allow_in_yolo` (bool; defaults to `True`; ignored when yolo mode is enabled)
  - `always_confirm` (bool; defaults to `False`; when true, the user must approve each run in non-yolo modes)
  - `input_schema` (optional string description)
  - `usage` (optional string or list of example tool JSON payloads for the `call` object; the system prompt renders them wrapped in a `{"type":"tool_call","call": ...}` step)
- `run(target, args, base, extra_roots, skill_roots, task_manager=None, debug_logger=None) -> str` that returns a JSON string with `{"tool": name, "success": bool, "data"|"error": ...}`.

Mode gating:
- Read-only mode blocks plugins unless `allow_in_read_only` is true (regardless of yolo).
- Normal mode includes plugins with `allow_in_normal`.
- Yolo mode allows all discovered plugins (read-only restrictions still apply).
- If `always_confirm` is true, the user is prompted before the plugin runs in non-yolo modes (used by bash).

Safety tips:
- Keep paths inside allowed roots; reuse `safe_target_path(target, base, extra_roots)` if you touch the filesystem.
- Long-running or risky plugins should ask for confirmation themselves (similar to the bash tool).
- Destructive plugins should usually set `allow_in_read_only=False`.
- Shared helpers: import from `lmao.plugin_helpers` to avoid reimplementing sandbox checks and text parsing, e.g., `from lmao.plugin_helpers import safe_target_path, normalize_path_for_output, parse_line_range, validate_skill_write_target, find_repo_root`.

Examples:
- Core plugins: file ops (read/write/mkdir/move/ls/find/grep), task tools, git add/commit, and bash all live under `lmao/tools/*`.
- Patch plugin (`lmao/tools/patch`) updates a line range in an existing file without rewriting the entire file content.
- Git plugins (`lmao/tools/git-add`, `lmao/tools/git-commit`) allow normal/yolo, block read-only.
- A hypothetical delete plugin could set `allow_in_normal=False`, `allow_in_yolo=True`, and `always_confirm=True` to prompt before executing.
