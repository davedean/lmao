# Plugin guide

Built-in plugins live under `lmao/tools/<plugin-name>/tool.py` and are auto-loaded at runtime (alongside any extra plugin dirs passed to the loop). Each plugin file must expose:

- `PLUGIN` manifest with keys:
  - `name` (string, <=64 chars, letters/numbers/[_-])
  - `description` (string)
  - `api_version` (`PLUGIN_API_VERSION`)
  - `is_destructive` (bool)
  - `allow_in_read_only` (bool; defaults to `not is_destructive`)
  - `allow_in_normal` (bool; defaults to `True`)
  - `allow_in_yolo` (bool; defaults to `True`; yolo mode also enables plugins allowed in normal)
  - `input_schema` (optional string description)
- `run(target, args, base, extra_roots, skill_roots, task_manager=None, debug_logger=None) -> str` that returns a JSON string with `{"tool": name, "success": bool, "data"|"error": ...}`.

Mode gating:
- Read-only mode blocks plugins unless `allow_in_read_only` is true (regardless of yolo).
- Normal mode includes plugins with `allow_in_normal`.
- Yolo mode includes plugins with `allow_in_yolo` or `allow_in_normal`; set `allow_in_normal=False, allow_in_yolo=True` for yolo-only tools (e.g., a delete command with extra confirmations).

Safety tips:
- Keep paths inside allowed roots; reuse `safe_target_path(target, base, extra_roots)` if you touch the filesystem.
- Long-running or risky plugins should ask for confirmation themselves (similar to the bash tool).
- Destructive plugins should usually set `allow_in_read_only=False`.

Examples:
- Git plugins (`lmao/tools/git-add`, `lmao/tools/git-commit`) allow normal/yolo, block read-only.
- A hypothetical delete plugin could set `allow_in_normal=False`, `allow_in_yolo=True` and prompt the user before executing.
