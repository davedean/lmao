from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, cast

from .debug_log import DebugLogger
from .hooks import HookRegistry

PLUGIN_API_VERSION = "1"

_DISCOVERED_TOOL_REGISTRY: Dict[str, "PluginTool"] = {}


@dataclass
class PluginTool:
    name: str
    description: str
    input_schema: Optional[str]
    usage_examples: list[str]
    details: list[str]
    is_destructive: bool
    allow_in_read_only: bool
    allow_in_normal: bool
    allow_in_yolo: bool
    always_confirm: bool
    handler: Callable[..., str]
    path: Path


def get_discovered_tool_registry() -> Dict[str, "PluginTool"]:
    """
    Read-only view of the most recently discovered plugin tool registry (name -> PluginTool).

    This is intended for helper tools like `tools_guide` that need access to tool metadata without
    requiring protocol changes or threading plugin registries through every handler signature.
    """
    return dict(_DISCOVERED_TOOL_REGISTRY)


def _validate_manifest(manifest: Dict[str, Any]) -> Optional[str]:
    if not isinstance(manifest, dict):
        return "manifest must be a dict"
    api_version = str(manifest.get("api_version", "") or PLUGIN_API_VERSION)
    if api_version != PLUGIN_API_VERSION:
        return f"incompatible api_version (expected {PLUGIN_API_VERSION}, got {api_version})"
    name = manifest.get("name")
    if not isinstance(name, str) or not name.strip():
        return "name is required"
    if len(name) > 64 or not re.match(r"^[A-Za-z0-9_\-]+$", name):
        return "name must be <=64 chars and only contain letters, numbers, underscores, or hyphens"
    desc = manifest.get("description")
    if not isinstance(desc, str) or not desc.strip():
        return "description is required"
    is_destructive = manifest.get("is_destructive", False)
    if not isinstance(is_destructive, bool):
        return "is_destructive must be a boolean"
    input_schema = manifest.get("input_schema")
    if input_schema is not None and not isinstance(input_schema, str):
        return "input_schema must be a string when provided"
    usage = manifest.get("usage")
    if usage is not None and not isinstance(usage, (str, list, tuple)):
        return "usage must be a string or list of strings when provided"
    details = manifest.get("details")
    if details is not None and not isinstance(details, (str, list, tuple)):
        return "details must be a string or list of strings when provided"
    for key in ("allow_in_read_only", "allow_in_normal", "allow_in_yolo"):
        value = manifest.get(key, None)
        if value is not None and not isinstance(value, bool):
            return f"{key} must be a boolean when provided"
    if manifest.get("always_confirm", None) is not None and not isinstance(
        manifest.get("always_confirm"), bool
    ):
        return "always_confirm must be a boolean when provided"
    return None


def _load_module(path: Path) -> Optional[ModuleType]:
    spec = importlib.util.spec_from_file_location(
        f"lmao_plugin_{path.stem}_{abs(hash(path))}", path
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        # Some stdlib features (e.g., dataclasses) expect the module to be present in sys.modules
        # while executing. When running via exec_module directly, we must register it ourselves.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        return None
    return module


def _coerce_usage_examples(raw_usage: Any) -> list[str]:
    if raw_usage is None:
        return []
    if isinstance(raw_usage, str):
        return [raw_usage.strip()] if raw_usage.strip() else []
    if isinstance(raw_usage, (list, tuple)):
        return [str(item).strip() for item in raw_usage if str(item).strip()]
    return []


def _coerce_details(raw_details: Any) -> list[str]:
    if raw_details is None:
        return []
    if isinstance(raw_details, str):
        return [raw_details.strip()] if raw_details.strip() else []
    if isinstance(raw_details, (list, tuple)):
        return [str(item).strip() for item in raw_details if str(item).strip()]
    return []


def _create_plugin_tool(
    manifest: Dict[str, Any],
    handler: Callable[..., str],
    *,
    path: Path,
) -> PluginTool:
    is_destructive = bool(manifest.get("is_destructive", False))
    usage_examples = _coerce_usage_examples(manifest.get("usage"))
    details = _coerce_details(manifest.get("details"))
    return PluginTool(
        name=str(manifest["name"]).strip(),
        description=str(manifest["description"]).strip(),
        input_schema=str(manifest["input_schema"]).strip()
        if manifest.get("input_schema")
        else None,
        usage_examples=usage_examples,
        details=details,
        is_destructive=is_destructive,
        allow_in_read_only=bool(manifest.get("allow_in_read_only", not is_destructive)),
        allow_in_normal=bool(manifest.get("allow_in_normal", True)),
        allow_in_yolo=bool(manifest.get("allow_in_yolo", True)),
        always_confirm=bool(manifest.get("always_confirm", False)),
        handler=handler,
        path=path,
    )


def load_plugins(
    path: Path,
    base: Path,
    debug_logger: Optional[DebugLogger] = None,
    allow_outside_base: bool = False,
) -> List[PluginTool]:
    try:
        resolved = path.resolve()
        if not allow_outside_base:
            resolved.relative_to(base)
    except Exception:
        if debug_logger:
            debug_logger.log("plugin.skip", f"path_outside_base={path}")
        return []
    if not resolved.name.endswith(".py") or resolved.name != "tool.py":
        return []
    module = _load_module(resolved)
    if module is None:
        if debug_logger:
            debug_logger.log("plugin.load_error", f"path={resolved}")
        return []

    handler = getattr(module, "run", None)
    if not handler or not callable(handler):
        if debug_logger:
            debug_logger.log("plugin.missing_fields", f"path={resolved} missing=run")
        return []
    # After runtime validation, we treat the plugin entrypoint as returning a string tool payload.
    handler = cast(Callable[..., str], handler)

    multi_manifests = getattr(module, "PLUGINS", None)
    if multi_manifests is not None:
        if not isinstance(multi_manifests, list) or not multi_manifests:
            if debug_logger:
                debug_logger.log(
                    "plugin.invalid_manifest",
                    f"path={resolved} error=PLUGINS must be a non-empty list",
                )
            return []
        tools: List[PluginTool] = []
        for idx, manifest in enumerate(multi_manifests):
            if not isinstance(manifest, dict):
                if debug_logger:
                    debug_logger.log(
                        "plugin.invalid_manifest",
                        f"path={resolved} error=PLUGINS[{idx}] must be a dict",
                    )
                return []
            error = _validate_manifest(manifest)
            if error:
                if debug_logger:
                    debug_logger.log(
                        "plugin.invalid_manifest",
                        f"path={resolved} error=PLUGINS[{idx}] {error}",
                    )
                return []
            tool_name = str(manifest.get("name", "")).strip()
            base_handler = handler

            def _wrapped(
                target: str,
                args: Any,
                base: Path,
                extra_roots,
                skill_roots,
                task_manager=None,
                debug_logger: Optional[DebugLogger] = None,
                meta: Optional[Dict[str, Any]] = None,
                _tool_name: str = tool_name,
            ) -> str:
                try:
                    sig = inspect.signature(base_handler)
                    params = list(sig.parameters.values())
                    accepts_varargs = any(
                        p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
                    )
                    accepts_meta = accepts_varargs or len(params) >= 9
                except Exception:
                    accepts_meta = False
                if accepts_meta:
                    return base_handler(
                        _tool_name,
                        target,
                        args,
                        base,
                        extra_roots,
                        skill_roots,
                        task_manager,
                        debug_logger,
                        meta,
                    )
                return base_handler(
                    _tool_name,
                    target,
                    args,
                    base,
                    extra_roots,
                    skill_roots,
                    task_manager,
                    debug_logger,
                )

            try:
                tools.append(_create_plugin_tool(manifest, _wrapped, path=resolved))
            except Exception as exc:
                if debug_logger:
                    debug_logger.log(
                        "plugin.create_error", f"path={resolved} error={exc}"
                    )
                return []
        return tools

    manifest = getattr(module, "PLUGIN", None)
    if not manifest or not isinstance(manifest, dict):
        if debug_logger:
            debug_logger.log(
                "plugin.missing_fields", f"path={resolved} missing=PLUGIN/PLUGINS"
            )
        return []
    error = _validate_manifest(manifest)
    if error:
        if debug_logger:
            debug_logger.log(
                "plugin.invalid_manifest", f"path={resolved} error={error}"
            )
        return []
    try:
        return [_create_plugin_tool(manifest, handler, path=resolved)]
    except Exception as exc:
        if debug_logger:
            debug_logger.log("plugin.create_error", f"path={resolved} error={exc}")
        return []


def load_plugin(
    path: Path,
    base: Path,
    debug_logger: Optional[DebugLogger] = None,
    allow_outside_base: bool = False,
) -> Optional[PluginTool]:
    """
    Back-compat wrapper: load the first plugin tool from a tool.py.

    Prefer `load_plugins()` for new code, as a tool.py may define multiple virtual tools via `PLUGINS`.
    """
    tools = load_plugins(
        path, base, debug_logger=debug_logger, allow_outside_base=allow_outside_base
    )
    return tools[0] if tools else None


def discover_plugins(
    plugin_dirs: Iterable[Path],
    base: Path,
    debug_logger: Optional[DebugLogger] = None,
    allow_outside_base: bool = False,
) -> Dict[str, PluginTool]:
    plugins: Dict[str, PluginTool] = {}
    for directory in plugin_dirs:
        try:
            resolved_dir = directory.expanduser().resolve()
        except Exception:
            if debug_logger:
                debug_logger.log("plugin.dir_error", f"path={directory}")
            continue
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            if debug_logger:
                debug_logger.log("plugin.dir_missing", f"path={resolved_dir}")
            continue
        for plugin_path in resolved_dir.rglob("tool.py"):
            loaded = load_plugins(
                plugin_path,
                base,
                debug_logger=debug_logger,
                allow_outside_base=allow_outside_base,
            )
            for plugin in loaded:
                if plugin.name in plugins:
                    if debug_logger:
                        debug_logger.log(
                            "plugin.duplicate",
                            f"name={plugin.name} kept={plugins[plugin.name].path} skipped={plugin.path}",
                        )
                    continue
                plugins[plugin.name] = plugin
    global _DISCOVERED_TOOL_REGISTRY
    _DISCOVERED_TOOL_REGISTRY = dict(plugins)
    return plugins


def discover_plugin_hooks(
    plugin_dirs: Iterable[Path],
    base: Path,
    hook_registry: HookRegistry,
    debug_logger: Optional[DebugLogger] = None,
    allow_outside_base: bool = False,
) -> None:
    for directory in plugin_dirs:
        try:
            resolved_dir = directory.expanduser().resolve()
        except Exception:
            if debug_logger:
                debug_logger.log("plugin.hook_dir_error", f"path={directory}")
            continue
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            if debug_logger:
                debug_logger.log("plugin.hook_dir_missing", f"path={resolved_dir}")
            continue
        for plugin_path in resolved_dir.rglob("tool.py"):
            try:
                resolved = plugin_path.resolve()
                if not allow_outside_base:
                    resolved.relative_to(base)
            except Exception:
                if debug_logger:
                    debug_logger.log("plugin.hook_skip", f"path_outside_base={plugin_path}")
                continue
            module = _load_module(resolved)
            if module is None:
                if debug_logger:
                    debug_logger.log("plugin.hook_load_error", f"path={resolved}")
                continue
            _register_module_hooks(
                module, resolved, hook_registry, debug_logger=debug_logger
            )


def _register_module_hooks(
    module: ModuleType,
    plugin_path: Path,
    hook_registry: HookRegistry,
    debug_logger: Optional[DebugLogger] = None,
) -> None:
    hook_specs = []
    module_hooks = getattr(module, "HOOKS", None)
    if isinstance(module_hooks, dict):
        hook_specs.append(("module", module_hooks))

    manifest = getattr(module, "PLUGIN", None)
    if isinstance(manifest, dict) and manifest.get("hooks"):
        hook_specs.append((manifest.get("name", "plugin"), manifest.get("hooks")))

    multi_manifests = getattr(module, "PLUGINS", None)
    if isinstance(multi_manifests, list):
        for idx, item in enumerate(multi_manifests):
            if not isinstance(item, dict):
                continue
            hooks = item.get("hooks")
            if hooks:
                hook_specs.append((item.get("name", f"plugin_{idx}"), hooks))

    for plugin_name, hooks in hook_specs:
        _register_hook_mapping(
            hooks,
            module,
            hook_registry,
            plugin_name=str(plugin_name),
            plugin_path=plugin_path,
            debug_logger=debug_logger,
        )


def _register_hook_mapping(
    hooks: Any,
    module: ModuleType,
    hook_registry: HookRegistry,
    *,
    plugin_name: str,
    plugin_path: Path,
    debug_logger: Optional[DebugLogger] = None,
) -> None:
    if not isinstance(hooks, dict):
        if debug_logger:
            debug_logger.log(
                "plugin.hook_invalid", f"path={plugin_path} error=hooks must be a dict"
            )
        return
    for hook_type, entries in hooks.items():
        for hook_spec in _iter_hook_specs(entries):
            hook_name = hook_spec["name"]
            hook_func = getattr(module, hook_name, None)
            if not callable(hook_func):
                if debug_logger:
                    debug_logger.log(
                        "plugin.hook_missing",
                        f"path={plugin_path} hook={hook_type} missing={hook_name}",
                    )
                continue
            hook_registry.register(
                str(hook_type),
                hook_func,
                priority=hook_spec["priority"],
                name=hook_name,
                plugin_name=plugin_name,
                plugin_path=str(plugin_path),
            )
            if debug_logger:
                debug_logger.log(
                    "plugin.hook_registered",
                    f"path={plugin_path} plugin={plugin_name} hook={hook_type} name={hook_name}",
                )


def _iter_hook_specs(entries: Any) -> Iterable[Dict[str, Any]]:
    if entries is None:
        return []
    if isinstance(entries, (str, dict)):
        entries = [entries]
    if not isinstance(entries, (list, tuple)):
        return []
    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"name": entry, "priority": 0})
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("func") or entry.get("callable")
            if not name:
                continue
            priority = entry.get("priority", 0)
            normalized.append({"name": str(name), "priority": int(priority)})
    return normalized
