from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Optional

from .debug_log import DebugLogger

PLUGIN_API_VERSION = "1"


@dataclass
class PluginTool:
    name: str
    description: str
    input_schema: Optional[str]
    usage_examples: list[str]
    is_destructive: bool
    allow_in_read_only: bool
    allow_in_normal: bool
    allow_in_yolo: bool
    always_confirm: bool
    handler: Callable[..., str]
    path: Path


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
    for key in ("allow_in_read_only", "allow_in_normal", "allow_in_yolo"):
        value = manifest.get(key, None)
        if value is not None and not isinstance(value, bool):
            return f"{key} must be a boolean when provided"
    if manifest.get("always_confirm", None) is not None and not isinstance(manifest.get("always_confirm"), bool):
        return "always_confirm must be a boolean when provided"
    return None


def _load_module(path: Path) -> Optional[ModuleType]:
    spec = importlib.util.spec_from_file_location(f"lmao_plugin_{path.stem}_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


def load_plugin(path: Path, base: Path, debug_logger: Optional[DebugLogger] = None, allow_outside_base: bool = False) -> Optional[PluginTool]:
    try:
        resolved = path.resolve()
        if not allow_outside_base:
            resolved.relative_to(base)
    except Exception:
        if debug_logger:
            debug_logger.log("plugin.skip", f"path_outside_base={path}")
        return None
    if not resolved.name.endswith(".py") or resolved.name != "tool.py":
        return None
    module = _load_module(resolved)
    if module is None:
        if debug_logger:
            debug_logger.log("plugin.load_error", f"path={resolved}")
        return None
    manifest = getattr(module, "PLUGIN", None)
    handler = getattr(module, "run", None)
    if not manifest or not handler or not callable(handler):
        if debug_logger:
            debug_logger.log("plugin.missing_fields", f"path={resolved}")
        return None
    error = _validate_manifest(manifest)
    if error:
        if debug_logger:
            debug_logger.log("plugin.invalid_manifest", f"path={resolved} error={error}")
        return None
    try:
        is_destructive = bool(manifest.get("is_destructive", False))
        raw_usage = manifest.get("usage") or []
        if isinstance(raw_usage, str):
            usage_examples = [raw_usage.strip()] if raw_usage.strip() else []
        else:
            usage_examples = [str(item).strip() for item in raw_usage if str(item).strip()]
        tool = PluginTool(
            name=str(manifest["name"]).strip(),
            description=str(manifest["description"]).strip(),
            input_schema=str(manifest["input_schema"]).strip() if manifest.get("input_schema") else None,
            usage_examples=usage_examples,
            is_destructive=is_destructive,
            allow_in_read_only=bool(manifest.get("allow_in_read_only", not is_destructive)),
            allow_in_normal=bool(manifest.get("allow_in_normal", True)),
            allow_in_yolo=bool(manifest.get("allow_in_yolo", True)),
            always_confirm=bool(manifest.get("always_confirm", False)),
            handler=handler,
            path=resolved,
        )
        return tool
    except Exception as exc:
        if debug_logger:
            debug_logger.log("plugin.create_error", f"path={resolved} error={exc}")
        return None


def discover_plugins(plugin_dirs: Iterable[Path], base: Path, debug_logger: Optional[DebugLogger] = None, allow_outside_base: bool = False) -> Dict[str, PluginTool]:
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
            plugin = load_plugin(plugin_path, base, debug_logger=debug_logger, allow_outside_base=allow_outside_base)
            if not plugin:
                continue
            if plugin.name in plugins:
                if debug_logger:
                    debug_logger.log("plugin.duplicate", f"name={plugin.name} kept={plugins[plugin.name].path} skipped={plugin.path}")
                continue
            plugins[plugin.name] = plugin
    return plugins
