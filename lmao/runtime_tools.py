from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .debug_log import DebugLogger
from .llm import LLMClient
from .memory import MemoryState
from .plugins import PluginTool

RuntimeToolHandler = Callable[["RuntimeContext", str, Any, Optional[dict]], str]


@dataclass(frozen=True)
class RuntimeTool:
    name: str
    description: str
    input_schema: Optional[str] = None
    usage_examples: Sequence[str] = ()
    details: Sequence[str] = ()
    is_destructive: bool = False
    allow_in_read_only: bool = True
    allow_in_normal: bool = True
    allow_in_yolo: bool = True
    handler: RuntimeToolHandler = lambda *_args, **_kwargs: ""  # overwritten in registry


@dataclass(frozen=True)
class RuntimeContext:
    client: LLMClient
    plugin_tools: Dict[str, PluginTool]
    base: Path
    extra_roots: Sequence[Path]
    skill_roots: Sequence[Path]
    yolo_enabled: bool
    read_only: bool
    headless: bool = False
    debug_logger: Optional[DebugLogger] = None
    memory_state: Optional[MemoryState] = None


def runtime_tool_allowed(tool: RuntimeTool, *, read_only: bool, yolo_enabled: bool) -> bool:
    if read_only:
        return tool.allow_in_read_only
    if yolo_enabled:
        return True
    return tool.allow_in_normal


def build_runtime_tool_registry() -> Dict[str, RuntimeTool]:
    # Local import to avoid circular deps at import time.
    from .subagents import subagent_run_tool

    tools = [
        RuntimeTool(
            name="subagent_run",
            description="Run a bounded, read-only sub-agent and return a structured result.",
            input_schema="args: {objective:string, context?:string, max_turns?:int, allowed_tools_profile?:'read_only'}; target may be used as objective when args.objective is omitted",
            usage_examples=(
                '{"tool":"subagent_run","target":"","args":{"objective":"Wait 10 seconds then report back.","max_turns":6,"allowed_tools_profile":"read_only"}}',
                '{"tool":"subagent_run","target":"Wait 10 seconds then report back.","args":""}',
            ),
            details=(
                "Sub-agents run read-only and cannot spawn nested sub-agents.",
                "Use the read-only `sleep` tool inside the sub-agent to wait.",
            ),
            is_destructive=False,
            allow_in_read_only=True,
            allow_in_normal=True,
            allow_in_yolo=True,
            handler=subagent_run_tool,
        ),
    ]
    return {tool.name: tool for tool in tools}
