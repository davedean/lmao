# By Mistral: Devstral 2 2512 

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from lmao.plugins import PLUGIN_API_VERSION

# In-memory storage for the scratchpad content
_scratchpad_content = ""

PLUGIN = {
    "name": "scratchpad",
    "description": "Allows writing to and reading from a temporary in-memory text file.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "v2 args: {'action': 'write'|'read'|'clear', 'content': str (for write)}",
    "usage": [
        "{\"tool\":\"scratchpad\",\"target\":\"\",\"args\":{\"action\":\"write\",\"content\":\"Hello, world!\"}}",
        "{\"tool\":\"scratchpad\",\"target\":\"\",\"args\":{\"action\":\"read\"}}",
        "{\"tool\":\"scratchpad\",\"target\":\"\",\"args\":{\"action\":\"clear\"}}",
    ],
}


def run(
    target: str,
    args: Any,
    base: Path,
    extra_roots: Sequence[Path],
    skill_roots: Sequence[Path],
    debug_logger: Optional[object] = None,
    meta: Optional[dict] = None,
) -> str:
    """Scratchpad plugin: handles write, read, and clear operations on an in-memory text file."""
    global _scratchpad_content
    
    try:
        if isinstance(args, str):
            # Handle v1 args format (if needed)
            args_dict = json.loads(args)
        else:
            args_dict = args
        
        action = args_dict.get("action", "read")
        
        if action == "write":
            content = args_dict.get("content", "")
            _scratchpad_content = content
            payload = {"tool": PLUGIN["name"], "success": True, "data": {"message": "Content written to scratchpad."}}
        elif action == "read":
            payload = {"tool": PLUGIN["name"], "success": True, "data": {"content": _scratchpad_content}}
        elif action == "clear":
            _scratchpad_content = ""
            payload = {"tool": PLUGIN["name"], "success": True, "data": {"message": "Scratchpad cleared."}}
        else:
            payload = {"tool": PLUGIN["name"], "success": False, "error": "Invalid action. Use 'write', 'read', or 'clear'."}
    except Exception as e:
        payload = {"tool": PLUGIN["name"], "success": False, "error": str(e)}
    
    return json.dumps(payload, ensure_ascii=False)
