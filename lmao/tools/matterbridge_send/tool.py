from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping

from lmao.matterbridge import get_matterbridge_config
from lmao.plugins import PLUGIN_API_VERSION
from lmao.matterbridge_common import (
    ToolError,
    _resolve_matterbridge_config,
    parse_timeout,
)

PLUGIN = {
    "name": "matterbridge_send",
    "description": "Send a message through a Matterbridge gateway.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "args: string (text) or object {text?:string, content?:string, gateway?:string, username?:string, avatar?:string, extra?:any, timeout?:number}",
    "usage": [
        "{\"tool\":\"matterbridge_send\",\"target\":\"\",\"args\":\"Status update from agent\"}",
        "{\"tool\":\"matterbridge_send\",\"target\":\"\",\"args\":{\"text\":\"Ready\",\"gateway\":\"gateway1\",\"username\":\"status-bot\"}}",
    ],
}


def _success(data: dict) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": False, "error": message}, ensure_ascii=False)


def _coerce_args(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return dict(args)
    if isinstance(args, str):
        return {"text": args}
    raise ToolError("matterbridge_send expects a string or object as args.")


def _extract_text(payload: Mapping[str, Any]) -> str:
    raw = payload.get("text") or payload.get("content")
    if raw is None:
        raise ToolError("matterbridge_send requires 'text' or 'content'.")
    message = str(raw)
    if not message.strip():
        raise ToolError("matterbridge_send requires non-empty text/content.")
    return message


def _build_body(message: str, gateway: str, extras: Mapping[str, Any]) -> dict[str, Any]:
    username = extras.get("username") or "lmao"
    body: dict[str, Any] = {"text": message, "gateway": gateway, "username": username}
    if "avatar" in extras:
        body["avatar"] = extras.get("avatar")
    if "extra" in extras:
        body["extra"] = extras.get("extra")
    return body


def _post_message(uri: str, payload: dict[str, Any], timeout: float) -> Any:
    endpoint = f"{uri.rstrip('/')}/api/message"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise ToolError(f"matterbridge_send failed ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise ToolError(f"matterbridge_send network error: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ToolError(f"matterbridge_send returned invalid JSON: {exc}") from exc


def run(
    target: str,
    args: Any,
    base,
    extra_roots,
    skill_roots,
    task_manager=None,
    debug_logger=None,
    meta=None,
) -> str:
    try:
        payload = _coerce_args(args)
        timeout = parse_timeout(payload.pop("timeout", None))
        context = get_matterbridge_config()
        uri, gateway = _resolve_matterbridge_config(context, payload, require_gateway=True)
        message = _extract_text(payload)
        body = _build_body(message, gateway, payload)
        response = _post_message(uri, body, timeout)
        return _success({"status": "sent", "gateway": gateway, "response": response})
    except ToolError as exc:
        return _error(str(exc))
