from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from lmao.matterbridge import get_matterbridge_config
from lmao.plugins import PLUGIN_API_VERSION
from lmao.tools.matterbridge_common import (
    ToolError,
    _resolve_matterbridge_config,
    parse_timeout,
)

PLUGIN = {
    "name": "matterbridge_read",
    "description": "Fetch buffered or streaming updates from a Matterbridge gateway.",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "input_schema": "args: {limit?:number, stream?:bool, timeout?:number, gateway?:string}",
    "usage": [
        "{\"tool\":\"matterbridge_read\",\"target\":\"\",\"args\":{\"limit\":5}}",
        "{\"tool\":\"matterbridge_read\",\"target\":\"\",\"args\":{\"stream\":true,\"limit\":1,\"timeout\":5}}",
    ],
}


def _success(data: dict) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": True, "data": data}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"tool": PLUGIN["name"], "success": False, "error": message}, ensure_ascii=False)


def _coerce_args(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    raise ToolError("matterbridge_read expects an object as args.")


def _parse_limit(value: Any | None) -> int:
    if value is None:
        return 10
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolError("limit must be an integer") from exc
    if limit <= 0:
        raise ToolError("limit must be greater than zero")
    return limit


def _parse_stream(value: Any | None) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ToolError("stream must be a boolean")


def _buffer_endpoint(uri: str, timeout: float, limit: int) -> list[Mapping[str, Any]]:
    endpoint = f"{uri.rstrip('/')}/api/messages"
    request = urllib.request.Request(endpoint)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise ToolError(f"matterbridge_read buffer failed ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise ToolError(f"matterbridge_read buffer network error: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ToolError(f"matterbridge_read buffer returned invalid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ToolError("matterbridge_read buffer expected a JSON array")
    return payload[:limit]


def _stream_endpoint(uri: str, timeout: float, limit: int) -> tuple[list[Mapping[str, Any]], bool]:
    endpoint = f"{uri.rstrip('/')}/api/stream"
    request = urllib.request.Request(endpoint)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _read_stream(response, timeout, limit)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise ToolError(f"matterbridge_read stream failed ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise ToolError(f"matterbridge_read stream network error: {exc}") from exc


def _read_stream(response, timeout: float, limit: int) -> tuple[list[Mapping[str, Any]], bool]:
    messages: list[Mapping[str, Any]] = []
    start = time.monotonic()
    truncated = False
    while True:
        if len(messages) >= limit:
            truncated = True
            break
        if timeout and time.monotonic() - start >= timeout:
            truncated = True
            break
        try:
            line = response.readline()
        except socket.timeout as exc:
            raise ToolError(f"matterbridge_read stream timed out: {exc}") from exc
        if not line:
            break
        if isinstance(line, bytes):
            text = line.decode("utf-8", errors="ignore").strip()
        else:
            text = str(line).strip()
        if not text:
            continue
        try:
            event = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolError(f"matterbridge_read stream returned invalid JSON: {exc}") from exc
        if isinstance(event, dict) and event.get("event") == "api_connected":
            continue
        if not isinstance(event, dict):
            raise ToolError("matterbridge_read stream received unexpected value")
        messages.append(event)
    return messages, truncated


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
        limit = _parse_limit(payload.get("limit"))
        stream = _parse_stream(payload.get("stream"))
        timeout = parse_timeout(payload.pop("timeout", None))
        context = get_matterbridge_config()
        uri, gateway = _resolve_matterbridge_config(context, payload, require_gateway=False)
        if stream:
            messages, truncated = _stream_endpoint(uri, timeout, limit)
            data = {"messages": messages, "via": "stream", "truncated": truncated}
        else:
            messages = _buffer_endpoint(uri, timeout, limit)
            data = {"messages": messages, "via": "buffer"}
        data["gateway"] = gateway
        return _success(data)
    except ToolError as exc:
        return _error(str(exc))
