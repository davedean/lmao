from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from lmao.matterbridge import MatterbridgeConfig

MATTERBRIDGE_DEFAULT_TIMEOUT = 10.0


class ToolError(Exception):
    pass


def _normalize_option(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _resolve_matterbridge_config(
    context: MatterbridgeConfig,
    args: Optional[Mapping[str, Any]],
    *,
    require_gateway: bool,
) -> tuple[str, Optional[str]]:
    environment = os.environ
    uri = context.uri or _normalize_option(environment.get("MATTERBRIDGE_URI"))
    if uri is None:
        raise ToolError(
            "Matterbridge URI is not configured; set --matterbridge-uri, [matterbridge].uri, or MATTERBRIDGE_URI."
        )

    overrides: Mapping[str, Any] = args or {}
    gateway_override = overrides.get("gateway")
    gateway = (
        _normalize_option(str(gateway_override))
        if gateway_override is not None
        else None
    )
    if gateway is None:
        gateway = context.gateway or _normalize_option(environment.get("MATTERBRIDGE_GATEWAY"))
    if require_gateway and gateway is None:
        raise ToolError(
            "Matterbridge gateway is not configured; set --matterbridge-gateway, [matterbridge].gateway, or MATTERBRIDGE_GATEWAY."
        )
    return uri, gateway


def parse_timeout(value: Any | None) -> float:
    if value is None:
        return MATTERBRIDGE_DEFAULT_TIMEOUT
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolError("timeout must be a number") from exc
    if timeout <= 0:
        raise ToolError("timeout must be greater than zero")
    return timeout
