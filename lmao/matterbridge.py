from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MatterbridgeConfig:
    uri: Optional[str] = None
    gateway: Optional[str] = None


_CURRENT_CONFIG = MatterbridgeConfig()


def _normalize_option(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized else None


def configure_matterbridge(uri: Optional[str], gateway: Optional[str]) -> None:
    """
    Set the resolved Matterbridge defaults discovered via CLI flags or configuration files.
    """
    global _CURRENT_CONFIG  # pylint: disable=global-statement
    _CURRENT_CONFIG = MatterbridgeConfig(
        uri=_normalize_option(uri),
        gateway=_normalize_option(gateway),
    )


def get_matterbridge_config() -> MatterbridgeConfig:
    return _CURRENT_CONFIG
