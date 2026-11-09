"""Small compatibility helpers for Arena's public registry API."""

from __future__ import annotations

from typing import Any


def registered_component(registry: Any, name: str) -> Any | None:
    """Return an existing component without coupling to registry internals."""

    if not registry.is_registered(name, ensure_loaded=False):
        return None
    return registry.get_component_by_name(name)
