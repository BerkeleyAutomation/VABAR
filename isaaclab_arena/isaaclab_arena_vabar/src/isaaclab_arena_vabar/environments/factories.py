"""Find the registered environment factory that builds a scene YAML."""

from __future__ import annotations


def factory_name_for(relative_yaml: str, registry) -> str | None:
    """Return the registered factory that builds one package-relative YAML."""
    for registered_name in registry.get_all_keys():
        factory_type = registry.get_component_by_name(registered_name)
        descriptor = getattr(factory_type, "descriptor", None)
        relative_graph_path = getattr(
            descriptor, "relative_graph_path", getattr(factory_type, "relative_graph_path", None)
        )
        if relative_graph_path == relative_yaml:
            return registered_name
    return None
