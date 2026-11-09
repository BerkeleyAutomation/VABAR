"""USB-C insertion environments."""

from dataclasses import replace

from isaaclab_arena_vabar.environments.usbc_insertion.environment import (
    UsbcInsertEnvironmentDescriptor,
)

ENVIRONMENTS = (
    UsbcInsertEnvironmentDescriptor(
        environment_name="vabar_usbc_insertion__easy",
        relative_graph_path="usbc_insertion/easy.yaml",
        cable_spec={
            "scale": 2.0,
            "plug": {
                "object": "plug",
                "links": 16,
                "length": 0.12,
                "radius": 0.0016,
                "density": 450.0,
                "rear_z": -0.034,
            },
        },
    ),
)

ENVIRONMENTS += (
    replace(
        ENVIRONMENTS[0],
        environment_name="vabar_usbc_insertion__medium",
        relative_graph_path="usbc_insertion/medium.yaml",
        cable_spec={
            **ENVIRONMENTS[0].cable_spec,
            "bulkhead": {
                "object": "bulkhead",
                "links": 8,
                "length": 0.05,
                "radius": 0.0016,
                "density": 450.0,
            },
        },
    ),
)
