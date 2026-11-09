"""Tool-hanging environments: AUTOLab Level-1 tasks as easy, Level-2 as medium."""

from pathlib import Path

import yaml

from isaaclab_arena_vabar.environments.tool_hanging.environment import (
    ToolHangingEnvironmentDescriptor,
)

_PLIER_FRICTION = yaml.safe_load(Path(__file__).with_name("plier_easy.yaml").read_text())[
    "fingertip_physics"
]


ENVIRONMENTS = (
    ToolHangingEnvironmentDescriptor(
        environment_name="vabar_tool_hanging__plier_easy",
        relative_graph_path="tool_hanging/plier_easy.yaml",
        work_arm=None,
        random_yaw_init=False,
        left_fingertip_torsional_friction=_PLIER_FRICTION["left_torsional_friction"],
        fingertip_friction_required_shape_tokens=tuple(
            _PLIER_FRICTION["required_shape_tokens"]
        ),
        fingertip_friction_forbidden_shape_tokens=tuple(
            _PLIER_FRICTION["forbidden_shape_tokens"]
        ),
    ),
    ToolHangingEnvironmentDescriptor(
        environment_name="vabar_tool_hanging__scissors_easy",
        relative_graph_path="tool_hanging/scissors_easy.yaml",
        work_arm=None,
        random_yaw_init=False,
        finger_contact_priority=True,
    ),
    ToolHangingEnvironmentDescriptor(
        environment_name="vabar_tool_hanging__screwdriver_easy",
        relative_graph_path="tool_hanging/screwdriver_easy.yaml",
        work_arm=None,
        random_yaw_init=False,
    ),
    ToolHangingEnvironmentDescriptor(
        environment_name="vabar_tool_hanging__wrench_easy",
        relative_graph_path="tool_hanging/wrench_easy.yaml",
        work_arm=None,
        random_yaw_init=False,
    ),
)

ENVIRONMENTS += tuple(
    ToolHangingEnvironmentDescriptor(
        environment_name=f"vabar_tool_hanging__{family}_medium",
        relative_graph_path=f"tool_hanging/{family}_medium.yaml",
        finger_contact_priority=family == "scissors",
    )
    for family in ("plier", "scissors", "screwdriver", "wrench")
)
