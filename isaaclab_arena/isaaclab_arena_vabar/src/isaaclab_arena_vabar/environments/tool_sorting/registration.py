"""Tool sorting environments."""

from isaaclab_arena_vabar.environments.tool_sorting.environment import ToolSortEnvironmentDescriptor


ENVIRONMENTS = tuple(
    ToolSortEnvironmentDescriptor(
        environment_name=f"vabar_tool_sorting__easy_{index}",
        relative_graph_path=f"tool_sorting/easy_{index}.yaml",
        random_yaw_init=True,
        inset_source_bin=True,
        upstream_pad_friction=True,
    )
    for index in (1, 2, 3)
)
