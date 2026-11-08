"""Syringe disposal environments; they use the tool-sorting environment class."""

from isaaclab_arena_vabar.environments.tool_sorting.environment import ToolSortEnvironmentDescriptor

# AUTOLab trials each variant is laid out from; the others use the solver's own placement.
TRAY_LAYOUTS = {"both": "pair_layouts.json", "cluttered": "clutter_layouts.json"}

ENVIRONMENTS = tuple(
    ToolSortEnvironmentDescriptor(
        environment_name=f"vabar_syringe_disposal__{name}",
        relative_graph_path=f"syringe_disposal/{name}.yaml",
        random_yaw_init=False,
        autolab_tray_layouts=TRAY_LAYOUTS.get(name),
    )
    for name in ("single", "both", "designated", "cluttered")
)
