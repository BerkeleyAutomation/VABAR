"""Shared Arena empty-warehouse HDR configuration for benchmark dome lights."""

from __future__ import annotations

from isaaclab import sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_arena.assets.hdr_image_library import EmptyWarehouseHDRRobolab

EMPTY_WAREHOUSE_HDR_REGISTRY_NAME = EmptyWarehouseHDRRobolab.name
# Arena resolves its assets from its staging asset root, which does not host Isaac Sim 6.1's
# copy of this HDR; Isaac Lab's asset root keeps the byte-identical file.
EMPTY_WAREHOUSE_HDR_TEXTURE_FILE = (
    f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/"
    "backgrounds/default/empty_warehouse.hdr"
)
EMPTY_WAREHOUSE_HDR_TEXTURE_FORMAT = EmptyWarehouseHDRRobolab.texture_format
EMPTY_WAREHOUSE_HDR_INTENSITY = 1500.0
EMPTY_WAREHOUSE_HDR_COLOR = (0.75, 0.75, 0.75)


def empty_warehouse_dome_light_cfg() -> sim_utils.DomeLightCfg:
    """Create an independent dome-light config using Arena's empty-warehouse HDR."""

    return sim_utils.DomeLightCfg(
        color=EMPTY_WAREHOUSE_HDR_COLOR,
        intensity=EMPTY_WAREHOUSE_HDR_INTENSITY,
        texture_file=EMPTY_WAREHOUSE_HDR_TEXTURE_FILE,
        texture_format=EMPTY_WAREHOUSE_HDR_TEXTURE_FORMAT,
        visible_in_primary_ray=True,
    )
