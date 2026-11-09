"""Owner-local I2RT YAM robot asset path."""

from __future__ import annotations

from pathlib import Path

ROBOT_ASSET_NAME = "vabar__i2rt_yam"
ROBOT_USD_PATH = (
    Path(__file__).resolve().parents[1] / ROBOT_ASSET_NAME / "i2rt_yam_default.usda"
)
ROBOT_INSTANCEABLE_USD_PATH = ROBOT_USD_PATH.with_name("i2rt_yam_instanceable.usda")


def register_assets() -> None:
    """Validate the robot asset consumed by the YAM embodiment."""

    for path in (ROBOT_USD_PATH, ROBOT_INSTANCEABLE_USD_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"robot asset {ROBOT_ASSET_NAME!r} is missing: {path}"
            )
