"""Owner-local path for the vendored industrial FR3/Robotiq closure."""

from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parents[1] / "vabar__fr3_robotiq_2f85"
ROBOT_USD_PATH = ASSET_ROOT / "franka_fr3_robotiq_2f85.usda"
ROBOT_INSTANCEABLE_USD_PATH = ASSET_ROOT / "franka_fr3_robotiq_2f85_instanceable.usda"
ROBOT_ON_CART_USD_PATH = (
    ASSET_ROOT.parent
    / "vabar__fr3_robotiq_2f85_on_cart"
    / "vabar__fr3_robotiq_2f85_on_cart.usda"
)
ROBOT_ON_CART_INSTANCEABLE_USD_PATH = ROBOT_ON_CART_USD_PATH.with_name(
    "vabar__fr3_robotiq_2f85_on_cart_instanceable.usda"
)


def register_assets() -> None:
    """Expose the owner module through the stable asset-discovery contract."""

    for path in (
        ROBOT_USD_PATH,
        ROBOT_INSTANCEABLE_USD_PATH,
        ROBOT_ON_CART_USD_PATH,
        ROBOT_ON_CART_INSTANCEABLE_USD_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"FR3/Robotiq asset is missing: {path}")
