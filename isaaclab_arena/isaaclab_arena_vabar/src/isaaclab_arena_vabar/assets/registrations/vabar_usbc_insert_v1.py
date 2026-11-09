from isaaclab_arena_vabar.assets.registry import (
    ConvertedAssetSpec,
    register_converted_assets,
)

ASSETS = (
    ConvertedAssetSpec(
        registry_name="vabar__bulkhead",
        role="object",
        relative_usd_path="vabar__bulkhead/vabar__bulkhead.usda",
        object_min_z=None,
    ),
    ConvertedAssetSpec(
        registry_name="vabar__plug",
        role="object",
        relative_usd_path="vabar__plug/vabar__plug.usda",
        object_min_z=None,
    ),
)


def register_assets() -> None:
    register_converted_assets(ASSETS, module_file=__file__)
