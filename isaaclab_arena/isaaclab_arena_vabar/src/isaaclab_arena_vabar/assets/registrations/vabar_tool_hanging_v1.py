from isaaclab_arena_vabar.assets.registry import (
    ConvertedAssetSpec,
    register_converted_assets,
)

ASSETS = (
    ConvertedAssetSpec(
        registry_name="vabar__hanging_hammer_1",
        role="object",
        relative_usd_path="vabar__hanging_hammer_1/vabar__hanging_hammer_1.usda",
        object_min_z=None,
    ),
    ConvertedAssetSpec(
        registry_name="vabar__hanging_hammer_2",
        role="object",
        relative_usd_path="vabar__hanging_hammer_2/vabar__hanging_hammer_2.usda",
        object_min_z=None,
    ),
    ConvertedAssetSpec(
        registry_name="vabar__plier_combination",
        role="object",
        relative_usd_path="vabar__plier_combination/vabar__plier_combination.usda",
        object_min_z=None,
    ),
    ConvertedAssetSpec(
        registry_name="vabar__plier_cutter",
        role="object",
        relative_usd_path="vabar__plier_cutter/vabar__plier_cutter.usda",
        object_min_z=None,
    ),
    ConvertedAssetSpec(
        registry_name="vabar__scissor_pair_1",
        role="object",
        relative_usd_path="vabar__scissor_pair_1/vabar__scissor_pair_1.usda",
        object_min_z=None,
    ),
    ConvertedAssetSpec(
        registry_name="vabar__scissor_pair_2",
        role="object",
        relative_usd_path="vabar__scissor_pair_2/vabar__scissor_pair_2.usda",
        object_min_z=None,
    ),
    ConvertedAssetSpec(
        registry_name="vabar__screwdriver_grip",
        role="object",
        relative_usd_path="vabar__screwdriver_grip/vabar__screwdriver_grip.usda",
        object_min_z=None,
    ),
    ConvertedAssetSpec(
        registry_name="vabar__screwdriver_slim",
        role="object",
        relative_usd_path="vabar__screwdriver_slim/vabar__screwdriver_slim.usda",
        object_min_z=None,
    ),
    ConvertedAssetSpec(
        registry_name="vabar__wrench_ring_17",
        role="object",
        relative_usd_path="vabar__wrench_ring_17/vabar__wrench_ring_17.usda",
        object_min_z=None,
    ),
    ConvertedAssetSpec(
        registry_name="vabar__wrench_ring_19",
        role="object",
        relative_usd_path="vabar__wrench_ring_19/vabar__wrench_ring_19.usda",
        object_min_z=None,
    ),
)


def register_assets() -> None:
    register_converted_assets(ASSETS, module_file=__file__)
