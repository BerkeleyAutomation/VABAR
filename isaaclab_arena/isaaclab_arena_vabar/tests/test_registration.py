"""Public registration seam for source consumption."""

from __future__ import annotations

import pytest

from ib_support import ENVIRONMENT_YAMLS, ENVIRONMENTS_ROOT, build_registered_environment

BENCHMARK_PREFIXES = (
    "vabar_cable_routing__",
    "vabar_gear_insertion__",
    "vabar_syringe_disposal__",
    "vabar_tool_hanging__",
    "vabar_tool_sorting__",
    "vabar_usbc_insertion__",
)


def test_public_registration_is_idempotent_and_exposes_core_components() -> None:
    from isaaclab_arena_vabar.environments.factories import factory_name_for
    from isaaclab_arena_vabar.registration import register_components
    from isaaclab_arena.assets.registries import (
        AssetRegistry,
        EnvironmentRegistry,
        TaskRegistry,
    )

    register_components()
    register_components()

    # Any entry point that registers the benchmark spreads placement seeds, not only GaP's.
    from isaaclab_arena.relations.placement_candidate_generator import PlacementCandidateGenerator

    assert getattr(PlacementCandidateGenerator.generate_candidates, "_isaac_cap_placement_seed_compat", False)

    assets = set(AssetRegistry().get_all_keys())
    registry = EnvironmentRegistry()
    environments = set(registry.get_all_keys())
    tasks = set(TaskRegistry().get_all_keys())
    assert {
        "vabar__fr3_robotiq_2f85",
        "vabar__bimanual_yam",
        "vabar__syringe_sort_bin",
    } <= assets
    syringe_bin = AssetRegistry().get_asset_by_name(
        "vabar__syringe_sort_bin"
    )()
    assert syringe_bin.usd_path.endswith(
        "vabar__tool_sort_bin/bin2_syringe.usda"
    )
    assert {
        "GearMeshTask",
        "ObjectsCenteredInRegionsTask",
        "IndustrialToolHangingTask",
        "UsbcInsertionTask",
    } <= tasks
    assert {
        "vabar__gear_mesh_16t",
        "vabar__gear_mesh_20t",
        "vabar__gear_mesh_24t",
        "vabar__gear_mesh_board_16",
        "vabar__gear_mesh_board_20",
        "vabar__gear_mesh_board_24",
        "vabar__gear_mesh_mat",
    } <= assets
    # Names consumed by experiment configs outside this package.
    assert {
        "vabar_cable_routing__easy",
        "vabar_cable_routing__medium",
        "vabar_gear_insertion__easy",
        "vabar_gear_insertion__easy_pair",
        "vabar_gear_insertion__medium_train",
        "vabar_usbc_insertion__easy",
        "vabar_usbc_insertion__medium",
        "vabar_tool_hanging__plier_easy",
        "vabar_tool_hanging__screwdriver_easy",
        "vabar_tool_hanging__scissors_easy",
        "vabar_tool_sorting__easy_1",
        "vabar_tool_sorting__easy_2",
        "vabar_tool_sorting__easy_3",
        "vabar_syringe_disposal__single",
        "vabar_syringe_disposal__cluttered",
    } <= environments
    benchmark_environments = {
        name for name in environments if name.startswith(BENCHMARK_PREFIXES)
    }
    # Every scene YAML has exactly one registered factory and vice versa.
    yaml_factories = [factory_name_for(path, registry) for path in ENVIRONMENT_YAMLS]
    assert None not in yaml_factories
    assert sorted(yaml_factories) == sorted(benchmark_environments)
    assert all(
        name.startswith("vabar_")
        and (name.count("__") == 1 or name.startswith("vabar_tool_sorting_easy_"))
        for name in benchmark_environments
    )
    assert all(
        "newton" not in name and "yam" not in name for name in benchmark_environments
    )
    for name in benchmark_environments:
        factory_type = registry.get_component_by_name(name)
        cfg_type = registry.get_environment_cfg_type(factory_type)
        cfg = cfg_type()
        assert cfg.use_tiled_cameras is False, name
        assert cfg.use_instanceable_meshes is False, name


def test_wrench_easy_build_uses_canonical_yam_and_native_termination(
    stub_scene_module,
) -> None:
    from isaaclab_arena_vabar.assets.registrations.industrial_i2rt_yam import (
        ROBOT_USD_PATH,
    )
    from isaaclab_arena_vabar.embodiments.bimanual_yam import (
        IndustrialBimanualYamEmbodiment,
    )
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    arena_env = build_registered_environment(
        "vabar_tool_hanging__wrench_easy", enable_cameras=True
    )

    assert isinstance(arena_env.embodiment, IndustrialBimanualYamEmbodiment)
    assert arena_env.embodiment.scene_config.left_robot.spawn.usd_path == str(
        ROBOT_USD_PATH
    )
    assert len(arena_env.embodiment.camera_config.camera_names()) == 5
    assert isinstance(arena_env.task.get_termination_cfg(), TaskTerminationCfg)
    # AUTOLab's wrench goal reads the hook's load, so the wrench carries a hook contact sensor.
    assert arena_env.task.contact_sensor_names == (
        "contact_sensor_ring_spanner_19_wrench_hook_2",
    )


@pytest.mark.parametrize("graph", [
    "syringe_disposal/single.yaml",
    "gear_insertion/easy.yaml",
])
def test_fr3_graph_selects_local_robot_and_continuous_gripper(graph):
    from isaaclab_arena_vabar.assets.registrations.industrial_fr3_robotiq_2f85 import (
        ROBOT_ON_CART_USD_PATH,
    )
    from isaaclab_arena_vabar.registration import register_components
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    register_components()
    spec = ArenaEnvGraphSpec.from_yaml(ENVIRONMENTS_ROOT / graph)
    embodiment = AssetRegistry().get_asset_by_name(spec.embodiment.registry_name)()

    assert embodiment.scene_config.robot.spawn.usd_path == str(ROBOT_ON_CART_USD_PATH)
    gripper = embodiment.action_config.gripper_action
    assert isinstance(gripper, JointPositionActionCfg)
    assert gripper.scale == pytest.approx(0.8)
    assert gripper.use_default_offset is False
