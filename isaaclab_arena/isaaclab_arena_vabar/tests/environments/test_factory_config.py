"""Environment factories apply their physics and placement configuration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_gear_mesh_factory_applies_unexposed_contact_controls() -> None:
    from isaaclab_arena_vabar.environments.gear_insertion.environment import (
        _configure_gear_mesh_physics,
    )
    from isaaclab_newton.physics import NewtonCfg

    cfg = SimpleNamespace(sim=SimpleNamespace(physics=NewtonCfg()))

    _configure_gear_mesh_physics(cfg)

    physics = cfg.sim.physics
    assert physics.solver_cfg.enable_multiccd is True
    assert physics.collision_cfg.reduce_contacts is False
    assert physics.collision_cfg.rigid_contact_max == 32768
    assert physics.collision_cfg.contact_matching == "latest"


def test_gear_mesh_factory_uses_strict_source_bounded_layout_pool() -> None:
    from isaaclab_arena_vabar.environments.gear_insertion.environment import (
        _configure_gear_mesh_placement,
    )

    arena_env = SimpleNamespace(
        placer_params=SimpleNamespace(
            min_unique_layouts_per_env=5,
            allow_best_loss_fallbacks=True,
            solver_params=SimpleNamespace(clearance_m=0.01),
        )
    )
    assert _configure_gear_mesh_placement(arena_env) is arena_env

    assert arena_env.placer_params.min_unique_layouts_per_env == 5
    assert arena_env.placer_params.allow_best_loss_fallbacks is False
    assert arena_env.placer_params.solver_params.clearance_m == 0.0


def test_gear_mesh_factory_attaches_enabled_coupled_family_variation(
    monkeypatch,
) -> None:
    from isaaclab_arena_vabar.environments.gear_insertion import (
        environment as environment_module,
    )
    from isaaclab_arena_vabar.environments.gear_insertion.environment import (
        GearInsertionEasyNewtonEnvironment,
        GearInsertionEasyNewtonEnvironmentCfg,
        GearMeshNewtonEnvironment,
    )
    from isaaclab_arena_vabar.tasks.gear_insertion.variation import GearFamilyVariation

    class _VariationHost(SimpleNamespace):
        def add_variation(self, variation):
            self.variation = variation

        def get_variation(self, name):
            assert name == self.variation.name
            return self.variation

    board = _VariationHost()
    gear = _VariationHost()
    task = SimpleNamespace()
    arena_env = SimpleNamespace(
        scene=SimpleNamespace(assets={"board": board, "gear_a": gear}),
        task=task,
    )
    monkeypatch.setattr(
        GearMeshNewtonEnvironment,
        "build",
        lambda self, cfg: arena_env,
    )
    monkeypatch.setattr(
        environment_module, "_configure_gear_mesh_placement", lambda env: env
    )
    arena_env = GearInsertionEasyNewtonEnvironment().build(
        GearInsertionEasyNewtonEnvironmentCfg()
    )
    variation = arena_env.scene.assets["board"].get_variation("gear_family")

    assert isinstance(variation, GearFamilyVariation)
    assert variation.enabled
    assert variation.cfg.tooth_counts == [16, 20, 24]
    assert variation._board is arena_env.scene.assets["board"]
    assert variation._gear is arena_env.scene.assets["gear_a"]
    assert variation._task is arena_env.task
    # The tooth count is drawn from the run's placement seed.
    arena_env.placer_params = SimpleNamespace(placement_seed=42)
    assert variation._seed_source() == 42


def test_gear_layout_factory_requires_support_and_places_the_selected_row(
    monkeypatch,
) -> None:
    from isaaclab_arena_vabar.environments.gear_insertion import (
        environment as environment_module,
    )
    from isaaclab_arena_vabar.environments.gear_insertion.environment import (
        GearMeshNewtonEnvironment,
        GearMeshTrainNewtonEnvironment,
        GearMeshTrainNewtonEnvironmentCfg,
    )
    from isaaclab_arena_vabar.tasks.gear_insertion import parallel_layouts

    class _VariationHost(SimpleNamespace):
        def add_variation(self, variation):
            self.variation = variation

    assets = {name: _VariationHost() for name in ("board", "gear_a", "gear_b", "gear_c")}
    arena_env = SimpleNamespace(
        scene=SimpleNamespace(assets=assets),
        task=SimpleNamespace(),
        placer_params=SimpleNamespace(),
        env_cfg_callback=lambda env_cfg: env_cfg,
    )
    monkeypatch.setattr(GearMeshNewtonEnvironment, "build", lambda self, cfg: arena_env)
    monkeypatch.setattr(environment_module, "_configure_gear_mesh_placement", lambda env: env)
    calls = []
    monkeypatch.setattr(
        parallel_layouts,
        "apply_selected_layout",
        lambda env_cfg, **kwargs: calls.append(kwargs) or env_cfg,
    )

    arena_env = GearMeshTrainNewtonEnvironment().build(GearMeshTrainNewtonEnvironmentCfg())
    # A candidate whose gears sink into the mat must be rejected.
    assert arena_env.placer_params.required_checks == {"on_relation"}
    # The layout is drawn from the run's placement seed.
    arena_env.placer_params.placement_seed = 43
    assert assets["board"].variation._seed_source() == 43

    # Arena solves placement before build-time variations run; the selected row
    # must still reach the solved pool.
    assets["board"].variation.selected_variant = "gearmesh-medium-train-099"
    arena_env.env_cfg_callback(SimpleNamespace())
    assert calls == [
        {
            "layout_name": "gearmesh-medium-train-099",
            "gear_names": ("gear_a", "gear_b", "gear_c"),
        }
    ]

def test_tool_sort_builder_can_opt_into_replicated_physics() -> None:
    from isaaclab_arena_vabar.environments.tool_sorting.environment import (
        finalize_tool_sort_env_cfg,
    )
    from isaaclab_newton.physics import NewtonCfg

    cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=NewtonCfg()),
        scene=SimpleNamespace(num_envs=20, replicate_physics=False),
    )

    finalize_tool_sort_env_cfg(cfg, replicate_physics=True)

    assert cfg.scene.num_envs == 20
    assert cfg.scene.replicate_physics is True
    assert cfg.sim.physics.solver_cfg.enable_multiccd is True


def test_tool_hanging_builder_owns_vector_width_and_can_replicate() -> None:
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab_arena_vabar.environments.tool_hanging.environment import (
        configure_tool_hanging_physics,
    )

    cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=NewtonCfg()),
        scene=SimpleNamespace(num_envs=20, replicate_physics=False),
        decimation=None,
    )

    configure_tool_hanging_physics(cfg, replicate_physics=True)

    assert cfg.scene.num_envs == 20
    assert cfg.scene.replicate_physics is True




def test_usbc_placement_disables_best_loss_fallbacks() -> None:
    from isaaclab_arena_vabar.environments.usbc_insertion.environment import (
        _configure_usbc_placement,
    )

    arena_env = SimpleNamespace(
        placer_params=SimpleNamespace(
            allow_best_loss_fallbacks=True,
            required_checks=None,
            solver_params=SimpleNamespace(clearance_m=0.01, lr=0.01),
        )
    )
    assert _configure_usbc_placement(arena_env) is arena_env
    assert arena_env.placer_params.allow_best_loss_fallbacks is False
    assert arena_env.placer_params.required_checks == {"on_relation"}
    assert arena_env.placer_params.solver_params.clearance_m == 0.0
    assert arena_env.placer_params.solver_params.lr == pytest.approx(0.001)


def test_tray_occupants_are_placed_on_the_cavity_floor_not_the_rim() -> None:
    from isaaclab_arena.relations.relations import On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    from isaaclab_arena_vabar.environments.tool_sorting.environment import (
        INSTRUMENT_TRAY_RIM_TO_FLOOR_M,
        seat_in_tray_cavity,
    )

    def asset(*relations):
        box = AxisAlignedBoundingBox((-0.01, -0.07, -0.0068), (0.01, 0.07, 0.0068))
        return SimpleNamespace(relations=list(relations), bounding_box=box, get_bounding_box=lambda: box)

    tray, table = asset(), asset()
    syringe, bin_ = asset(On(tray)), asset(On(table))
    arena_env = SimpleNamespace(
        scene=SimpleNamespace(
            assets={"instrument_tray": tray, "table": table, "syringe_0": syringe, "bin": bin_, "light": object()}
        )
    )
    seat_in_tray_cavity(arena_env)

    assert INSTRUMENT_TRAY_RIM_TO_FLOOR_M == pytest.approx(0.009)
    # Raised box, so on(tray) at the rim stands the syringe 9 mm lower; footprint unchanged.
    assert syringe.bounding_box.min_point[0].tolist() == pytest.approx([-0.01, -0.07, -0.0068 + 0.009])
    assert syringe.bounding_box.max_point[0].tolist() == pytest.approx([0.01, 0.07, 0.0068 + 0.009])
    for untouched in (tray, table, bin_):
        assert untouched.bounding_box.min_point[0].tolist() == pytest.approx([-0.01, -0.07, -0.0068])
