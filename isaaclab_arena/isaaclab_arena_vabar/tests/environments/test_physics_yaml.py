"""Fixed Newton profiles are resolved and materialized from graph YAML."""

from types import SimpleNamespace

import pytest

from ib_support import ENVIRONMENT_YAMLS, ENVIRONMENTS_ROOT

# Cable routing owns its Newton configuration in the factory, not in YAML.
GRAPH_PHYSICS_YAMLS = tuple(
    path for path in ENVIRONMENT_YAMLS if not path.startswith("cable_routing/")
)


def _materialize(relative_graph: str):
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environment_spec.env_cfg_override import (
        apply_env_cfg_override,
    )

    @configclass
    class SceneCfg:
        num_envs: int = 1
        # Arena's builder forces replicate_physics=True for Newton before applying
        # the graph override.
        replicate_physics: bool = True

    @configclass
    class EnvCfg:
        sim: SimulationCfg = SimulationCfg(physics=NewtonCfg())
        scene: SceneCfg = SceneCfg()
        decimation: int = 1
        num_rerenders_on_reset: int = 0

    graph = ArenaEnvGraphSpec._load_yaml_dict(str(ENVIRONMENTS_ROOT / relative_graph))
    assert graph["default_physics_backend"] == "newton"
    return apply_env_cfg_override(EnvCfg(), graph["env_cfg_override"])


@pytest.mark.parametrize("relative_graph", GRAPH_PHYSICS_YAMLS)
def test_every_graph_yaml_resolves_a_replicated_implicit_newton_profile(
    relative_graph: str,
) -> None:
    """Shared ``external_yaml`` profiles reach every graph-configured scene."""
    cfg = _materialize(relative_graph)

    assert cfg.sim.physics.solver_cfg.integrator == "implicitfast"
    # Newton does not support replicate_physics=False: it parses every env into one
    # world, so num_envs > 1 fails articulation init. Keep Arena's True.
    assert cfg.scene.replicate_physics is True
    # Visual GAP runs need camera observations refreshed after reset.
    assert cfg.num_rerenders_on_reset == 5


@pytest.mark.parametrize(
    ("relative_graph", "nconmax", "num_substeps", "use_mujoco_contacts"),
    [
        *(
            (f"tool_hanging/{family}_{level}.yaml", 4096, 8, True)
            for family in ("wrench", "plier", "scissors", "screwdriver")
            for level in ("easy", "medium")
        ),
        ("usbc_insertion/easy.yaml", 2048, 16, False),
        ("usbc_insertion/medium.yaml", 2048, 16, False),
    ],
)
def test_graph_yaml_materializes_newton_physics(
    relative_graph: str,
    nconmax: int,
    num_substeps: int,
    use_mujoco_contacts: bool,
) -> None:
    cfg = _materialize(relative_graph)
    assert cfg.sim.dt == pytest.approx(1.0 / 60.0)
    assert cfg.sim.use_newton_actuators is True
    assert cfg.sim.physics.num_substeps == num_substeps
    assert cfg.sim.physics.solver_cfg.nconmax == nconmax
    assert cfg.sim.physics.solver_cfg.use_mujoco_contacts is use_mujoco_contacts
    assert cfg.sim.physics.solver_cfg.disable_sensors is True
    assert cfg.num_rerenders_on_reset == 5
    if relative_graph.startswith("tool_hanging/"):
        from isaaclab_arena_vabar.environments.tool_hanging.environment import (
            configure_tool_hanging_physics,
        )

        configure_tool_hanging_physics(cfg)
        assert cfg.sim.physics.solver_cfg.njmax == 8192
        assert cfg.sim.physics.solver_cfg.jacobian == "sparse"
        assert cfg.sim.physics.solver_cfg.enable_multiccd is True
        assert cfg.sim.physics.collision_cfg is None
        assert cfg.sim.physics.default_shape_cfg.gap == pytest.approx(0.0005)
    else:
        from isaaclab_arena_vabar.environments.usbc_insertion.environment import (
            _configure_env_cfg,
        )

        expected_links = 24 if relative_graph.endswith("medium.yaml") else 16
        cfg.events = SimpleNamespace()
        _configure_env_cfg(cfg, cable_links=expected_links)
        assert cfg.decimation == 1
        assert cfg.sim.physics.collision_decimation == 1
        assert cfg.sim.physics.solver_cfg.njmax == 4096
        assert cfg.sim.physics.solver_cfg.integrator == "implicitfast"
        assert cfg.sim.physics.solver_cfg.iterations == 100
        assert cfg.sim.physics.solver_cfg.ls_iterations == 50
        assert cfg.sim.physics.solver_cfg.cone == "elliptic"
        assert cfg.sim.physics.solver_cfg.impratio == pytest.approx(10.0)
        assert cfg.sim.physics.solver_cfg.enable_multiccd is False
        assert cfg.sim.physics.collision_cfg.rigid_contact_max == 8192
        assert cfg.sim.physics.collision_cfg.contact_matching == "disabled"
        assert cfg.sim.physics.default_shape_cfg.gap == pytest.approx(0.0002)
        assert cfg.sim.physics.use_cuda_graph is False
        assert cfg.events.reset_usbc_cables.params == {"expected_links": expected_links}


@pytest.mark.parametrize("relative_graph", GRAPH_PHYSICS_YAMLS)
def test_graph_yaml_caps_convex_collision_iterations_at_mujoco_default(
    relative_graph: str,
) -> None:
    """The YAML must override the 15000 of Arena's dexterous-hand Newton preset.

    AUTOLab runs every suite at MuJoCo's default of 35. At 15000, a resting convex pair
    whose GJK does not converge costs about 50 ms per call, every substep.
    """
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environment_spec.env_cfg_override import (
        apply_env_cfg_override,
    )

    cfg = _materialize(relative_graph)
    # Isaac Lab's own default is already 35; start from Arena's preset value instead.
    cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(ccd_iterations=15000))
    graph = ArenaEnvGraphSpec._load_yaml_dict(str(ENVIRONMENTS_ROOT / relative_graph))
    cfg = apply_env_cfg_override(cfg, graph["env_cfg_override"])

    assert cfg.sim.physics.solver_cfg.ccd_iterations == 35


def test_kinematic_fixtures_do_not_collide_with_each_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isaaclab_arena_vabar.physics import _filter_kinematic_contacts
    from isaaclab_newton.physics import NewtonManager
    from newton import BodyFlags

    kinematic, dynamic = int(BodyFlags.KINEMATIC), int(BodyFlags.DYNAMIC)
    # Bodies: rack, hook (world 0), wrench (dynamic), rack (world 1).
    pairs = []
    builder = SimpleNamespace(
        body_flags=[kinematic, kinematic, dynamic, kinematic],
        body_world=[0, 0, 0, 1],
        body_shapes={-1: [9], 0: [0, 1], 1: [2, 3], 2: [4], 3: [5]},
        add_shape_collision_filter_pair=lambda a, b: pairs.append((a, b)),
    )
    monkeypatch.setattr(NewtonManager, "_builder", builder)

    _filter_kinematic_contacts()

    assert sorted(pairs) == [(0, 2), (0, 3), (1, 2), (1, 3)]


def test_benchmark_physics_installs_the_kinematic_contact_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isaaclab_arena_vabar import physics

    calls = []
    monkeypatch.setattr(
        physics, "register_kinematic_contact_filter", lambda: calls.append(True)
    )
    monkeypatch.setattr(physics, "with_unexposed_fields", lambda config, *_: config)
    physics.install_benchmark_physics(
        SimpleNamespace(solver_cfg=None), {"solver_cfg": {}}
    )
    assert calls == [True]
