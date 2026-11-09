"""Build, reset, and briefly step one benchmark YAML in a fresh Isaac process."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab_arena_vabar.environments.factories import factory_name_for

ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENT_ROOT = ROOT / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments"


def _with_safe_openusd_thread_limit(args):
    """Configure the app for the requested camera and USD smoke coverage."""
    args.limit_cpu_threads = 1
    args.enable_cameras = getattr(args, "with_cameras", False)
    return args


def _factory_name_for(relative_text: str, registry) -> str:
    factory_name = factory_name_for(relative_text, registry)
    if factory_name is None:
        raise RuntimeError(
            f"factory-backed environment is not registered: {relative_text}"
        )
    return factory_name


def _build_environment(
    relative_yaml: Path,
    *,
    enable_cameras: bool = False,
    use_tiled_cameras: bool = False,
    use_instanceable_meshes: bool = False,
    replicate_physics: bool | None = None,
):
    from isaaclab_arena_vabar.registration import register_components
    from isaaclab_arena.assets.registries import EnvironmentRegistry

    register_components()
    relative_text = relative_yaml.as_posix()
    registry = EnvironmentRegistry()
    factory_name = _factory_name_for(relative_text, registry)
    factory_type = registry.get_component_by_name(factory_name)
    cfg_type = registry.get_environment_cfg_type(factory_type)
    cfg_kwargs = {"enable_cameras": enable_cameras}
    if hasattr(cfg_type, "use_tiled_cameras"):
        cfg_kwargs["use_tiled_cameras"] = use_tiled_cameras
    if hasattr(cfg_type, "use_instanceable_meshes"):
        cfg_kwargs["use_instanceable_meshes"] = use_instanceable_meshes
    if hasattr(cfg_type, "replicate_physics"):
        cfg_kwargs["replicate_physics"] = replicate_physics
    return factory_type().build(cfg_type(**cfg_kwargs))


def _hold_action(env):
    import torch
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
    from isaaclab.utils.array import convert_to_torch

    manager = env.action_manager
    action = torch.zeros((env.num_envs, manager.total_action_dim), device=env.device)
    offset = 0
    for name, width in zip(manager.active_terms, manager.action_term_dim, strict=True):
        term = manager.get_term(name)
        if isinstance(term, JointPositionAction):
            positions = term._asset.data.joint_pos
            positions = convert_to_torch(
                positions.torch if hasattr(positions, "torch") else positions,
                device=env.device,
            )
            joint_ids = term._joint_ids
            if not isinstance(joint_ids, slice):
                joint_ids = convert_to_torch(
                    joint_ids, dtype=torch.long, device=env.device
                )
            action[:, offset : offset + width] = positions[:, joint_ids]
        offset += width
    return action


def _assert_finite(value, path: str = "observation") -> None:
    import torch

    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, torch.Tensor):
        valid = torch.isfinite(value)
        name = path.rsplit(".", 1)[-1]
        if name == "distance_to_image_plane" or name.endswith(
            "_distance_to_image_plane"
        ):
            valid |= torch.isposinf(value)
        assert bool(valid.all()), f"non-finite tensor at {path}"
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")
    elif isinstance(value, float):
        assert math.isfinite(value), f"non-finite float at {path}"


def _instanceable_prim_counts() -> tuple[int, int]:
    import isaaclab.sim as sim_utils
    from pxr import Usd

    stage = sim_utils.get_current_stage()
    roots = (
        "/World/envs/env_0/Robot",
        "/World/envs/env_0/LeftRobot",
        "/World/envs/env_0/RightRobot",
    )
    env_root = stage.GetPrimAtPath("/World/envs/env_0")
    total = sum(prim.IsInstanceable() for prim in Usd.PrimRange(env_root))
    robot = sum(
        prim.IsInstanceable()
        for path in roots
        if (robot_root := stage.GetPrimAtPath(path)).IsValid()
        for prim in Usd.PrimRange(robot_root)
    )
    return total, robot


def _run(
    relative_yaml: Path,
    steps: int,
    *,
    num_envs: int = 1,
    enable_cameras: bool = False,
    use_tiled_cameras: bool = False,
    use_instanceable_meshes: bool = False,
    replicate_physics: bool | None = None,
) -> None:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    relative_text = relative_yaml.as_posix()
    arena_env = _build_environment(
        relative_yaml,
        enable_cameras=enable_cameras,
        use_tiled_cameras=use_tiled_cameras,
        use_instanceable_meshes=use_instanceable_meshes,
        replicate_physics=replicate_physics,
    )
    builder_kwargs = {"num_envs": num_envs, "placement_seed": 0}
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(**builder_kwargs))
    _name, env_cfg, env_kwargs = builder.build_registered()
    env = None
    try:
        env = builder.make_registered(env_cfg, env_kwargs)
        observation, _ = env.reset()
        unwrapped = env.unwrapped
        action = _hold_action(unwrapped)
        instanceable_prims, instanceable_robot_meshes = _instanceable_prim_counts()
        if use_instanceable_meshes:
            assert instanceable_prims > 0
        assert action.shape == (num_envs, unwrapped.action_manager.total_action_dim)
        _assert_finite(observation)
        for _ in range(steps):
            observation, reward, terminated, truncated, _ = env.step(action)
            _assert_finite((observation, reward))
            assert terminated.shape == truncated.shape == (num_envs,)
        print(
            "INDUSTRIAL_ENVIRONMENT_SMOKE "
            + json.dumps(
                {
                    "environment": relative_text,
                    "steps": steps,
                    "num_envs": num_envs,
                    "enable_cameras": enable_cameras,
                    "use_tiled_cameras": use_tiled_cameras,
                    "use_instanceable_meshes": use_instanceable_meshes,
                    "instanceable_prims": instanceable_prims,
                    "instanceable_robot_meshes": instanceable_robot_meshes,
                    "instanceable_scene_prims": (
                        instanceable_prims - instanceable_robot_meshes
                    ),
                    "replicate_physics": replicate_physics,
                    "action_dim": unwrapped.action_manager.total_action_dim,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if env is not None:
            env.close()


def main() -> None:
    from isaaclab.app import AppLauncher
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--with-cameras", action="store_true")
    parser.add_argument("--use-tiled-cameras", action="store_true")
    parser.add_argument("--use-instanceable-meshes", action="store_true")
    parser.add_argument("--replicate-physics", action="store_true", default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = _with_safe_openusd_thread_limit(parser.parse_args())
    relative_yaml = Path(args.environment)
    if relative_yaml.is_absolute() or ".." in relative_yaml.parts:
        raise ValueError(
            "environment must be relative to the benchmark environment root"
        )
    if not (ENVIRONMENT_ROOT / relative_yaml).is_file():
        raise ValueError(f"unknown environment YAML: {relative_yaml}")
    with SimulationAppContext(args):
        from isaaclab_arena_gap.newton_15_tendon_compat import (
            install_newton_15_passive_tendon_compat,
        )

        install_newton_15_passive_tendon_compat()
        _run(
            relative_yaml,
            args.steps,
            num_envs=args.num_envs,
            enable_cameras=args.with_cameras,
            use_tiled_cameras=args.use_tiled_cameras,
            use_instanceable_meshes=args.use_instanceable_meshes,
            replicate_physics=args.replicate_physics,
        )


if __name__ == "__main__":
    main()
