"""Build, reset and step one shipped Arena Experiment in a fresh Isaac process."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
CONFIG_ROOT = ROOT / "experiment_configs"


def _load_run(config: Path) -> dict:
    resolved = config.resolve()
    if not resolved.is_relative_to(CONFIG_ROOT.resolve()) or not resolved.is_file():
        raise ValueError(f"unknown Arena Experiment config: {config}")
    document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    runs = document.get("runs")
    if not isinstance(runs, dict) or len(runs) != 1:
        raise ValueError(f"expected exactly one Run in {config}")
    name, run = next(iter(runs.items()))
    environment = dict(run["environment"])
    environment_type = environment.pop("type")
    return {
        "name": name,
        "arena_env_args": {
            "environment": environment_type,
            **environment,
            **dict(run.get("environment_builder") or {}),
        }
    }


def _build_environment(job: dict):
    from isaaclab_arena_vabar.registration import register_components
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    register_components()
    args = dict(job["arena_env_args"])
    environment = str(args.pop("environment"))
    enable_cameras = bool(args.pop("enable_cameras", False))
    if environment.endswith((".yaml", ".yml")):
        path = Path(environment)
        if not path.is_absolute():
            path = WORKSPACE_ROOT / path
        return ArenaEnvGraphSpec.from_yaml(str(path)).to_arena_env(
            enable_cameras=enable_cameras
        )

    registry = EnvironmentRegistry()
    factory_type = registry.get_component_by_name(environment)
    cfg_type = registry.get_environment_cfg_type(factory_type)
    accepted = {field.name for field in fields(cfg_type)}
    cfg_args = {key: value for key, value in args.items() if key in accepted}
    if "enable_cameras" in accepted:
        cfg_args["enable_cameras"] = enable_cameras
    return factory_type().build(cfg_type(**cfg_args))


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


def _run(config: Path, steps: int) -> None:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    run = _load_run(config)
    env_args = run["arena_env_args"]
    builder_fields = {field.name for field in fields(ArenaEnvBuilderCfg)}
    builder_args = {
        key: value
        for key, value in env_args.items()
        if key in builder_fields and key != "num_envs"
    }
    builder = ArenaEnvBuilder(
        _build_environment(run), ArenaEnvBuilderCfg(num_envs=1, **builder_args)
    )
    _name, env_cfg, env_kwargs = builder.build_registered()
    env = None
    try:
        env = builder.make_registered(env_cfg, env_kwargs)
        observation, _ = env.reset()
        unwrapped = env.unwrapped
        action = _hold_action(unwrapped)
        _assert_finite((observation, action))
        for _ in range(steps):
            observation, reward, terminated, truncated, _ = env.step(action)
            _assert_finite((observation, reward))
            assert terminated.shape == truncated.shape == (1,)
        print(
            "ARENA_EXPERIMENT_SMOKE "
            + json.dumps(
                {
                    "run": run["name"],
                    "steps": steps,
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
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.limit_cpu_threads = 1
    with SimulationAppContext(args):
        from isaaclab_arena_gap.newton_15_tendon_compat import (
            install_newton_15_passive_tendon_compat,
        )

        install_newton_15_passive_tendon_compat()
        _run(args.config, args.steps)


if __name__ == "__main__":
    main()
