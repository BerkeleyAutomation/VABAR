# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Arena experiments with this folder's components registered at runtime."""

from __future__ import annotations

import sys
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from isaaclab_arena_gap.arena.registration import register_arena_components

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg


def _with_safe_openusd_thread_limit(
    parse_args: Callable[..., tuple[Any, list[str]]],
) -> Callable[..., tuple[Any, list[str]]]:
    """Keep collider-heavy Newton startup off unsafe concurrent USD parsing."""

    def parse_args_with_limit(*args, **kwargs):
        args_cli, overrides = parse_args(*args, **kwargs)
        if not hasattr(args_cli, "limit_cpu_threads"):
            args_cli.limit_cpu_threads = 1
        return args_cli, overrides

    return parse_args_with_limit


def _build_environment_for_policy(
    cfg: ArenaRunCfg,
    render_mode: str | None,
    *,
    output_dir: str = "",
    rebuild_index: int = 0,
    run_execution: Any | None = None,
):
    """Build one environment after applying policy-owned runtime requirements."""
    if run_execution is None:
        from isaaclab_arena.evaluation import run_execution

    run_cfg = deepcopy(cfg)
    trajectory_recording_enabled = run_cfg.environment_builder.record_trajectories
    run_cfg.environment_builder.recorder_dataset_export_dir_path = (
        output_dir if trajectory_recording_enabled else None
    )
    run_cfg.environment_builder.recorder_dataset_filename = (
        f"dataset_{cfg.name}_rebuild{rebuild_index}"
    )
    arena_builder = run_execution.build_arena_builder_from_run_cfg(run_cfg)
    _, env_cfg, env_kwargs = arena_builder.build_registered()

    from isaaclab_newton.physics import NewtonCfg

    if not isinstance(env_cfg.sim.physics, NewtonCfg):
        raise TypeError(
            f"Run {cfg.name!r} built unsupported physics backend "
            f"{type(env_cfg.sim.physics).__name__}; VABAR on Isaac Lab Arena supports only Newton"
        )
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena_gap.external_policy_termination import (
        configure_external_policy_termination,
        external_policy_termination,
    )

    configure_external_policy_termination(env_cfg.scene.num_envs)
    env_cfg.terminations.external_policy = TerminationTermCfg(
        func=external_policy_termination
    )

    if getattr(cfg.policy, "requires_newton_native_actuators", False):
        env_cfg.sim.use_newton_actuators = True
        env_cfg.decimation = 1
        minimum_njmax = getattr(cfg.policy, "minimum_newton_njmax", None)
        if minimum_njmax is not None:
            solver_cfg = env_cfg.sim.physics.solver_cfg
            if hasattr(solver_cfg, "njmax"):
                solver_cfg.njmax = max(solver_cfg.njmax, minimum_njmax)
        print(
            f"[vabar] Newton-native actuators enabled for run {cfg.name!r}",
            flush=True,
        )

    from isaaclab_arena_gap.newton_camera_rendering import configure_newton_camera_renderers

    camera_count = configure_newton_camera_renderers(env_cfg)
    if camera_count:
        print(
            f"[vabar] Isaac RTX rendering enabled for {camera_count} camera(s)",
            flush=True,
        )

    env = arena_builder.make_registered(
        env_cfg,
        env_kwargs,
        render_mode=render_mode,
    )
    from isaaclab_arena_gap.external_policy_termination import (
        requested_external_policy_terminations,
    )

    env.unwrapped.external_policy_termination_requested = (
        requested_external_policy_terminations
    )
    return env


def initialize_arena_runtime() -> None:
    """Register downstream components and install this repository's policy build contract."""
    register_arena_components()

    from isaaclab_arena_gap.newton_15_tendon_compat import (
        install_newton_15_passive_tendon_compat,
    )

    install_newton_15_passive_tendon_compat()

    from isaaclab_arena.evaluation import run_execution

    run_execution._build_environment_from_cfg = _build_environment_for_policy


#: Isaac Lab's rendering apps enable UJITSO geometry streaming, which loses
#: Fabric-driven dynamic geometry: whole scenes render empty mid-episode.
#: Only the UJITSO master switch silences RTX's streaming warning; Kit reads
#: both settings at startup.
_DISABLE_GEOMETRY_STREAMING = ("--/UJITSO/enabled=false", "--/UJITSO/geometry=false")


def _with_render_kit_args(
    parse_args: Callable[..., tuple[Any, list[str]]],
) -> Callable[..., tuple[Any, list[str]]]:
    """Disable RTX geometry streaming, keeping any caller-supplied Kit arguments."""

    def parse_args_with_kit_args(*args, **kwargs):
        args_cli, overrides = parse_args(*args, **kwargs)
        existing = (getattr(args_cli, "kit_args", None) or "").split()
        missing = [arg for arg in _DISABLE_GEOMETRY_STREAMING if arg not in existing]
        args_cli.kit_args = " ".join([*existing, *missing])
        return args_cli, overrides

    return parse_args_with_kit_args


def main() -> None:
    """Delegate to Arena and register downstream components after AppLauncher starts."""
    from isaaclab_arena.evaluation import experiment_runner

    original_parse_args = experiment_runner.parse_experiment_runner_args
    original_context = experiment_runner.SimulationAppContext

    class RegisteredSimulationAppContext:
        def __init__(self, args_cli):
            self._context = original_context(args_cli)

        def __enter__(self):
            result = self._context.__enter__()
            try:
                initialize_arena_runtime()
            except BaseException:
                self._context.__exit__(*sys.exc_info())
                raise
            return result

        def __exit__(self, exception_type, exception, traceback):
            return self._context.__exit__(exception_type, exception, traceback)

    experiment_runner.parse_experiment_runner_args = _with_render_kit_args(
        _with_safe_openusd_thread_limit(original_parse_args)
    )
    experiment_runner.SimulationAppContext = RegisteredSimulationAppContext
    try:
        experiment_runner.main()
    finally:
        experiment_runner.parse_experiment_runner_args = original_parse_args
        experiment_runner.SimulationAppContext = original_context


if __name__ == "__main__":
    main()
