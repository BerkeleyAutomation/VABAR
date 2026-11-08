# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open a VABAR on Isaac Lab Arena scene in the viewer with no policy attached.

``isaaclab_arena_gap.arena_runner`` always runs a benchmark: it attaches a policy, spawns
a GaP process for managed jobs, records episodes and stops at an episode budget.
None of that helps when the question is "what is this scene actually doing" --
inspecting collision geometry, checking a grasp pose by eye, or watching contact
behaviour settle.

This builds the same scene through Arena's own builder, steps it with zero
actions and holds it open until interrupted. Terminations are cleared so the
environment is never reset out from under you mid-inspection.

Accepts a scene YAML directly, or an Arena job JSON whose
``arena_env_args.environment`` points at one, so the same argument that selects a
job also selects what to look at.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

import yaml

from isaaclab_arena_gap.paths import WORKSPACE_ROOT

@dataclass(frozen=True)
class SceneRequest:
    scene: Path
    env_args: dict
    variations: dict
    job_name: str | None = None


def _scene_declares_cameras(path: Path) -> bool:
    """Return whether the scene explicitly instantiates embodiment cameras."""
    document = yaml.safe_load(path.read_text()) or {}
    if not isinstance(document, dict):
        return False
    embodiment = document.get("embodiment") or {}
    if not isinstance(embodiment, dict):
        return False
    params = embodiment.get("params") or {}
    return isinstance(params, dict) and params.get("enable_cameras") is True


def _load_scene_request(path: Path, job_name: str | None = None) -> SceneRequest:
    """Resolve one unambiguous scene/job and preserve its complete build inputs."""
    if path.suffix.lower() in {".yaml", ".yml"}:
        if job_name is not None:
            raise SystemExit("--job applies only to Arena job JSON files")
        return SceneRequest(path, {}, {})

    job_config = json.loads(path.read_text())
    jobs = job_config.get("jobs") or []
    if not jobs:
        raise SystemExit(f"Job config declares no jobs: {path}")
    if job_name is None and len(jobs) != 1:
        raise SystemExit(
            f"Job config contains {len(jobs)} jobs; select one with --job NAME: {path}"
        )
    matches = (
        jobs
        if job_name is None
        else [job for job in jobs if job.get("name") == job_name]
    )
    if not matches:
        available = ", ".join(str(job.get("name")) for job in jobs)
        raise SystemExit(f"Unknown job {job_name!r}; available jobs: {available}")
    job = matches[0]
    env_args = dict(job.get("arena_env_args") or {})
    environment = env_args.get("environment")
    if not environment:
        raise SystemExit(f"Job config declares no environment: {path}")
    if not str(environment).endswith((".yaml", ".yml")):
        raise SystemExit(
            f"Job selects the environment factory {environment!r} rather than a scene "
            "YAML; pass the scene YAML directly to inspect it."
        )
    # Job configs spell ``environment`` relative to the repository root, not
    # relative to the config's own directory.
    scene = Path(environment)
    if not scene.is_absolute():
        scene = WORKSPACE_ROOT / environment
    variations = job.get("variations") or {}
    if not isinstance(variations, dict):
        raise SystemExit(f"Job variations must be a mapping: {path}")
    return SceneRequest(scene, env_args, dict(variations), job.get("name"))


def _builder_cfg_from(env_args: dict, presets: str | None):
    """Build ``ArenaEnvBuilderCfg`` from a job's own environment arguments.

    Forwarded by dataclass field rather than by an explicit list, so a field
    Arena adds later reaches ``show`` without this needing an edit. Anything the
    job declares that is not a builder field is reported rather than dropped
    silently -- an ignored ``placement_seed`` would be invisible otherwise, and
    would look like the layout itself had changed.
    """
    from dataclasses import fields

    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    accepted = {field.name for field in fields(ArenaEnvBuilderCfg)}
    forwarded = {key: value for key, value in env_args.items() if key in accepted}

    # Clutter relations assert on placement_seed rather than defaulting it, so a
    # scene holding a pile cannot be opened at all unless show supplies one --
    # and the failure arrives as a bare assertion during construction. Seed it
    # from the run's own seed, which keeps the pile reproducible between runs.
    if forwarded.get("placement_seed") is None:
        forwarded["placement_seed"] = forwarded.get("seed", ArenaEnvBuilderCfg.seed)

    # One GaP/cuRobo process serves one synchronous environment, which makes
    # more than one environment meaningless here.
    if forwarded.pop("num_envs", 1) != 1:
        print("[show] ignoring num_envs; inspection always runs one env.", flush=True)
    if presets is not None:
        forwarded["presets"] = presets

    # ``environment`` selects the scene and is consumed above; ``enable_cameras``
    # is a to_arena_env argument, not a builder one.
    ignored = sorted(set(env_args) - accepted - {"environment", "enable_cameras"})
    if ignored:
        print(
            f"[show] job settings not applicable to show: {', '.join(ignored)}",
            flush=True,
        )

    applied = ", ".join(f"{k}={v}" for k, v in sorted(forwarded.items()))
    print(f"[show] builder: num_envs=1, {applied}", flush=True)
    return ArenaEnvBuilderCfg(num_envs=1, **forwarded)


def _cli_seed_args(args: argparse.Namespace) -> dict:
    """Return the seeds the CLI set explicitly, as builder-config arguments.

    ``getattr`` rather than attribute access: the CLI tests build this namespace
    by hand, and a field they omit means "not requested", not an error.
    """
    env_args: dict[str, int] = {}
    for name in ("seed", "placement_seed"):
        value = getattr(args, name, None)
        if value is not None:
            env_args[name] = value
    return env_args


def _arena_builder_from(arena_env, builder_cfg, variations: dict):
    """Construct Arena's builder with the same Hydra overrides as evaluation."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.variations.variations_hydra import overrides_from_dict

    return ArenaEnvBuilder(
        arena_env,
        builder_cfg,
        hydra_overrides=overrides_from_dict(variations),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a VABAR on Isaac Lab Arena scene in the viewer with no policy."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene", help="Scene YAML or Arena job JSON.")
    source.add_argument(
        "--env",
        default=None,
        help=(
            "Registered environment name, e.g. 'vabar_usbc_insertion__medium'. Use this "
            "for environments built by a factory rather than declared as graph YAML."
        ),
    )
    parser.add_argument(
        "--job",
        default=None,
        help="Job name to inspect when the supplied JSON contains multiple jobs.",
    )
    parser.add_argument(
        "--presets",
        default=None,
        choices=("newton",),
        help="Newton physics preset. Overrides a job's own preset.",
    )
    # No --num_envs. One GaP/cuRobo process serves one synchronous environment;
    # a second env would only mirror the first.
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--placement_seed",
        type=int,
        default=None,
        help="Seed for clutter piles; defaults to --seed. Change it to draw a "
        "different layout of the same scene.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without the viewer."
    )
    parser.add_argument(
        "--viz",
        "--visualizer",
        dest="viz",
        default=None,
        help=(
            "Isaac Lab visualizer backends as CSV: kit, newton, rerun, viser. "
            "rerun/viser/newton draw physics geometry and need no display."
        ),
    )
    parser.add_argument(
        "--enable_cameras",
        action="store_true",
        help="Instantiate the scene's cameras (slower; off by default).",
    )
    return parser.parse_args(argv)


def _require_newton(env_cfg, subject: str) -> None:
    """Reject a finalized environment configured for another solver."""
    from isaaclab_newton.physics import NewtonCfg

    if not isinstance(env_cfg.sim.physics, NewtonCfg):
        raise TypeError(
            f"{subject} built unsupported physics backend "
            f"{type(env_cfg.sim.physics).__name__}; VABAR on Isaac Lab Arena supports only Newton"
        )


def _clear_episode_terms(env_cfg) -> None:
    """Clear termination and recorder terms so inspection is never interrupted.

    Inspection must not be interrupted by the environment resetting itself, and
    Arena's success-rate recorder asserts that ``success`` is an active
    termination term, so clearing one group without the other turns any reset
    into an AssertionError.
    """
    for cfg in (
        getattr(env_cfg, "terminations", None),
        getattr(env_cfg, "recorders", None),
    ):
        if cfg is None:
            continue
        for term in vars(cfg):
            if not term.startswith("_") and getattr(cfg, term) is not None:
                setattr(cfg, term, None)


def _apply_viewer_cfg(env) -> None:
    """Put the camera where the environment's ``ViewerCfg`` asks for it.

    ``ViewportCameraController`` sets the camera in its own ``__init__``, before
    the visualizers are initialised, and that call is silently dropped -- so an
    environment's configured eye and lookat never reach the viewport. Arena
    re-applies it after ``gym.make`` in ``make_registered_and_return_cfg``, but
    this CLI builds through ``build_registered`` plus ``gym.make`` directly and
    so never passed through that. The framing an environment ships is part of
    how it is meant to be read: this scene mates a 0.075 mm fit on a bench, and
    seen from the default camera the bench goes edge-on and everything standing
    on it reads as hanging in mid-air over the floor a metre below.
    """
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    reapply_viewer_cfg(env)


def _pin_collider_overlay() -> None:
    """Turn the physics collider wireframe off unless the caller asked for it.

    ``/persistent/physics/visualizationDisplayColliders`` is persisted in the Kit
    user config, so a toggle flipped once in the viewport silently changes what
    every later session shows. This install has it stored as 2 ("All"), which
    draws the bench's collider: a single convex box spanning the full worktop
    footprint and a metre straight down. The table's underframe is mostly air --
    1.5% of that volume has geometry behind it -- so the box reads as a wireframe
    floating over bare floor and looks like a collider that has come adrift from
    its mesh. It has not; laterally it sits inside the visual everywhere.

    Inspecting collision geometry is one of the reasons this command exists, so
    the overlay is not removed -- it is made deterministic and announced. Set
    ISAAC_CAP_SHOW_COLLIDERS=1 to draw it.
    """
    import carb

    wanted = os.environ.get("ISAAC_CAP_SHOW_COLLIDERS") == "1"
    carb.settings.get_settings().set(
        "/persistent/physics/visualizationDisplayColliders", 2 if wanted else 0
    )
    print(
        f"[show] collider overlay: {'on' if wanted else 'off'}"
        f" (ISAAC_CAP_SHOW_COLLIDERS={'1' if wanted else '0'})"
    )


def _describe_physics(env_cfg, presets: str | None) -> str:
    """Name the physics backend the built config actually carries.

    A registered environment sets its own backend inside ``env_cfg_callback``,
    so the ``--presets`` flag describes only what the CLI requested.
    """
    physics = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if physics is None:
        raise RuntimeError("environment has no physics configuration")
    backend = type(physics).__name__
    if presets is not None:
        return f"{backend} (--presets {presets})"
    return f"{backend} (environment's own configuration)"


def _launcher_args(args: argparse.Namespace, enable_cameras: bool) -> dict:
    """Assemble ``AppLauncher`` arguments."""
    launcher_args = {"headless": args.headless, "enable_cameras": enable_cameras}
    # AppLauncher parses --visualizer as CSV into a list; pass the same shape.
    visualizers = [item.strip() for item in (args.viz or "").split(",") if item.strip()]
    if visualizers:
        launcher_args["visualizer"] = visualizers
    return launcher_args


def hold_action(unwrapped):
    """Hold absolute joints, zero relative terms, and preserve binary grippers.

    Also the action row of ``isaaclab_arena_gap.hold_policy.CapHoldPolicy``.
    """
    import torch
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
    from isaaclab.utils.array import convert_to_torch
    from isaaclab_arena.embodiments.droid.actions import (
        BinaryJointPositionZeroToOneAction,
    )

    def term_joint_positions(term):
        positions = term._asset.data.joint_pos
        positions = convert_to_torch(
            positions.torch if hasattr(positions, "torch") else positions,
            device=unwrapped.device,
        )
        joint_ids = term._joint_ids
        if not isinstance(joint_ids, slice):
            joint_ids = convert_to_torch(
                joint_ids, dtype=torch.long, device=positions.device
            )
        return positions[:, joint_ids]

    manager = unwrapped.action_manager
    row = torch.zeros(
        (unwrapped.num_envs, manager.total_action_dim), device=unwrapped.device
    )
    offset = 0
    for name, width in zip(manager.active_terms, manager.action_term_dim, strict=True):
        term = manager.get_term(name)
        segment = slice(offset, offset + width)
        if isinstance(term, JointPositionAction):
            row[:, segment] = term_joint_positions(term)
        elif isinstance(term, BinaryJointPositionZeroToOneAction):
            current = term_joint_positions(term)
            open_distance = torch.linalg.vector_norm(
                current - term._open_command, dim=1
            )
            close_distance = torch.linalg.vector_norm(
                current - term._close_command, dim=1
            )
            closed = (close_distance < open_distance).to(row.dtype)
            row[:, segment] = closed.unsqueeze(1).expand(-1, width)
        offset += width
    return row


@contextmanager
def _closing_simulation_app(simulation_app):
    """Close Kit when guarded scene registration or construction raises."""
    try:
        yield simulation_app
    except BaseException as error:
        print(
            f"[show] setup failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        simulation_app.close()
        raise


def _guarded_setup(simulation_app, operation):
    """Run one setup operation under the tested SimulationApp error guard."""
    with _closing_simulation_app(simulation_app):
        return operation()


def _hold(env, simulation_app) -> None:
    """Step the environment holding the robot still until interrupted.

    Shared by the scene-YAML and registered-environment paths so both drive the
    same loop; a second copy would be free to drift from this one.
    """
    import torch

    unwrapped = env.unwrapped
    try:
        # Captured once: rebuilding the hold target each step commands wherever
        # the arm already is, so the error is always zero and nothing pulls it back.
        with torch.inference_mode():
            held_target = hold_action(unwrapped).clone()
        # Isaac Lab develop can intentionally run Kit headless while a
        # standalone Newton/Rerun/Viser visualizer owns the interactive
        # lifetime.  SimulationApp.is_running() is already false in that mode;
        # SimulationContext is the authoritative cross-backend lifecycle seam.
        while unwrapped.sim.is_headless_or_exist_active_visualizer():
            with torch.inference_mode():
                env.step(held_target)
    except KeyboardInterrupt:
        print("[show] interrupted")
    finally:
        env.close()
        simulation_app.close()


def _run_built_environment(
    simulation_app,
    builder,
    *,
    subject: str,
    ready: str,
    presets: str | None,
) -> None:
    """Build, reset, and drive an environment until interrupted.

    Shared tail of the scene-YAML and registered-environment paths.
    ``subject`` names the source in the Newton check, ``ready`` is the
    readiness line, and ``presets`` is the preset reported with the physics.
    """
    import gymnasium as gym

    env_name, env_cfg, env_kwargs = _guarded_setup(
        simulation_app, builder.build_registered
    )
    _guarded_setup(simulation_app, lambda: _require_newton(env_cfg, subject))

    from isaaclab_arena_gap.newton_camera_rendering import configure_newton_camera_renderers

    camera_count = _guarded_setup(
        simulation_app, lambda: configure_newton_camera_renderers(env_cfg)
    )
    if camera_count:
        print(f"[show] Isaac RTX rendering enabled for {camera_count} camera(s)")

    _clear_episode_terms(env_cfg)

    env = _guarded_setup(
        simulation_app, lambda: gym.make(env_name, cfg=env_cfg, **env_kwargs)
    )
    _guarded_setup(simulation_app, env.reset)
    _apply_viewer_cfg(env)
    _pin_collider_overlay()

    print(f"[show] {ready}")
    # Report what was built, not merely what was requested. Registered
    # environments may configure a specialized Newton profile in their callback.
    print(f"[show] physics: {_describe_physics(env_cfg, presets)}")
    print(
        "[show] holding the robot's current pose; Ctrl-C or close the window to exit."
    )
    _hold(env, simulation_app)


def _show_registered_environment(args: argparse.Namespace) -> None:
    """Open an environment that a registered factory builds, rather than a scene YAML.

    Graph YAML cannot express an `env_cfg_callback`, custom action terms or
    reset events, so environments needing those are factories. They are still
    benchmark tasks and still have to be inspectable.
    """
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(_launcher_args(args, args.enable_cameras))
    simulation_app = app_launcher.app

    from isaaclab_arena_gap.arena.registration import register_arena_components

    _guarded_setup(simulation_app, register_arena_components)

    from isaaclab_arena.assets.registries import EnvironmentRegistry

    def _build():
        registry = EnvironmentRegistry()
        if not registry.is_registered(args.env, ensure_loaded=False):
            available = ", ".join(sorted(registry.get_all_keys()))
            raise SystemExit(
                f"Unknown environment {args.env!r}. Registered: {available}"
            )
        factory_type = registry.get_component_by_name(args.env)
        factory = factory_type()
        cfg_type = registry.get_environment_cfg_type(factory_type)
        # Mirror the job runner (Arena's legacy_eval_config): CLI values reach
        # the cfg only through fields the environment's cfg type declares, and
        # only when the operator actually asked for them. A bare `--env` build
        # keeps the dataclass defaults.
        declared = {field.name for field in fields(cfg_type)}
        cfg_kwargs = {}
        if args.enable_cameras and "enable_cameras" in declared:
            # Without this the app starts camera-enabled while the scene is
            # built camera-less, so --enable_cameras appears to do nothing.
            cfg_kwargs["enable_cameras"] = True
        return factory.build(cfg_type(**cfg_kwargs))

    arena_env = _guarded_setup(simulation_app, _build)

    if args.presets is not None:
        # Arena applies presets after the environment's own env_cfg_callback and
        # replaces sim.physics wholesale, so an environment that tunes physics
        # itself is silently detuned by a preset. Say so rather than let the run
        # look normal.
        print(
            f"[show] warning: --presets {args.presets} replaces any physics this "
            "environment configures itself; drop it to keep the environment's own "
            "tuning.",
            flush=True,
        )

    builder = _guarded_setup(
        simulation_app,
        lambda: _arena_builder_from(
            arena_env, _builder_cfg_from(_cli_seed_args(args), args.presets), {}
        ),
    )
    _run_built_environment(
        simulation_app,
        builder,
        subject=f"Environment {args.env!r}",
        ready=f"environment ready: {args.env}",
        presets=args.presets,
    )


def main() -> None:
    args = _parse_args()
    # getattr, not attribute access: callers construct this namespace directly
    # (the CLI tests do), and a missing field should mean "scene YAML" rather
    # than an AttributeError.
    if getattr(args, "env", None) is not None:
        _show_registered_environment(args)
        return
    request = _load_scene_request(Path(args.scene).resolve(), job_name=args.job)
    scene, env_args = request.scene, request.env_args
    if not scene.is_file():
        raise SystemExit(f"Scene YAML does not exist: {scene}")
    presets = args.presets or env_args.get("presets") or "newton"
    env_args.update(_cli_seed_args(args))
    # Camera extensions must be active before AppLauncher starts. Honor an
    # explicit CLI flag, a job setting, or the scene embodiment params.
    enable_cameras = (
        args.enable_cameras
        or bool(env_args.get("enable_cameras"))
        or _scene_declares_cameras(scene)
    )

    from isaaclab.app import AppLauncher

    launcher_args = _launcher_args(args, enable_cameras)
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    # This folder's assets, embodiments and tasks must be registered before the
    # builder resolves the scene's registry names.
    from isaaclab_arena_gap.arena.registration import register_arena_components

    _guarded_setup(simulation_app, register_arena_components)

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    # The scene is a graph spec; cameras are declared on the embodiment, so
    # whether to instantiate them is decided when the spec becomes an env.
    spec = _guarded_setup(
        simulation_app, lambda: ArenaEnvGraphSpec.from_yaml(str(scene))
    )

    arena_env = _guarded_setup(
        simulation_app, lambda: spec.to_arena_env(enable_cameras=enable_cameras)
    )

    builder = _guarded_setup(
        simulation_app,
        lambda: _arena_builder_from(
            arena_env,
            _builder_cfg_from(env_args, presets),
            request.variations,
        ),
    )
    _run_built_environment(
        simulation_app,
        builder,
        subject=f"Scene {scene}",
        ready=f"scene ready: {scene}",
        presets=presets,
    )


if __name__ == "__main__":
    main()
