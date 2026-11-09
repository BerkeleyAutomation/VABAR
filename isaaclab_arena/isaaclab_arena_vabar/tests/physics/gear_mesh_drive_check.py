"""Drive a seated gear from the board's pinion in the production easy env, no policy.

Run in a fresh Isaac process (see ``test_gear_mesh_drive.py``). It builds
``vabar_gear_insertion__easy`` exactly as the evaluator does, then:

1. checks that MuJoCo's welded board root and pinion sit where Newton's
   colliders do after Arena's reset placement moved the board;
2. clocks the loose gear into mesh on its peg, latches the task's own motor
   (the success term writes the 4 rad/s pinion target), and measures the gear's
   swept-angle rate over the last half of a few seconds.

Prints one ``GEAR_MESH_DRIVE {json}`` line.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os

from isaaclab_arena_gap.task_sources import autolab_root

GEARING = autolab_root() / "tasks/gear_mesh/assets/gearing.py"
MODULE = 0.0025
PINION_TEETH = 14


def _gearing():
    spec = importlib.util.spec_from_file_location("autolab_gear_mesh_gearing", GEARING)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _yaw(quat_xyzw) -> float:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher({"headless": True, "enable_cameras": False}).app

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_newton.physics import NewtonManager

    from isaaclab_arena_gap.arena_show import hold_action
    from isaaclab_arena_vabar.numeric import as_torch
    from isaaclab_arena_vabar.registration import register_components
    from isaaclab_arena_vabar.tasks.gear_insertion.terminations import gear_mesh_success

    register_components()
    registry = EnvironmentRegistry()
    factory_type = registry.get_component_by_name("vabar_gear_insertion__easy")
    cfg_type = registry.get_environment_cfg_type(factory_type)
    arena_env = factory_type().build(cfg_type())
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=1, seed=args.seed, placement_seed=args.seed),
    )
    env_name, env_cfg, env_kwargs = builder.build_registered()
    env = gym.make(env_name, cfg=env_cfg, **env_kwargs)
    env.reset()
    unwrapped = env.unwrapped
    held = hold_action(unwrapped).clone()
    env.step(held)

    board, gear = unwrapped.scene["board"], unwrapped.scene["gear_a"]
    device = unwrapped.device
    pinion_body = board.data.body_names.index("pinion")
    (term,) = [obj for obj in gc.get_objects() if isinstance(obj, gear_mesh_success)]

    model, solver = NewtonManager.get_model(), NewtonManager._solver
    shape_labels = list(model.shape_label)
    shape = next(i for i, label in enumerate(shape_labels) if label.endswith("/board/pinion/tooth_00"))
    mj_body = int(solver.mjw_model.geom_bodyid.numpy()[int(solver.newton_shape_to_mjc_geom.numpy()[shape])])
    mujoco_pinion = solver.mjw_data.xpos.numpy()[0][mj_body]
    newton_pinion = NewtonManager.get_state_0().body_q.numpy()[int(model.shape_body.numpy()[shape])][:3]
    anchor_offset_mm = float(1000.0 * np.linalg.norm(mujoco_pinion - newton_pinion))

    # The loose gear's tooth count: one tooth collider per tooth on its body.
    gear_body = next(i for i, label in enumerate(model.body_label) if label.endswith("/gear_a"))
    body_of_shape = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    teeth = sum(
        1
        for i, label in enumerate(shape_labels)
        if body_of_shape[i] == gear_body and "/tooth_" in label and flags[i] & 2
    )

    gearing = _gearing()
    pinion_frame = gearing.gear_frame(module=MODULE, teeth=PINION_TEETH, face=0.016, bore_radius=0.006)
    gear_frame = gearing.gear_frame(module=MODULE, teeth=teeth, face=0.016, bore_radius=0.006)
    centre_distance = MODULE * (PINION_TEETH + teeth) / 2.0
    board_yaw = _yaw(as_torch(board.data.root_quat_w)[0])
    pinion_pos = as_torch(board.data.body_pos_w)[0, pinion_body]
    pinion_yaw = _yaw(as_torch(board.data.body_quat_w)[0, pinion_body]) - board_yaw
    gear_yaw = board_yaw + gearing.mesh_phase(pinion_frame, gear_frame, centre_distance, pinion_yaw)[0]
    pose = torch.zeros(1, 7, device=device)
    pose[0, :3] = pinion_pos + torch.tensor(
        [centre_distance * math.cos(board_yaw), centre_distance * math.sin(board_yaw), 0.0],
        device=device,
        dtype=pinion_pos.dtype,
    )
    pose[0, 5], pose[0, 6] = math.sin(gear_yaw / 2.0), math.cos(gear_yaw / 2.0)
    gear.write_root_pose_to_sim(pose)
    gear.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
    for _ in range(60):  # settle one second with the motor off
        env.step(held)

    term.latched[:] = True
    swept = [0.0, 0.0]
    last = None
    trace = []
    for _ in range(int(args.seconds * 60)):
        env.step(held)
        now = (
            _yaw(as_torch(gear.data.root_quat_w)[0]),
            _yaw(as_torch(board.data.body_quat_w)[0, pinion_body]),
        )
        if last is not None:
            for k in range(2):
                swept[k] += (now[k] - last[k] + math.pi) % (2.0 * math.pi) - math.pi
        last = now
        trace.append(tuple(swept))
    half = len(trace) // 2
    span = (len(trace) - 1 - half) / 60.0
    gear_rate = (trace[-1][0] - trace[half][0]) / span
    pinion_rate = (trace[-1][1] - trace[half][1]) / span
    ratio = -gear_rate / (pinion_rate * PINION_TEETH / teeth)
    print(
        "GEAR_MESH_DRIVE "
        + json.dumps(
            {
                "seed": args.seed,
                "teeth": teeth,
                "anchor_offset_mm": anchor_offset_mm,
                "pinion_rate": pinion_rate,
                "gear_rate": gear_rate,
                "ratio": ratio,
            }
        ),
        flush=True,
    )
    # Kit can hang in shutdown; the result is already printed.
    del app
    os._exit(0)


if __name__ == "__main__":
    main()
