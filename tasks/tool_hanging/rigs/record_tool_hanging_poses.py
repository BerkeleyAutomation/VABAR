# SPDX-License-Identifier: Apache-2.0
"""Record the tool-hanging suite's hung poses, from its own routines.

Rolls each routine-driven variant once through the standard rollout path --
:func:`robosimstudio_bench.run.rollout` driving ``PolicyScripted``, exactly
what ``tools/bench_run.py --policy scripted`` does -- and writes the settled object
poses to ``tasks/tool_hanging/poses/<variant>.json``, keyed by object
name, seven free-joint coordinates each. ``tests/test_tool_hanging_scenes.py``
plants these back and holds the suite to them: goal false at spawn, true and
stable at the recorded hang.

``hammers`` is not recorded: its holder's slots are side mouths under a solid
top (measured off the mesh), so no drop can produce a hang -- inserting a
handle through a mouth is the engagement the suite's README lists as open.
The drop table below remains for whoever picks that item up.

Usage::

    uv run python tasks/tool_hanging/rigs/record_tool_hanging_poses.py [--variants screwdrivers,...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POSES = Path(__file__).resolve().parents[1] / "poses"
"""The suite's own ``poses/``, beside this rig: the recording belongs to the task."""

_S = float(np.sqrt(0.5))
HAMMER_DROPS = {
    # Handle vertical through the U, head across the cradle top, extending
    # toward the robot so it clears the board: R_z(180) . R_y(90), which maps
    # the handle's local +x to world -z and the head's local z run to world -x.
    # The holders mount at y_origin 0.105/0.305 but their -90-degree turn puts
    # the cradles' centres at world y -/+0.100 (the mesh runs along -local_x
    # from its mount); the origin offsets place the head's underside centre
    # 10 mm above the cradle top for a short drop.
    "hammer_1": (0.898, -0.0845, 1.23, -_S, 0.0, _S, 0.0),
    "hammer_2": (0.898, 0.1155, 1.23, -_S, 0.0, _S, 0.0),
}


def _settled(env, steps: int) -> None:
    for _ in range(steps):
        env.step(env.idle_action())


def _object_poses(env) -> dict[str, list[float]]:
    body_q = env.state.body_q.numpy()
    return {name: [float(v) for v in body_q[handle.body][:7]] for name, handle in env.handles.objects.items()}


def _plant(env, poses: dict[str, tuple[float, ...]]) -> None:
    import newton  # noqa: PLC0415

    for state in (env.pipeline.state_0, env.pipeline.state_1):
        q = state.joint_q.numpy()
        qd = state.joint_qd.numpy()
        for name, pose in poses.items():
            c = env.handles.objects[name].coord_start
            q[c : c + 7] = np.asarray(pose, dtype=np.float64)
        qd[:] = 0.0
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        newton.eval_fk(env.model, state.joint_q, state.joint_qd, state)


def _mirror_wrench_17(env) -> bool:
    """Derive the 17's hang from the 19's achieved one, and settle it.

    The 19 hangs reliably; the 17's grasp does not capture yet (the open
    routine item). The two rings hang on identical hooks 120 mm apart, so the
    19's measured ring-to-hook relation, applied to the 17's own ring
    landmark, is a physically checkable hang -- planted, settled, and only
    recorded if the evaluator agrees. The contract test needs the goal
    *satisfiable*; the routine's capture gap stays a routine bug.
    """
    from robosimstudio.assetlib.registry import get_asset  # noqa: PLC0415
    from robosimstudio.core.math import quat_to_mat  # noqa: PLC0415

    # Through the asset registry rather than by importing the task: a task is a
    # directory with no importable name, and a rig may not reach into `tests/`
    # for the discovery seam the tests use. The records are registered, so the
    # public lookup answers it.
    WRENCH_150, WRENCH_200 = get_asset("wrench_adjustable_150"), get_asset("wrench_adjustable_200")

    body_q = env.state.body_q.numpy()
    pose19 = np.asarray(body_q[env.handles.objects["ring_spanner_19"].body][:7], dtype=np.float64)
    ring19 = pose19[:3] + quat_to_mat(pose19[3:7]) @ np.asarray(WRENCH_150.feature("ring").payload["center"])
    ring17_local = np.asarray(WRENCH_200.feature("ring").payload["center"])
    target_ring = ring19 + np.array([0.0, -0.120, 0.0])
    pos17 = target_ring - quat_to_mat(pose19[3:7]) @ ring17_local
    _plant(env, {"ring_spanner_17": (*pos17, *pose19[3:7])})
    _settled(env, 180)
    _, _, _, _, info = env.step(env.idle_action())
    return bool(info.get("success", False))


def record(variant: str) -> bool:
    from robosimstudio_bench.envs.env import Env  # noqa: PLC0415
    from robosimstudio_bench.policies.scripted import PolicyScripted  # noqa: PLC0415
    from robosimstudio_bench.run import rollout  # noqa: PLC0415

    env = Env(f"tool_hanging/{variant}", renderer="")
    try:
        if variant == "hammers":
            env.reset(0)
            _plant(env, HAMMER_DROPS)
            _settled(env, 240)
            _, reward, _, _, info = env.step(env.idle_action())
            ok = bool(info.get("success", False))
            print(f"[{variant}] planted drop: success={ok} reward={reward:.2f}")
        else:
            result = rollout(env, PolicyScripted(env), episode_id=0)
            ok = result.success
            print(f"[{variant}] rollout: success={result.success} reward={result.reward:.2f} steps={result.steps}")
            if variant == "wrenches" and not ok and result.reward == 0.5:
                ok = _mirror_wrench_17(env)
                print(f"[{variant}] 17 derived from the 19's hang and settled: success={ok}")
            if ok:
                # Record the *re-settled* state, not the rollout's last frame.
                # A plant zeroes velocities into a state that is not a contact
                # equilibrium; the contract test settles from it, so what it
                # needs stored is the fixed point that settling reaches --
                # scissors_2 slides 20 mm down its hook arm on the way there.
                _plant(env, {k: tuple(v) for k, v in _object_poses(env).items()})
                _settled(env, 180)
                _, _, _, _, info = env.step(env.idle_action())
                ok = bool(info.get("success", False))
                print(f"[{variant}] re-settled fixed point: success={ok}")
            if not ok:
                metrics = result.info.get("metrics", {})
                for key in sorted(metrics):
                    if any(word in key for word in ("threaded", "force", "speed", "containment", "height")):
                        print(f"    {key} = {metrics[key]}")
        POSES.mkdir(exist_ok=True)
        out = POSES / f"{variant}.json"
        out.write_text(json.dumps(_object_poses(env), indent=2) + "\n")
        print(f"[{variant}] wrote {out}{'' if ok else '  (FROM A FAILED RUN -- do not commit)'}")
        return ok
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="screwdrivers,scissors,pliers")
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="fresh attempts per variant; the runtime's thread timing is not pinned, so rollouts genuinely differ",
    )
    args = parser.parse_args()
    failed = []
    for v in (v.strip() for v in args.variants.split(",") if v.strip()):
        if not any(record(v) for _ in range(max(1, args.retries))):
            failed.append(v)
    if failed:
        print(f"NOT HUNG after retries: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
