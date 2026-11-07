# SPDX-License-Identifier: Apache-2.0
"""Does a candidate's cable start where the reference's does? One episode, no stepping.

    python tools/fast_tune/spawn_check.py 'sdx:rod=2' --task cable_route/port_easy --episode 10009

Builds the task under the candidate, resets to the episode, and prints the rod's
centerline (``Centerline.centerline`` -- capsule endpoints on the chain, nodes on
the Cosserat rod) as JSON, so two runs can be diffed node for node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cable_route


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--task", default="cable_route/port_easy")
    parser.add_argument("--episode", type=int, default=10009)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cable_route.install(args.candidate)
    from robosimstudio_bench.envs.env import make_env

    env = make_env(args.task)
    env.reset(args.episode)
    cable = env.scene.by_name[env.task.deformables[0].name]
    snapshot = env.snapshot() if hasattr(env, "snapshot") else None
    if snapshot is None or not len(cable.centerline(snapshot)):
        # The chain reads body_q off any object that has it; the centerline its probe.
        class _Snap:
            body_q = env.state.body_q.numpy()
            particle_q = env.state.particle_q.numpy() if env.state.particle_q is not None else None

        snapshot = _Snap()
    nodes = cable.centerline(snapshot)
    Path(args.out).write_text(json.dumps({"candidate": args.candidate, "nodes": nodes.tolist()}))
    print(f"{args.candidate}: {len(nodes)} nodes, first {nodes[0].round(5)}, last {nodes[-1].round(5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
