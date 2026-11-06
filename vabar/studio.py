# SPDX-License-Identifier: Apache-2.0
"""Open a VABAR task in the RoboSimStudio Studio: one browser page over the scene.

    python -m vabar studio                                         # the first task; pick others from the menu
    python -m vabar studio --task tool_hanging/wrenches_level1 --port 8080
    python -m vabar studio --task gear_mesh/gearmesh-easy-pair-01 --motion plan

Open the printed URL (forward the port when the machine is remote). The page is
upstream's :class:`robosimstudio_studio.StudioSession`, unchanged: its **Task**
menu lists every VABAR suite and variant, the **Goal** panel scores the scene
clause by clause, and the **Policy** panel runs the task's scripted ``routine``
or any graph it ships. What is added here: :func:`vabar.install` before the page
is built, and an experiment config (``--config``, ``--set``; see
:mod:`vabar.config`) -- its per-suite physics, cameras and budgets rewrite the
variants the menu offers, and its environment (tool servers kept warm, the
planner cached) applies to the graphs the page runs.
"""

from __future__ import annotations

import argparse
import sys

from . import config as cfg
from . import install

FLEET_SOLUTIONS = {
    "graph": "robosimstudio_policy_agent.graph_solution:SolutionGraph",
    "idle": "robosimstudio_bench.solutions.base:SolutionIdle",
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vabar studio", description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default="", help="suite/variant to open; defaults to the first registered")
    parser.add_argument("--episode", type=int, default=0, help="which episode to open")
    parser.add_argument("--port", type=int, default=8080, help="viser port (the next free one is used if busy)")
    parser.add_argument("--share", action="store_true", help="also request a public viser share URL")
    parser.add_argument("--out", default="outputs/studio", help="where run directories, videos and traces go")
    parser.add_argument("--device", default=None, help="Warp device, e.g. cuda:0")
    parser.add_argument("--motion", default="", help="how robot.go_to_pose moves in a graph: osc or plan")
    parser.add_argument("--no-privileged", action="store_true", help="start without the sim.* tools for graphs")
    parser.add_argument("--dark", action="store_true", help="dark page")
    parser.add_argument("--num-worlds", type=int, default=1, help="above 1, a grid of worlds each on its own episode")
    parser.add_argument(
        "--fleet-solution",
        default="",
        help=f"what drives the worlds when --num-worlds > 1: {', '.join(FLEET_SOLUTIONS)}",
    )
    parser.add_argument("--graph", default="gt", help="the graph a --fleet-solution graph runs")
    parser.add_argument(
        "--config",
        default=None,
        help=f"experiment config: {', '.join(cfg.names())} or a .toml path (default: $VABAR_CONFIG, else default)",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override one config key; repeatable"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        config = cfg.activate(cfg.resolve(args.config, args.set))
    except cfg.ConfigError as exc:
        print(f"[vabar] {exc}", file=sys.stderr)
        return 2
    cfg.apply_env(config, args.task)
    install()
    from robosimstudio_bench.taskbuild.registry import list_tasks
    from robosimstudio_studio import StudioConfig, StudioSession

    task = config.map_task(args.task or list_tasks()[0])
    overrides = {
        "port": args.port,
        "share": args.share,
        "out": args.out,
        "device": args.device,
        "dark_mode": args.dark,
        "num_worlds": args.num_worlds,
        "privileged": not args.no_privileged,
    }
    if args.motion:
        overrides["motion"] = args.motion
    if args.fleet_solution:
        overrides["fleet_solution"] = FLEET_SOLUTIONS.get(args.fleet_solution, args.fleet_solution)
        if args.fleet_solution == "graph":
            overrides["fleet_solution_kwargs"] = (("graph", args.graph),)
    StudioSession(task, episode=args.episode, config=StudioConfig(**overrides)).run()
    return 0
