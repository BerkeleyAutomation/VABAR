# SPDX-License-Identifier: Apache-2.0
"""Measure whether a gear train built from ``tasks/gear_mesh`` actually drives.

The suite's gate. Everything downstream -- the documents, the ``spin`` atom's
calibration, the ground-truth graph -- rests on one question this answers with a
number: **does a pinion turning at a commanded speed carry every gear on the
board at its own ratio, and hold it?** Tooth-on-tooth contact between fifty-odd
convex prisms a couple of millimetres across is not obviously a thing this solver
does well, and the honest way to find out is to build it and look, before a
document is written against it.

The same run produces the two things authoring needs: the **assembled poses**,
which are the goal, and the two populations the minted atom's calibration states
-- a gear in a complete chain, and a gear whose upstream neighbour is missing.

    uv run python tasks/gear_mesh/rigs/gear_bench.py spin
    uv run python tasks/gear_mesh/rigs/gear_bench.py spin --teeth 16,20,24,20
    uv run python tasks/gear_mesh/rigs/gear_bench.py sweep-module
    uv run python tasks/gear_mesh/rigs/gear_bench.py broken

``polish_bench.py`` and ``decant_bench.py``'s sibling: a bench is where a
suite's constants are measured, and the numbers it prints are what the records'
provenance quotes.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

import robosimstudio as rss
from robosimstudio.scene.registry import get_cell
from robosimstudio.specs import CollisionSpec, backend
from robosimstudio.specs.physics import ArrangementSpec
from robosimstudio_bench.taskbuild.discovery import task_module

# The suite's own board, gear and gearing arithmetic, reached by suite rather
# than imported: a task directory has no importable name, and the rig has to
# build the very entity classes the task registers rather than a second copy.
board_mod = task_module("gear_mesh", "assets.board")
gear_mod = task_module("gear_mesh", "assets.gear")
gearing = task_module("gear_mesh", "assets.gearing")

CELL = "bench_075"
BOARD_AT = (-0.02, 0.0)
"""Where the board's plate centre sits on the bench [m, surface frame]."""

PHYSICS = ArrangementSpec(
    control_hz=60.0,
    substeps=16,
    backends=(
        (
            "*",
            backend(
                "mujoco",
                use_mujoco_contacts=True,
                nconmax=16384,
                njmax=32768,
                iterations=100,
                ls_iterations=50,
                cone="elliptic",
                impratio=10.0,
                enable_multiccd=True,
                jacobian="sparse",
            ),
        ),
    ),
    collision=CollisionSpec(
        contact_ke=6.0e4,
        contact_kd=500.0,
        rigid_contact_max=32768,
        rigid_gap=5.0e-5,
        contact_matching="latest",
    ),
)
"""The arrangement ``tasks/gear_mesh/staging.py`` will ship, stated here first.

The contact buffers are the part that is not a copy of ``polish``'s. A four-gear
chain is five toothed rings in mesh, and every pair in contact is several tooth
pairs at once; an overflow of ``nconmax`` does not warn, it escalates into an
illegal memory access. Sized for the busiest frame and measured by ``--contacts``.
"""


def build(
    *,
    teeth: tuple[int, ...],
    module: float,
    pinion: int = 14,
    seated: tuple[bool, ...] | None = None,
    backlash: float = 0.0,
    face: float = 0.0,
    gear_friction: float = 0.0,
    physics: ArrangementSpec | None = None,
    verbose: bool = False,
    **board_kwargs: Any,
) -> dict[str, Any]:
    """An assembled board: the pinion, the pegs, and a gear clocked onto each.

    Every gear is placed at the phase ``mesh_phase`` computes against the one
    before it, chained from the pinion. That is what "assembled" means here --
    not a pose somebody recorded, a configuration derived from the tooth counts,
    which is the same rule ``chain_centres`` places the pegs by.

    ``seated`` may drop a gear from the board (``False``) to leave a hole in the
    chain; the ones after it should then not turn, which is the negative
    population the atom's calibration needs.
    """
    keep = seated if seated is not None else tuple(True for _ in teeth)
    names = tuple(f"gear_{chr(ord('a') + i)}" for i in range(len(teeth)))

    board = board_mod.SpecGearBoard(
        name="board",
        at=BOARD_AT,
        module=module,
        pinion_teeth=pinion,
        stations=tuple(teeth),
        # Every gear stays named, including a dropped one. That is what a failed
        # episode looks like: the policy left it on the bench, and the board
        # reports it off-peg rather than absent. A board that did not know about
        # it would be a different board, and the comparison would be of two.
        gears=names,
        backlash=backlash,
        **board_kwargs,
    )
    frame = board_mod.board_frame(board)
    counts = frame["counts"]
    frames = frame["frames"]
    centres = frame["centres"]

    # Clock every gear against the one before it, from the pinion outward.
    yaws = [0.0]
    backlashes = []
    for i in range(len(counts) - 1):
        yaw, window = gearing.mesh_phase(frames[i], frames[i + 1], centres[i + 1] - centres[i], yaws[i])
        yaws.append(yaw)
        backlashes.append(window)

    seat_z = frame["plate_half"][2] * 2.0 + frame["seat_z"]
    specs = []
    for index, (name, z, on) in enumerate(zip(names, teeth, keep, strict=True)):
        centre = centres[index + 1]
        # A dropped gear is parked well clear rather than deleted, so the model
        # is the same size either way and a comparison is of one thing.
        at = (
            (BOARD_AT[0] + centre, BOARD_AT[1], seat_z)
            if on
            else (BOARD_AT[0] + centre, BOARD_AT[1] - 0.30, frame["seat_z"])
        )
        half = 0.5 * yaws[index + 1]
        specs.append(
            gear_mod.SpecGear(
                name=name,
                teeth=int(z),
                module=module,
                backlash=backlash,
                **({"face": face} if face else {}),
                **({"friction": gear_friction} if gear_friction else {}),
                pose=at,
                quat=(0.0, 0.0, math.sin(half), math.cos(half)),
            )
        )

    scene = rss.Scene(cell=CELL, physics=physics or PHYSICS)
    scene.add(board)
    for spec in specs:
        scene.add(spec)
    sim = scene.build(verbose=verbose)
    # The entity itself, out of the same bin a goal reads a published state from.
    entity = sim.make_context().extras["board"]
    return {
        "sim": sim,
        "board": entity,
        "spec": board,
        "frame": frame,
        "specs": specs,
        "names": names,
        "keep": keep,
        "yaws": yaws,
        "backlash": backlashes,
        "ideal": tuple(board.drive_speed * counts[0] / z for z in teeth),
    }


def press_and_run(rig: dict[str, Any], *, steps: int = 900, settle: int = 120) -> dict[str, Any]:
    """Let the board settle, drive the plunger down, then watch the train.

    The button is pressed through its own joint rather than by a hand: the
    plunger is a position servo held at zero by its return spring, so commanding
    the target to full travel is exactly the press a finger makes, through the
    same degree of freedom. That keeps the bench measuring the drive rather than
    a grasp.
    """
    sim, board = rig["sim"], rig["board"]
    target_q = sim.control.joint_target_q
    for _ in range(settle):
        sim.step()
    before = _read(sim, board)

    commanded = target_q.numpy()
    commanded[board.button_coord] = -float(rig["spec"].button_travel)
    target_q.assign(commanded)

    history: list[dict[str, float]] = []
    for _ in range(steps):
        sim.step()
        history.append(_read(sim, board))
    return {"before": before, "history": history, "final": history[-1]}


def _read(sim, board) -> dict[str, float]:
    return dict(board.published_states(sim.make_context()))


def _tail(history: list[dict[str, float]], key: str, fraction: float = 0.25) -> tuple[float, float]:
    """Mean and spread of one published state over the last of a run."""
    values = _window(history, key, fraction)
    return float(values.mean()), float(values.std())


def _window(history: list[dict[str, float]], key: str, fraction: float = 0.25) -> np.ndarray:
    return np.array([h.get(key, 0.0) for h in history[-max(1, int(len(history) * fraction)) :]])


def _floor(history: list[dict[str, float]], key: str, fraction: float = 0.25) -> float:
    """The 5th percentile of a published state over the last of a run.

    **The number a gate is set from, and not the mean.** A driven gear turns
    through its own backlash: the pinion is servo-held to +-0.02 rad/s while the
    gears it carries swing +-2 on a mean of 3.5, because each one is free between
    the flanks either side of it until a tooth catches. A threshold placed under
    the mean is crossed downward several times a second, and every ``hold-for``
    sequenced behind it restarts. What separates a turning gear from a stopped
    one is the bottom of its ripple, not the middle."""
    return float(np.percentile(_window(history, key, fraction), 5.0))


def _report(rig: dict[str, Any], run: dict[str, Any]) -> bool:
    frame, names, keep = rig["frame"], rig["names"], rig["keep"]
    print(f"  pinion  commanded {rig['spec'].drive_speed:6.2f}   ", end="")
    mean, sd = _tail(run["history"], "pinion_speed")
    print(f"measured {mean:6.2f} +-{sd:5.3f} rad/s")
    print(f"  started={run['final']['started']:.0f}  press_travel={run['final']['press_travel'] * 1000:5.2f} mm")
    healthy = run["final"]["started"] > 0.0
    for index, (name, on) in enumerate(zip(names, keep, strict=True)):
        teeth = frame["counts"][index + 1]
        ideal = rig["ideal"][index]
        mean, sd = _tail(run["history"], f"gear{index}_spin")
        seat, _ = _tail(run["history"], f"gear{index}_seated")
        floor = _floor(run["history"], f"gear{index}_spin")
        ratio = mean / ideal if ideal else 0.0
        flag = ("ok" if ratio > 0.9 else "SLIP") if seat > 0.99 else ("off-peg (dropped)" if not on else "OFF-PEG")
        print(
            f"  {name} z={teeth:<3d} ideal {ideal:6.2f}  measured {mean:6.2f} +-{sd:5.3f}  "
            f"p5 {floor:5.2f} ({floor / ideal if ideal else 0.0:4.2f})  "
            f"ratio {ratio:5.3f}  seated {seat:4.2f}  {flag}"
        )
        healthy = healthy and (ratio > 0.9 and seat > 0.99 if on else True)
    print(f"  turning_fraction={run['final']['turning_fraction']:.2f}")
    return healthy


def cmd_spin(args) -> int:
    teeth = tuple(int(v) for v in args.teeth.split(","))
    rig = build(teeth=teeth, module=args.module, pinion=args.pinion, backlash=args.backlash, verbose=args.verbose)
    print(f"\nassembled board: pinion z={args.pinion}, stations z={teeth}, module {args.module * 1000:.2f} mm")
    print(f"  peg radius {rig['frame']['peg_radius'] * 1000:.3f} mm, seats at x = ", end="")
    print(", ".join(f"{c * 1000:.1f}" for c in rig["frame"]["centres"][1:]), "mm")
    print("  clocking: " + ", ".join(f"{math.degrees(y):+.2f}d" for y in rig["yaws"][1:]))
    print("  backlash: " + ", ".join(f"{math.degrees(b):.3f}d" for b in rig["backlash"]))
    run = press_and_run(rig, steps=args.steps)
    ok = _report(rig, run)
    rig["sim"].close()
    print(f"\n  VERDICT: {'the train drives' if ok else 'the train does NOT drive'}\n")
    return 0 if ok else 1


def cmd_broken(args) -> int:
    """The negative population: one gear missing from the middle of the chain."""
    teeth = tuple(int(v) for v in args.teeth.split(","))
    for drop in range(len(teeth)):
        keep = tuple(i != drop for i in range(len(teeth)))
        rig = build(teeth=teeth, module=args.module, pinion=args.pinion, seated=keep)
        print(f"\nwithout {rig['names'][drop]} (z={teeth[drop]}):")
        run = press_and_run(rig, steps=args.steps)
        _report(rig, run)
        rig["sim"].close()
    return 0


def reach_report(variant: str, tol: float = 0.01) -> list[tuple[str, np.ndarray, float]]:
    """Solve error at every point this layout asks the arm to get to.

    Not a grid sweep -- ``tools/reach_bench.py`` is that, and it answers a
    different question. This asks the suite's own: the stations it has to seat a
    gear on, the spots it has to pick one off, and the button. A layout is
    feasible only if every one solves, and a station that does not is a variant
    nobody can finish however good the policy is.

    Top-down, tool +z into the bench, which is the only approach a bore over a
    peg admits. Two heights per station: the seat itself, and an approach clear
    above the peg's top.
    """
    from robosimstudio.scene.frames import in_surface  # noqa: PLC0415
    from robosimstudio_bench.envs.env import Env  # noqa: PLC0415

    env = Env(f"gear_mesh/{variant}", renderer="")
    try:
        board = next(e for e in env.request.entities if type(e).__name__ == "SpecGearBoard")
        frame = board_mod.board_frame(board)
        cell = get_cell(env.request.scene)
        targets: list[tuple[str, tuple[float, float, float]]] = []
        top = board.plate_thickness + frame["peg_height"]
        for index, centre in enumerate(frame["centres"][1:]):
            at = (board.at[0], board.at[1] + centre)
            targets.append((f"station{index}-seat", (*at, board.plate_thickness + frame["seat_z"])))
            targets.append((f"station{index}-above", (*at, top + 0.06)))
        bx, by = frame["button_at"]
        targets.append(("button", (board.at[0] + bx, board.at[1] + by, board.plate_thickness + 0.05)))
        for gear in env.request.objects:
            if gear.name.startswith("gear_"):
                targets.append((f"pick-{gear.name}", (gear.pose[0], gear.pose[1], gear.pose[2] + 0.02)))

        ik = env.arm(next(iter(env.handles.arms)))
        seed = np.asarray(env.state.joint_q.numpy(), dtype=np.float64)
        out = []
        for label, surface in targets:
            world = np.asarray(in_surface(cell, (*surface, 0.0, 0.0, 0.0, 1.0)), dtype=np.float64)[:3]
            _, err = ik.solve_best([seed], world, (0.0, 0.0, -1.0))
            out.append((label, world, float(err)))
        return out
    finally:
        env.close()


def cmd_reach(args) -> int:
    print(f"building gear_mesh/{args.variant} ...", flush=True)
    rows = reach_report(args.variant, args.tol)
    bad = 0
    for label, world, err in rows:
        flag = "ok" if err <= args.tol else "UNREACHABLE"
        bad += int(err > args.tol)
        print(
            f"  {label:24s} world ({world[0]:+.3f},{world[1]:+.3f},{world[2]:+.3f})  err {err * 1000:7.2f} mm  {flag}"
        )
    print(f"\n  worst {max(r[2] for r in rows) * 1000:.2f} mm over {len(rows)} targets; {bad} unreachable")
    return 1 if bad else 0


def cmd_contacts(args) -> int:
    """Peak simultaneous contacts on a driven board -- what sizes the buffers.

    ``nconmax`` and ``njmax`` do not warn when they overflow; they escalate into
    an illegal memory access, so the arrangement has to be sized for the busiest
    frame rather than the average one. This is that frame, on the longest chain
    this suite ships, driven rather than resting -- a settled board is not the
    busy case, a turning one is.
    """
    teeth = tuple(int(v) for v in args.teeth.split(","))
    rig = build(teeth=teeth, module=args.module, pinion=args.pinion)
    sim, board = rig["sim"], rig["board"]
    for _ in range(120):
        sim.step()
    target = sim.control.joint_target_q
    commanded = target.numpy()
    commanded[board.button_coord] = -float(rig["spec"].button_travel)
    target.assign(commanded)
    peak = 0
    counts: list[int] = []
    for _ in range(args.steps):
        sim.step()
        probe = sim.make_context().contacts
        if probe is None:
            print("  no contact probe: the arrangement must run mujoco with use_mujoco_contacts=True")
            rig["sim"].close()
            return 1
        # `update()` first: the probe caches a pull, and reading without one
        # reports the last frame anybody asked for -- which, on a fresh probe,
        # is no frame at all and reads as a scene with no contacts in it.
        probe.update()
        now = int(len(probe.penetrations(-1.0e9)))
        counts.append(now)
        peak = max(peak, now)
    arrangement = PHYSICS.backends[0][1]
    nconmax = dict(arrangement.params).get("nconmax", 0)
    print(f"\nstations z={teeth}, driven for {args.steps} steps")
    print(f"  contacts: median {int(np.median(counts))}  peak {peak}")
    print(f"  nconmax {nconmax} is {nconmax / max(1, peak):.1f}x the peak")
    rig["sim"].close()
    return 0


def cmd_sweep_module(args) -> int:
    teeth = tuple(int(v) for v in args.teeth.split(","))
    print(f"\nmodule sweep, stations z={teeth}\n")
    for module in (0.002, 0.0025, 0.003):
        rig = build(teeth=teeth, module=module, pinion=args.pinion)
        print(
            f"module {module * 1000:.2f} mm  (largest OD {2 * max(f['r_tip'] for f in rig['frame']['frames']) * 1000:.1f} mm)"
        )
        run = press_and_run(rig, steps=args.steps)
        _report(rig, run)
        rig["sim"].close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teeth", default="16,20,24", help="tooth counts on the pegs, in chain order")
    parser.add_argument("--pinion", type=int, default=14, help="the drive gear's tooth count")
    parser.add_argument("--module", type=float, default=0.0025, help="tooth size [m]")
    parser.add_argument("--backlash", type=float, default=0.0, help="pitch-circle slack [m]; 0 takes the default")
    parser.add_argument("--steps", type=int, default=900, help="control steps to watch after the press")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("spin", help="does an assembled train drive?").set_defaults(run=cmd_spin)
    sub.add_parser("broken", help="what a chain with a hole in it does").set_defaults(run=cmd_broken)
    sub.add_parser("sweep-module", help="tooth size against rate error").set_defaults(run=cmd_sweep_module)
    sub.add_parser("contacts", help="peak simultaneous contacts, which is what sizes the buffers").set_defaults(
        run=cmd_contacts
    )
    reach = sub.add_parser("reach", help="can the arm get to every station, spot and the button?")
    reach.add_argument("--variant", default="quad", help="which board's layout to probe")
    reach.add_argument("--tol", type=float, default=0.01, help="solve error that still counts as reach [m]")
    reach.set_defaults(run=cmd_reach)
    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
