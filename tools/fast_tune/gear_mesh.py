# SPDX-License-Identifier: Apache-2.0
"""gear_mesh's ``fast`` config, chosen by measurement: the sweep that chose it, re-runnable.

Every candidate is the shipped arrangement (``gear_mesh_960hz@v1``) and rig
(``gear_mesh_views@v2``) with knobs turned, named by the knobs: ``ss8+ls20``
is sixteen substeps down to eight and the line search capped at 20. Nothing
shipped is edited. A candidate becomes a *new* variant of the lowered task
(``dataclasses.replace(task, name=..., physics=..., cameras=...)``) added with
``SUITES.replace``, in the probing process only, and is run through VABAR's own
``python -m vabar run`` entry point, so a probe is the cell's code path with one
document swapped.

    # the knobs, and what a candidate name means
    python tools/fast_tune/gear_mesh.py knobs

    # probe 1, speed: 600 idle steps, no renderer, on the L1 and L2 reference instances
    python tools/fast_tune/gear_mesh.py speed --cands ref,ss8,it20,ls20 --out OUT
    # probe 2, the gear-train gate (rigs/gear_bench.py spin on the candidate arrangement)
    python tools/fast_tune/gear_mesh.py gate --cands ref,ss8 --parallel 4 --out OUT
    # probe 3, graph fidelity: gap_perception under the cell's flags, --renderer rtx,
    # with the runner given strict turns (the probe of record; see below)
    python tools/fast_tune/gear_mesh.py graph --paced 0.25 --cands ref --rigs views --parallel 3 --out OUT
    python tools/fast_tune/gear_mesh.py graph --paced 0.25 --cands ss8 --rigs s50 --parallel 3 --out OUT
    # the shipped cell protocol exactly (shipped graph, every camera rendered every step)
    python tools/fast_tune/gear_mesh.py graph --observation per-step --cands @gear_mesh_fast@v1 \
        --rigs @gear_mesh_views_fast@v1 --variants gearmesh-easy-pair-01 --out OUT
    # the table every probe above feeds
    python tools/fast_tune/gear_mesh.py table --out OUT

    # one probe in this process -- what the sweep's subprocesses run
    python tools/fast_tune/gear_mesh.py run --cand ss8 --rig s50 --variant gearmesh-easy-pair-01 -- \\
        --solve idle --max-steps 600 --out OUT/speed --name ss8__views__gearmesh-easy-pair-01
    python tools/fast_tune/gear_mesh.py spin --cand ss8 --json OUT/gate/ss8.json

**Candidate names.** ``ref`` is the shipped arrangement. Otherwise ``+``-joined
knobs, applied left to right (``knobs`` lists them): ``ss<N>`` substeps,
``it<N>`` solver iterations, ``ls<N>`` line-search iterations, ``pyr``
pyramidal cone, ``noccd`` multiccd off, ``imp<X>`` impratio, ``nc<N>``/``nj<N>``
contact and constraint buffers per world, ``ce<N>`` collide_every, ``gap<X>``
CollisionSpec.rigid_gap, ``tol<N>`` solver tolerance 1e-N, ``ccd<N>``
ccd_iterations, ``cg`` the CG solver. ``@name`` is a *registered* arrangement
(``@gear_mesh_fast@v1``), which is how the shipped winner is re-verified. A rig
is ``views`` (the shipped one), ``s<P>`` (every camera at P percent of its
pixels per side, ``CameraSpec.scaled``) or ``@name``.

**What each probe proves the override did.** ``run`` wraps ``SolverMuJoCo`` and
the scene pipeline in its own process (RoboSimStudio is not edited) and writes
``tune_probe.json`` beside the run's ``run_meta.json``: the options the compiled
MuJoCo model actually carries (iterations, line search, cone, impratio,
tolerance, multiccd, buffers), whether a Newton collision pipeline exists (so
whether ``collide_every`` can do anything), and -- with ``--watch-contacts`` --
the peak contact and constraint-row counts per step, which is what sizes the
buffers. ``run_meta.json``'s ``builds[]`` carries the substeps.

**Graph runs are single-world.** The fleet trick (``--num-worlds 2``) is refused
for this suite: ``CannotReplicate ... these entities contribute substep physics
written for one world: board`` (the board's motor and latch). Single-world RTX
renders every camera every control step, ~5 steps/s, unless the workflow says it
reads on demand -- so ``graph`` runs a staged copy of ``gap_perception`` whose
only difference is ``meta.observation_mode: "on_demand"`` (the runtime's own
switch: the per-step frames and gap's 10 Hz stream stop, ``robot.get_observation``
still renders fresh RGB-D). ``--observation per-step`` runs the shipped graph
exactly as the cell does. In both, the shipped runner keeps stepping while the
graph thinks, so a faster arrangement spends more of the step budget per second
of SAM 3 or cuRobo -- measured here: ``ss8`` truncated two L1 episodes at 12,000
steps with every gear seated, and the same episode with the budget doubled
succeeded. ``--paced S`` removes that confound (see :func:`pace`): each control
step waits, up to S seconds, until the graph has something for the simulation,
which is the rule the fleet's barrier applies to a wave. Compare candidates only
against the reference run under the same mode.

Repeats: ``graph --tag r2`` runs the same jobs under ``<name>__r2``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SUITE = "gear_mesh"
REF_PHYSICS_NAME = "gear_mesh_960hz@v1"
REF_RIG_NAME = "gear_mesh_views@v2"
GRAPH = "gap_perception"
CELL_FLAGS = (
    "--solve", "graph", "--motion", "curobo", "--no-privileged", "--perception", "vision", "--bus-timeout", "180",
)  # fmt: skip
"""``python -m vabar bench --cell gear_meshing/L1 --print``'s protocol, less the graph (named per mode)."""

SPEED_VARIANTS = ("gearmesh-easy-pair-01", "gearmesh-medium-train-001")
GRAPH_VARIANTS = (
    "gearmesh-easy-pair-01",
    "gearmesh-easy-pair-02",
    "gearmesh-easy-pair-03",
    "gearmesh-medium-train-001",
    "gearmesh-medium-train-002",
)
GATE_RATIO = (0.95, 0.99)
"""The spin gate: every gear's mean ratio over the last quarter of the drive. The shipped arrangement reads 0.967."""


# --------------------------------------------------------------------------- environment
def _skills_checkout() -> None:
    """Point VABAR at an open-robot-skills checkout before it reads one.

    ``vabar.skill_registries`` looks beside the checkout; a worktree is nested
    deeper than the clone it was made from, so walk up until one is found.
    """
    if os.environ.get("VABAR_OPEN_ROBOT_SKILLS") or os.environ.get("GAP_SKILLS_PATH"):
        return
    for parent in REPO.parents:
        candidate = parent / "open-robot-skills"
        if candidate.is_dir():
            os.environ["VABAR_OPEN_ROBOT_SKILLS"] = str(candidate)
            return


def _install() -> None:
    _skills_checkout()
    sys.path.insert(0, str(REPO))
    import vabar

    vabar.install()


# --------------------------------------------------------------------------- knobs
KNOB_DOC = {
    "ss<N>": "ArrangementSpec.substeps (even)",
    "it<N>": "mujoco iterations",
    "ls<N>": "mujoco ls_iterations",
    "pyr": "mujoco cone='pyramidal'",
    "noccd": "mujoco enable_multiccd=False",
    "imp<X>": "mujoco impratio",
    "nc<N>": "mujoco nconmax (per world)",
    "nj<N>": "mujoco njmax (per world)",
    "ce<N>": "CollisionSpec.collide_every",
    "gap<X>": "CollisionSpec.rigid_gap [m], e.g. gap2e-5",
    "tol<N>": "mujoco tolerance = 1e-N",
    "ccd<N>": "mujoco ccd_iterations",
    "cg": "mujoco solver='cg'",
    "dense": "mujoco jacobian='dense'",
}


def _mj(arr, **params):
    return arr.with_backend("*", arr.backend_for("*").with_(**params))


def _col(arr, **changes):
    return dataclasses.replace(arr, collision=dataclasses.replace(arr.collision, **changes))


def _knob(arr, token: str):
    if m := re.fullmatch(r"ss(\d+)", token):
        return dataclasses.replace(arr, substeps=int(m[1]))
    if m := re.fullmatch(r"it(\d+)", token):
        return _mj(arr, iterations=int(m[1]))
    if m := re.fullmatch(r"ls(\d+)", token):
        return _mj(arr, ls_iterations=int(m[1]))
    if token == "pyr":
        return _mj(arr, cone="pyramidal")
    if token == "noccd":
        return _mj(arr, enable_multiccd=False)
    if m := re.fullmatch(r"imp([\d.]+)", token):
        return _mj(arr, impratio=float(m[1]))
    if m := re.fullmatch(r"nc(\d+)", token):
        return _mj(arr, nconmax=int(m[1]))
    if m := re.fullmatch(r"nj(\d+)", token):
        return _mj(arr, njmax=int(m[1]))
    if m := re.fullmatch(r"ce(\d+)", token):
        return _col(arr, collide_every=int(m[1]))
    if m := re.fullmatch(r"gap([\d.eE+-]+)", token):
        return _col(arr, rigid_gap=float(m[1]))
    if m := re.fullmatch(r"tol(\d+)", token):
        return _mj(arr, tolerance=float(10.0 ** -int(m[1])))
    if m := re.fullmatch(r"ccd(\d+)", token):
        return _mj(arr, ccd_iterations=int(m[1]))
    if token == "cg":
        return _mj(arr, solver="cg")
    if token == "dense":
        return _mj(arr, jacobian="dense")
    raise SystemExit(f"unknown knob {token!r}; `knobs` lists them")


def _loaded() -> None:
    """Discover the suites, which is what registers the suite's arrangement and rig names."""
    from robosimstudio_bench.taskbuild.registry import get_suite

    get_suite(SUITE)


def arrangement(cand: str):
    """The ArrangementSpec a candidate name stands for."""
    from robosimstudio_bench.taskbuild.arrangements import get_arrangement

    _loaded()
    if cand.startswith("@"):
        return get_arrangement(cand[1:])
    arr = get_arrangement(REF_PHYSICS_NAME)
    if cand == "ref":
        return arr
    for token in cand.split("+"):
        arr = _knob(arr, token)
    return arr


def camera_rig(rig: str):
    """The camera tuple a rig name stands for."""
    from robosimstudio_bench.taskbuild.arrangements import get_camera_rig

    _loaded()
    if rig.startswith("@"):
        return get_camera_rig(rig[1:])
    cams = get_camera_rig(REF_RIG_NAME)
    if rig == "views":
        return cams
    if m := re.fullmatch(r"s(\d+)", rig):
        f = int(m[1]) / 100.0
        return tuple(c.scaled(round(c.width * f), round(c.height * f)) for c in cams)
    raise SystemExit(f"unknown rig {rig!r}: views, s<percent> or @registered")


def describe(arr) -> dict[str, Any]:
    """An arrangement as the knobs this sweep turns."""
    b = arr.backend_for("*")
    c = arr.collision
    return {
        "substeps": arr.substeps,
        "control_hz": arr.control_hz,
        **{k: v for k, v in b.params},
        "collide_every": c.collide_every,
        "rigid_gap": c.rigid_gap,
        "contact_ke": c.contact_ke,
        "contact_kd": c.contact_kd,
        "rigid_contact_max": c.rigid_contact_max,
        "contact_matching": c.contact_matching,
    }


def slug(cand: str) -> str:
    return cand.replace("@", "at-").replace("+", "_")


def derive(variant: str, cand: str, rig: str) -> str:
    """Register ``<variant>.<cand>.<rig>`` beside the shipped variants; return its full name."""
    from robosimstudio_bench.taskbuild.arrangements import register_arrangement, register_camera_rig
    from robosimstudio_bench.taskbuild.registry import SUITES, get_suite

    arr = arrangement(cand)
    cams = camera_rig(rig)
    # Registered under sweep names so the run's process holds the candidate by
    # name as well as by value; nothing outside this process sees them.
    if not cand.startswith("@"):
        register_arrangement(f"gear_mesh_tune_{slug(cand)}@sweep", arr)
    if not rig.startswith("@"):
        register_camera_rig(f"gear_mesh_views_tune_{slug(rig)}@sweep", cams)
    suite = get_suite(SUITE)
    base = suite.variant(variant)
    name = f"{variant}.{slug(cand)}.{slug(rig)}"
    if name in suite.variant_names:
        return f"{SUITE}/{name}"
    spec = dataclasses.replace(base, name=name, physics=arr, cameras=cams)
    SUITES.replace(SUITE, dataclasses.replace(suite, variants=(*suite.variants, spec)), origin=__name__)
    return f"{SUITE}/{name}"


# --------------------------------------------------------------------------- in-process instrumentation
PROBE: dict[str, Any] = {"solvers": [], "pipelines": [], "contacts": {"peak_nacon": 0, "peak_nefc": 0, "samples": 0}}


def instrument(watch_every: int = 0) -> None:
    """Record what the solver was really built with; optionally watch the contact counts.

    Patches classes in this process only. ``watch_every`` > 0 reads ``nacon`` and
    ``nefc`` off the MuJoCo-Warp data every that many control steps -- a device
    read each time, so it is off for the speed probe.
    """
    from newton.solvers import SolverMuJoCo
    from robosimstudio.core.pipeline import ScenePipeline
    from robosimstudio.core.schedule import ScheduleExecutor

    init = SolverMuJoCo.__init__

    def solver_init(self, *a, **kw):
        init(self, *a, **kw)
        with contextlib.suppress(Exception):
            PROBE["solvers"].append(solver_options(self))

    SolverMuJoCo.__init__ = solver_init

    pinit = ScenePipeline.__init__

    def pipeline_init(self, *a, **kw):
        pinit(self, *a, **kw)
        PROBE["pipelines"].append(
            {
                "substeps": int(self.substeps),
                "control_hz": float(self.control_hz),
                "newton_collision_pipeline": self.collision is not None,
                "collide_every": int(self.collide_every),
                "solver": type(self.solver).__name__,
            }
        )

    ScenePipeline.__init__ = pipeline_init

    if watch_every > 0:
        step = ScheduleExecutor.step

        def watched(self):
            step(self)
            n = getattr(self, "_tune_n", 0) + 1
            self._tune_n = n
            if n % watch_every:
                return
            data = getattr(self.solver, "mjw_data", None)
            if data is None:
                return
            c = PROBE["contacts"]
            nacon = int(data.nacon.numpy()[0])
            nefc = int(data.nefc.numpy().max())
            c["peak_nacon"] = max(c["peak_nacon"], nacon)
            c["peak_nefc"] = max(c["peak_nefc"], nefc)
            c["samples"] += 1

        ScheduleExecutor.step = watched


PACE: dict[str, Any] = {"timeout_s": 0.0, "held_steps": 0, "held_s": 0.0, "timed_out": 0}


def pace(timeout_s: float) -> None:
    """Give the single-world graph runner strict turns: no step while the graph thinks.

    The serial runner steps once per loop whatever the graph is doing, so the
    seconds SAM 3 or cuRobo take are paid in control steps, at whatever rate the
    physics runs -- a faster arrangement burns more of the step budget per
    second of thought, and the comparison stops being about physics. This holds
    each loop until the graph has something for the simulation (a request on
    the bus, or a thread blocked in ``wait_for_step``), the graph has finished,
    or ``timeout_s`` passes (so an unknown kind of wait cannot deadlock). It is
    what the fleet's barrier does for a wave (``robosimstudio_policy_agent.fleet.barrier``),
    and only this process is patched.
    """
    from robosimstudio.core.bus import SimBus
    from robosimstudio_policy_agent.graph_solution import SolutionGraph

    PACE["timeout_s"] = float(timeout_s)
    wait_for_step = SimBus.wait_for_step

    def waiting(self, *, after, timeout=None):
        self._tune_waiters = getattr(self, "_tune_waiters", 0) + 1
        try:
            return wait_for_step(self, after=after, timeout=timeout)
        finally:
            self._tune_waiters -= 1

    SimBus.wait_for_step = waiting
    get_action = SolutionGraph.get_action

    def paced(self, obs):
        bus = self.bus
        t0 = time.perf_counter()
        deadline = t0 + PACE["timeout_s"]
        while (
            not self.finished
            and bus._queue.empty()
            and not getattr(bus, "_tune_waiters", 0)
            and not bus._closed.is_set()
            and time.perf_counter() < deadline
        ):
            time.sleep(0.0005)
        held = time.perf_counter() - t0
        if held > 0.002:
            PACE["held_steps"] += 1
            PACE["held_s"] += held
        if time.perf_counter() >= deadline:
            PACE["timed_out"] += 1
        return get_action(self, obs)

    SolutionGraph.get_action = paced


def solver_options(solver) -> dict[str, Any]:
    """What the compiled MuJoCo model carries, read back rather than assumed."""
    import mujoco

    o = solver.mj_model.opt
    d = solver.mjw_data
    return {
        "iterations": int(o.iterations),
        "ls_iterations": int(o.ls_iterations),
        "cone": ["pyramidal", "elliptic"][int(o.cone)],
        "impratio": float(o.impratio),
        "tolerance": float(o.tolerance),
        "ccd_iterations": int(o.ccd_iterations),
        "solver": {0: "pgs", 1: "cg", 2: "newton"}.get(int(o.solver), int(o.solver)),
        "jacobian": {0: "dense", 1: "sparse", 2: "auto"}.get(int(o.jacobian), int(o.jacobian)),
        "multiccd": not bool(int(o.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)),
        "timestep": float(o.timestep),
        "naconmax": int(d.naconmax),
        "njmax": int(d.njmax),
        "nworld": int(d.nworld),
    }


# --------------------------------------------------------------------------- one probe, in process
def cmd_run(args, rest: list[str]) -> int:
    """Register the candidate variant, then ``python -m vabar run`` it in this process."""
    _install()
    instrument(args.watch_contacts)
    if args.paced > 0:
        pace(args.paced)
    task = derive(args.variant, args.cand, args.rig)
    from vabar.run import main as vabar_run

    argv = ["--task", task, *rest]
    t0 = time.perf_counter()
    code = vabar_run(argv)
    out = Path(_flag(rest, "--out") or "outputs") / (_flag(rest, "--name") or "")
    if out.is_dir():
        (out / "tune_probe.json").write_text(
            json.dumps(
                {
                    "cand": args.cand,
                    "rig": args.rig,
                    "variant": args.variant,
                    "tag": args.tag,
                    "task": task,
                    "argv": argv,
                    "arrangement": describe(arrangement(args.cand)),
                    "cameras": [[c.name, c.width, c.height] for c in camera_rig(args.rig)],
                    "process_wall_s": round(time.perf_counter() - t0, 2),
                    "pace": dict(PACE),
                    **PROBE,
                },
                indent=1,
                default=str,
            )
        )
    return code


def _flag(argv: list[str], flag: str) -> str:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return ""


def cmd_spin(args) -> int:
    """``rigs/gear_bench.py spin`` on the candidate arrangement, as numbers in a file."""
    _install()
    instrument(1)
    import numpy as np
    from robosimstudio_bench.taskbuild.discovery import task_module

    gb = task_module(SUITE, "rigs.gear_bench")
    arr = arrangement(args.cand)
    teeth = tuple(int(v) for v in args.teeth.split(","))
    seated = tuple(i != args.drop for i in range(len(teeth))) if args.drop >= 0 else None
    t0 = time.perf_counter()
    rig = gb.build(teeth=teeth, module=args.module, pinion=args.pinion, physics=arr, seated=seated)
    build_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    run = gb.press_and_run(rig, steps=args.steps)
    run_s = time.perf_counter() - t1
    hist = run["history"]
    gears = []
    for i, (name, on) in enumerate(zip(rig["names"], rig["keep"], strict=True)):
        ideal = rig["ideal"][i]
        mean, sd = gb._tail(hist, f"gear{i}_spin")
        seat, _ = gb._tail(hist, f"gear{i}_seated")
        floor = gb._floor(hist, f"gear{i}_spin")
        gears.append(
            {
                "name": name,
                "teeth": teeth[i],
                "on": bool(on),
                "ideal": ideal,
                "mean": mean,
                "sd": sd,
                "ratio": mean / ideal if ideal else 0.0,
                "p5_ratio": floor / ideal if ideal else 0.0,
                "seated": seat,
            }
        )
    pinion, pinion_sd = gb._tail(hist, "pinion_speed")
    final = run["final"]
    on = [g for g in gears if g["on"]]
    ok = (
        final.get("started", 0.0) > 0.0
        and all(g["seated"] > 0.99 for g in on)
        and all(GATE_RATIO[0] <= g["ratio"] <= GATE_RATIO[1] for g in on)
    )
    finite = all(math.isfinite(g["mean"]) for g in gears)
    result = {
        "cand": args.cand,
        "teeth": teeth,
        "drop": args.drop,
        "arrangement": describe(arr),
        "pinion": {"commanded": float(rig["spec"].drive_speed), "mean": pinion, "sd": pinion_sd},
        "started": float(final.get("started", 0.0)),
        "turning_fraction": float(final.get("turning_fraction", 0.0)),
        "press_travel_mm": float(final.get("press_travel", 0.0)) * 1000.0,
        "gears": gears,
        "ratio_min": min((g["ratio"] for g in on), default=float("nan")),
        "ratio_max": max((g["ratio"] for g in on), default=float("nan")),
        "p5_min": min((g["p5_ratio"] for g in on), default=float("nan")),
        "pass": bool(ok and finite),
        "build_s": round(build_s, 2),
        "drive_steps": args.steps + 120,
        "drive_steps_per_s": round((args.steps + 120) / run_s, 1),
        "contacts": dict(PROBE["contacts"]),
        "solver": PROBE["solvers"][-1] if PROBE["solvers"] else {},
        "pipeline": PROBE["pipelines"][-1] if PROBE["pipelines"] else {},
        "nan_free": bool(finite and np.isfinite(pinion)),
    }
    rig["sim"].close()
    text = json.dumps(result, indent=1, default=str)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text)
    print(
        f"[spin] {args.cand} z={teeth}: ratio {result['ratio_min']:.3f}-{result['ratio_max']:.3f} "
        f"p5 {result['p5_min']:.2f} seated {[round(g['seated'], 2) for g in gears]} "
        f"started={result['started']:.0f} peak contacts {result['contacts']['peak_nacon']} "
        f"rows {result['contacts']['peak_nefc']} -> {'PASS' if result['pass'] else 'FAIL'}"
    )
    return 0 if result["pass"] or args.drop >= 0 else 1


# --------------------------------------------------------------------------- the sweep
def _python() -> str:
    return sys.executable


def _launch(jobs: list[tuple[str, list[str], Path]], parallel: int) -> None:
    """Run ``(label, argv, log)`` jobs, ``parallel`` at a time, in fresh interpreters."""
    env = dict(os.environ)
    _skills_checkout()
    env.update({k: os.environ[k] for k in ("VABAR_OPEN_ROBOT_SKILLS",) if k in os.environ})
    pending = list(jobs)
    running: list[tuple[str, subprocess.Popen, float]] = []
    while pending or running:
        while pending and len(running) < max(1, parallel):
            label, argv, log = pending.pop(0)
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log, "w")  # noqa: SIM115 -- closed with the process
            proc = subprocess.Popen(argv, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running.append((label, proc, time.perf_counter()))
            print(f"[sweep] start {label}", flush=True)
        for item in list(running):
            label, proc, t0 = item
            if proc.poll() is not None:
                running.remove(item)
                print(f"[sweep] done  {label} rc={proc.returncode} ({time.perf_counter() - t0:.0f}s)", flush=True)
        time.sleep(1.0)


def _cands(text: str) -> list[str]:
    return [c.strip() for c in text.split(",") if c.strip()]


def cmd_speed(args) -> int:
    """Idle steps/s. With ``--renderer rtx`` the cameras are path traced every step, which prices a rig."""
    out = Path(args.out) / ("speed" if not args.renderer else f"speed_{args.renderer}")
    jobs = []
    for cand in _cands(args.cands):
        for rig in _cands(args.rigs):
            for variant in args.variants.split(","):
                for rep in range(args.repeats):
                    name = f"{slug(cand)}__{slug(rig)}__{variant}" + (f"__r{rep}" if args.repeats > 1 else "")
                    if (out / name / "run_meta.json").exists() and not args.force:
                        continue
                    argv = [_python(), __file__, "run", "--cand", cand, "--rig", rig, "--variant", variant, "--",
                            "--solve", "idle", "--max-steps", str(args.steps), "--out", str(out), "--name", name]  # fmt: skip
                    if args.renderer:
                        argv += ["--renderer", args.renderer]
                    jobs.append((name, argv, out / f"{name}.log"))
    _launch(jobs, args.parallel)
    return cmd_table(args)


def cmd_gate(args) -> int:
    out = Path(args.out) / "gate"
    jobs = []
    for cand in _cands(args.cands):
        for teeth in args.teeth_sets.split(";"):
            name = f"{slug(cand)}__z{teeth.replace(',', '-')}"
            if (out / f"{name}.json").exists() and not args.force:
                continue
            argv = [_python(), __file__, "spin", "--cand", cand, "--teeth", teeth, "--json", str(out / f"{name}.json")]
            jobs.append((name, argv, out / f"{name}.log"))
    _launch(jobs, args.parallel)
    return cmd_table(args)


def stage_graph(out: Path, observation: str) -> str:
    """The graph a probe runs: the shipped one, or a copy that reads on demand."""
    if observation == "per-step":
        return GRAPH
    src = REPO / "tasks" / SUITE / "graphs" / GRAPH
    dst = out / "_graphs" / f"{GRAPH}_on_demand"
    # Copied once and left alone: a graph that is not re-staged reads its
    # scripts from here for as long as an earlier sweep's runs are going.
    if not dst.exists():
        shutil.copytree(src, dst)
        doc = json.loads((dst / "workflow.json").read_text())
        doc["meta"]["observation_mode"] = "on_demand"
        (dst / "workflow.json").write_text(json.dumps(doc, indent=2))
    return str(dst)


def cmd_graph(args) -> int:
    out = Path(args.out) / (
        ("graph" if args.observation == "on-demand" else "graph_per_step") + ("_paced" if args.paced > 0 else "")
    )
    out.mkdir(parents=True, exist_ok=True)
    graph = stage_graph(out, args.observation)
    jobs = []
    for cand in _cands(args.cands):
        for rig in _cands(args.rigs):
            for variant in args.variants.split(","):
                name = f"{slug(cand)}__{slug(rig)}__{variant}" + (f"__{args.tag}" if args.tag else "")
                if (out / name / "run_meta.json").exists() and not args.force:
                    continue
                argv = [_python(), __file__, "run", "--cand", cand, "--rig", rig, "--variant", variant, "--tag", args.tag,
                        "--watch-contacts", "1", "--", *CELL_FLAGS, "--graph", graph, "--renderer", "rtx",
                        "--episodes", "0", "--out", str(out), "--name", name]  # fmt: skip
                if args.num_worlds > 1:
                    argv += ["--num-worlds", str(args.num_worlds), "--episodes", f"0:{args.num_worlds}"]
                if args.max_steps:
                    argv += ["--max-steps", str(args.max_steps)]
                if args.paced > 0:
                    argv[argv.index("--") : argv.index("--")] = ["--paced", str(args.paced)]
                jobs.append((name, argv, out / f"{name}.log"))
    _launch(jobs, args.parallel)
    return cmd_table(args)


# --------------------------------------------------------------------------- the table
def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def collect(root: Path) -> dict[str, Any]:
    """Every probe under ``root``, by candidate."""
    table: dict[str, dict[str, Any]] = {}

    def row(cand: str) -> dict[str, Any]:
        return table.setdefault(cand, {"speed": {}, "gate": {}, "graph": {}})

    for meta in sorted(root.glob("speed*/*/run_meta.json")):
        probe = _load(meta.parent / "tune_probe.json")
        whole = _load(meta)
        prov = whole.get("provenance", {})
        cand, rig, variant = probe.get("cand"), probe.get("rig"), probe.get("variant")
        if not cand:
            continue
        build = (prov.get("builds") or [{}])[0]
        key = (
            variant
            if rig == "views" and not whole.get("renderer")
            else f"{variant}|{rig}|{whole.get('renderer') or 'none'}"
        )
        row(cand)["speed"].setdefault(key, []).append(
            {
                "steps_per_s": prov.get("steps_per_s"),
                "build_s": build.get("build_s"),
                "substeps": build.get("substeps"),
                "captured": build.get("captured"),
                "solver": (probe.get("solvers") or [{}])[-1],
                "pipeline": (probe.get("pipelines") or [{}])[-1],
                "arrangement": probe.get("arrangement"),
            }
        )
    for path in sorted((root / "gate").glob("*.json")):
        g = _load(path)
        if g:
            row(g["cand"])["gate"]["z" + "-".join(str(t) for t in g["teeth"])] = g
    for sub in ("graph", "graph_per_step", "graph_paced", "graph_per_step_paced"):
        for meta in sorted((root / sub).glob("*/run_meta.json")):
            probe = _load(meta.parent / "tune_probe.json")
            prov = _load(meta).get("provenance", {})
            rows = [
                json.loads(line) for line in (meta.parent / "episode_results.jsonl").read_text().splitlines() if line
            ]
            cand, rig, variant = probe.get("cand"), probe.get("rig"), probe.get("variant")
            tag = f"|{probe['tag']}" if probe.get("tag") else ""
            row(cand)["graph"].setdefault(f"{rig}|{sub}{tag}", {})[variant] = {
                "success": [bool(r.get("success")) for r in rows],
                "steps": [r.get("episode_length") for r in rows],
                "reward": [r.get("reward") for r in rows],
                "close": [r.get("close_kind") for r in rows],
                "error": [str(r.get("graph_error", ""))[:160] for r in rows],
                "wall_s": prov.get("wall_s"),
                "steps_per_s": prov.get("steps_per_s"),
                "process_wall_s": probe.get("process_wall_s"),
                "peak_nacon": probe.get("contacts", {}).get("peak_nacon"),
                "peak_nefc": probe.get("contacts", {}).get("peak_nefc"),
                "build_s": ((prov.get("builds") or [{}])[0]).get("build_s"),
            }
    return table


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else float("nan")


def cmd_table(args) -> int:
    root = Path(args.out)
    table = collect(root)
    (root / "summary.json").write_text(json.dumps(table, indent=1, default=str))
    ref_speed = {v: _mean([r["steps_per_s"] for r in rs]) for v, rs in table.get("ref", {}).get("speed", {}).items()}
    print(
        "| candidate | steps/s L1 | steps/s L2 | x ref | build_s | spin ratio (z16-20-24) | p5 "
        "| peak contacts/rows (z16-20-24-20) | boards passed (worst ratio) | graph |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for cand, r in table.items():
        s = r["speed"]
        l1 = _mean([x["steps_per_s"] for x in s.get(SPEED_VARIANTS[0], [])])
        l2 = _mean([x["steps_per_s"] for x in s.get(SPEED_VARIANTS[1], [])])
        build = _mean([x["build_s"] for rs in s.values() for x in rs])
        gain = _mean(
            [l1 / ref_speed.get(SPEED_VARIANTS[0], float("nan")), l2 / ref_speed.get(SPEED_VARIANTS[1], float("nan"))]
        )
        gate = r["gate"].get("z16-20-24", {})
        gate4 = r["gate"].get("z16-20-24-20", {})
        ratio = f"{gate['ratio_min']:.3f}-{gate['ratio_max']:.3f} {'pass' if gate['pass'] else 'FAIL'}" if gate else ""
        p5 = f"{gate['p5_min']:.2f}" if gate else ""
        peaks = (
            f"{gate4.get('contacts', {}).get('peak_nacon', '')}/{gate4.get('contacts', {}).get('peak_nefc', '')}"
            if gate4
            else ""
        )
        boards = list(r["gate"].values())
        passed = (
            f"{sum(b['pass'] for b in boards)}/{len(boards)} ({min(b['ratio_min'] for b in boards):.3f})"
            if boards
            else ""
        )
        graphs = []
        for key, by_variant in r["graph"].items():
            ok = sum(sum(v["success"]) for v in by_variant.values())
            n = sum(len(v["success"]) for v in by_variant.values())
            graphs.append(f"{key}: {ok}/{n}")
        print(
            f"| {cand} | {l1:.1f} | {l2:.1f} | {gain:.2f} | {build:.1f} | {ratio} | {p5} | {peaks} | {passed} "
            f"| {'; '.join(graphs)} |"
        )
    for cand, r in table.items():
        for key, rs in sorted(r["speed"].items()):
            if "|" in key:
                print(
                    f"  rendered {cand} {key}: {_mean([x['steps_per_s'] for x in rs]):.1f} steps/s over {len(rs)} run(s)"
                )
    for cand, r in table.items():
        for key, by_variant in r["graph"].items():
            for variant, v in sorted(by_variant.items()):
                print(
                    f"  graph {cand} {key} {variant}: success={v['success']} steps={v['steps']} "
                    f"wall={v['wall_s']}s ({v['steps_per_s']}/s) peak contacts {v['peak_nacon']} rows {v['peak_nefc']}"
                    + (f" err={v['error']}" if any(v["error"]) else "")
                )
    return 0


def cmd_knobs(args) -> int:
    del args
    print(__doc__.split("**What each probe")[0].split("**Candidate names.**")[1].strip())
    for k, v in KNOB_DOC.items():
        print(f"  {k:8s} {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1 :]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="one candidate variant through `python -m vabar run`; bench_run flags after --")
    p.add_argument("--cand", default="ref")
    p.add_argument("--rig", default="views")
    p.add_argument("--variant", required=True)
    p.add_argument("--watch-contacts", type=int, default=0, help="read nacon/nefc every N control steps (0: off)")
    p.add_argument("--tag", default="", help="a repeat label, kept in tune_probe.json")
    p.add_argument(
        "--paced", type=float, default=0.0, help="strict turns: hold each step up to this many s while the graph thinks"
    )
    p.set_defaults(fn=lambda a: cmd_run(a, rest))

    p = sub.add_parser("spin", help="the gear-train gate on a candidate arrangement")
    p.add_argument("--cand", default="ref")
    p.add_argument("--teeth", default="16,20,24")
    p.add_argument("--pinion", type=int, default=14)
    p.add_argument("--module", type=float, default=0.0025)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--drop", type=int, default=-1, help="leave this station empty (the negative control)")
    p.add_argument("--json", default="")
    p.set_defaults(fn=cmd_spin)

    for name, fn in (("speed", cmd_speed), ("gate", cmd_gate), ("graph", cmd_graph), ("table", cmd_table)):
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(REPO / "outputs" / "fast_tune" / SUITE))
        p.add_argument("--cands", default="ref")
        p.add_argument("--parallel", type=int, default=1)
        p.add_argument("--force", action="store_true")
        if name == "speed":
            p.add_argument("--variants", default=",".join(SPEED_VARIANTS))
            p.add_argument("--steps", type=int, default=600)
            p.add_argument("--repeats", type=int, default=1)
            p.add_argument("--rigs", default="views")
            p.add_argument("--renderer", default="", help="empty (the physics probe) or rtx (prices the rig)")
        if name == "gate":
            p.add_argument("--teeth-sets", default="16,20,24;16,20,24,20", help="';'-separated boards")
        if name == "graph":
            p.add_argument("--variants", default=",".join(GRAPH_VARIANTS))
            p.add_argument("--rigs", default="views")
            p.add_argument("--observation", choices=("on-demand", "per-step"), default="on-demand")
            p.add_argument("--num-worlds", type=int, default=1, help="the fleet trick; refused for this suite")
            p.add_argument("--tag", default="", help="a repeat label: run names get __<tag>")
            p.add_argument("--max-steps", type=int, default=0, help="0 keeps the task budget")
            p.add_argument(
                "--paced", type=float, default=0.0, help="strict turns (see pace()); 0 is the shipped runner"
            )
        p.set_defaults(fn=fn)

    sub.add_parser("knobs").set_defaults(fn=cmd_knobs)
    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
