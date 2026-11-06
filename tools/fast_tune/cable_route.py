# SPDX-License-Identifier: Apache-2.0
"""The cable_route ``fast`` sweep: SuperDex vs reduced VBD, by measurement.

Re-runnable. Every candidate is a string of knobs, an arrangement is built from
it, registered under a NEW name (``cable_fast_tune/<knobs>@v0``) and swapped
into every ``cable_route`` variant in-process with ``dataclasses.replace`` +
``SUITES.replace`` -- so no shipped name, document or arrangement changes, and
``run_meta.json`` (builds[].substeps, provenance) records what actually ran.

Candidates::

    ref                                  cable_coupled_960hz@v1, untouched
    name:cable_route_fast@v1             a registered arrangement, by name (the winner)
    vbd:substeps=8,passes=1,iters=20     reduced VBD (any subset of the knobs)
        substeps  arrangement substeps (16 = 960 Hz)
        passes    proxy coupler relaxation passes (2 in the reference)
        iters     VBD iterations (40 in the reference)
        pitch     capsule count scale on the rod (1.0; 0.5 halves capsules/m)
        mj_iters  MuJoCo iterations (100) ; mj_ls  MuJoCo line-search iterations (20)
    sdx:rod=2,fb=off,arm=16              SuperDex rod + MuJoCo arms (needs superdex-physics;
                                         the rod is cable_superdex.EntityCableSuperdex)
        rod       SuperDex engine steps per control step (2)
        fb        feedback "off" (one-way) or "auto" (two-way)
        arm       arrangement substeps for the MuJoCo arm island (16)
        mj_iters, mj_ls   MuJoCo iterations (100 / 20)
        threads   SuperDex worker threads (8)
        mjc       1 = MuJoCo's own contacts for arm-vs-scenery (0 = Newton's pipeline, as the reference)
        ff, mu, pad, pen, bend   the rod's friction_falloff, mu_cable, contact_gap, contact_penalty,
                  and a scale on bend_ke (SuperDex reads these off the RodSpec)
        mff, msmooth  mirror_friction_falloff / mirror_smoothing on the SuperDex island
        ls, nit   the engine's linear solver by name (cg, ldlt, ...) and Newton iterations per step

Usage (from the worktree root; GPUs 3 and 7 only)::

    # one run, in-process: install the candidate, then `vabar run` with the rest
    python tools/fast_tune/cable_route.py run 'sdx:rod=2' -- --task cable_route/port_easy \\
        --episodes 10009 --solve idle --max-steps 400 --out OUT --name NAME

    # a probe stage over several candidates, sequentially on one GPU (background-friendly)
    python tools/fast_tune/cable_route.py probe --gpu 3 --stage speed --out OUT 'sdx:rod=2' 'sdx:rod=4'
        stages: speed   idle 400 steps on port_easy ep 10009, no renderer
                weave3  --solve routine on cable_route/weave3
                port    --solve routine on cable_route/port_easy, 3 L1 instances (10009,10062,10084)
                graph   the cable_routing/L1 graph protocol on 2 instances (10009,10062)
        --episodes a,b,c overrides the port/graph instances; --parallel starts the runs at once
        (fine for routine/graph fidelity, never for speed); a finished run directory is skipped.

    The sweep as run for docs/validation/fast_cable_route.md (OUT under agent_scratch/d1)::

        probe --gpu 3 --stage speed  --out OUT/speed ref 'vbd:substeps=8,passes=1' 'vbd:passes=1' ...
        probe --gpu 7 --stage weave3 --parallel --out OUT/fid ref 'vbd:substeps=8,passes=1' 'sdx:rod=2' ...
        probe --gpu 7 --stage port   --parallel --episodes 10009,10062,10084,10043,10053 --out OUT/fid ...
        probe --gpu 3 --stage graph  --parallel --out OUT/graph ref 'vbd:substeps=8,passes=1' ...

    # the table, from every run directory under OUT
    python tools/fast_tune/cable_route.py table --out OUT
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
# `vabar` is an editable install of whichever checkout ran `pip install -e`; the
# checkout this file sits in is the one whose tasks a sweep must measure.
if str(WORKTREE) not in sys.path[:1]:
    sys.path.insert(0, str(WORKTREE))

REFERENCE = "cable_coupled_960hz@v1"
SPEED_EPISODE = 10009
PORT_EPISODES = (10009, 10062, 10084)
GRAPH_EPISODES = (10009, 10062)
GRAPH_FLAGS = (
    "--solve", "graph", "--graph", "gt", "--no-privileged", "--perception", "vision",
    "--renderer", "rtx", "--bus-timeout", "600", "--settle-steps", "60",
)  # fmt: skip


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def parse(candidate: str) -> tuple[str, dict[str, str]]:
    """``"vbd:substeps=8,iters=20"`` -> ``("vbd", {"substeps": "8", "iters": "20"})``."""
    family, _, rest = candidate.partition(":")
    knobs = {}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        knobs[key.strip()] = value.strip()
    return family.strip(), knobs


def slug(candidate: str) -> str:
    """A filesystem- and registry-safe spelling of a candidate."""
    return re.sub(r"[^A-Za-z0-9]+", "_", candidate).strip("_") or "ref"


def arrangement_for(candidate: str):
    """The :class:`ArrangementSpec` a candidate stands for (``None`` for ``ref``)."""
    from robosimstudio.specs import CollisionSpec, backend, coupler
    from robosimstudio_bench.taskbuild.arrangements import get_arrangement

    family, knobs = parse(candidate)
    ref = get_arrangement(REFERENCE)
    if family == "ref":
        return None
    if family == "name":
        # A registered arrangement, by its name: `name:cable_route_fast@v1`.
        return get_arrangement(candidate.partition(":")[2])
    table = dict(ref.backends)
    mj = table["articulated"]
    mj_changes = {}
    if "mj_iters" in knobs:
        mj_changes["iterations"] = int(knobs["mj_iters"])
    if "mj_ls" in knobs:
        mj_changes["ls_iterations"] = int(knobs["mj_ls"])
    if family == "vbd":
        vbd = table["rod"]
        if "iters" in knobs:
            vbd = vbd.with_(iterations=int(knobs["iters"]))
        coupling = ref.coupling
        if "passes" in knobs:
            coupling = coupling.with_(iterations=int(knobs["passes"]))
        if "collide" in knobs:
            coupling = coupling.with_(collide_interval=int(knobs["collide"]))
        return dataclasses.replace(
            ref,
            substeps=int(knobs.get("substeps", ref.substeps)),
            backends=(("articulated", mj.with_(**mj_changes)), ("rod", vbd)),
            coupling=coupling,
        )
    if family == "sdx":
        if knobs.get("mjc", "0") == "1":
            mj_changes["use_mujoco_contacts"] = True
        extra = {}
        if "mff" in knobs:
            extra["mirror_friction_falloff"] = float(knobs["mff"])
        if "msmooth" in knobs:
            extra["mirror_smoothing"] = float(knobs["msmooth"])
        if "ls" in knobs:
            extra["linear_solver"] = str(knobs["ls"])
        if "nit" in knobs:
            extra["newton_iterations"] = int(knobs["nit"])
        rod = backend(
            "superdex",
            substeps=int(knobs.get("rod", 2)),
            feedback=str(knobs.get("fb", "off")),
            threads=int(knobs.get("threads", 8)),
            **extra,
        )
        return dataclasses.replace(
            ref,
            substeps=int(knobs.get("arm", ref.substeps)),
            backends=(("articulated", mj.with_(**mj_changes)), ("rod", rod)),
            # One Newton island: nothing to couple. Stated for the record.
            coupling=coupler("proxy"),
            collision=CollisionSpec(rigid_gap=0.002, contact_matching="latest"),
        )
    raise ValueError(f"unknown candidate family {family!r} in {candidate!r}")


def install(candidate: str) -> str:
    """Register the candidate's arrangement and swap it into every cable_route variant. Returns its name."""
    import vabar

    vabar.install()
    from robosimstudio_bench.taskbuild.arrangements import register_arrangement
    from robosimstudio_bench.taskbuild.registry import SUITES, get_suite

    family, knobs = parse(candidate)
    suite = get_suite("cable_route")
    if family == "ref":
        return REFERENCE
    if family == "sdx":
        from robosimstudio.scene.registry import register_entity

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from cable_superdex import EntityCableSuperdex

        register_entity("cable_rod", engine="superdex")(EntityCableSuperdex)
    name = f"cable_fast_tune/{slug(candidate)}@v0"
    spec = register_arrangement(name, arrangement_for(candidate))
    scale = float(knobs.get("pitch", 1.0))

    def rebuilt(variant):
        entities = []
        for entity in variant.entities:
            if type(entity).__name__ == "RodSpec":
                if scale != 1.0:
                    # Fewer, longer capsules: a joint's bend gain is EI over the pitch it
                    # spans, so it scales with the count to keep the same rod
                    # (cable.weave_bend's own rule).
                    segments = max(4, 2 * round(0.5 * entity.segments * scale))
                    ratio = segments / entity.segments
                    entity = dataclasses.replace(
                        entity, segments=segments, bend_ke=entity.bend_ke * ratio, bend_kd=entity.bend_kd * ratio
                    )
                if family == "sdx":
                    # The library's "rod" on SuperDex is the bare centerline; "cable_rod" on
                    # SuperDex is this suite's adapter (spawn shape, stiffness, reader).
                    rod_changes = {"entity": "cable_rod"}
                    if "ff" in knobs:  # friction_falloff [m/s]: below it there is no static friction
                        rod_changes["friction_falloff"] = float(knobs["ff"])
                    if "mu" in knobs:  # mu_cable; the engine is handed sqrt(mu)
                        rod_changes["mu_cable"] = float(knobs["mu"])
                    if "pad" in knobs:  # contact_gap -> the rod's penalty_threshold_extra_padding
                        rod_changes["contact_gap"] = float(knobs["pad"])
                    if "pen" in knobs:  # contact_penalty [Pa/m]
                        rod_changes["contact_penalty"] = float(knobs["pen"])
                    if "scenefr" in knobs:  # 0 = leave every shape's friction as built (1.0)
                        rod_changes["scene_friction"] = knobs["scenefr"] == "1"
                    if "bend" in knobs:  # scale on the bend stiffness handed to the engine
                        rod_changes["bend_ke"] = entity.bend_ke * float(knobs["bend"])
                    entity = dataclasses.replace(entity, **rod_changes)
            entities.append(entity)
        return dataclasses.replace(variant, physics=spec, entities=tuple(entities))

    SUITES.replace(
        "cable_route",
        dataclasses.replace(suite, variants=tuple(rebuilt(v) for v in suite.variants)),
        origin="tools/fast_tune/cable_route.py",
    )
    return name


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    name = install(args.candidate)
    print(f"[fast_tune] candidate {args.candidate!r} -> physics {name}", flush=True)
    from vabar.cli import main

    rest = [a for a in args.rest if a != "--"]
    out = None
    if "--out" in rest and "--name" in rest:
        out = Path(rest[rest.index("--out") + 1]) / rest[rest.index("--name") + 1]
    code = main(["run", *rest])
    if out is not None and out.exists():
        (out / "fast_tune.json").write_text(json.dumps({"candidate": args.candidate, "physics": name}, indent=2))
    return int(code or 0)


def stage_jobs(stage: str, episodes: tuple[int, ...] = ()) -> list[tuple[str, list[str], dict[str, str]]]:
    """``(run-name suffix, vabar run args, extra env)`` per run in a stage. ``episodes`` overrides the stage's own."""
    if stage == "speed":
        args = ["--task", "cable_route/port_easy", "--episodes", str(SPEED_EPISODE), "--solve", "idle"]
        return [("speed", [*args, "--max-steps", "400"], {})]
    if stage == "weave3":
        return [("weave3", ["--task", "cable_route/weave3", "--solve", "routine"], {})]
    if stage == "port":
        return [
            (f"port_{ep}", ["--task", "cable_route/port_easy", "--episodes", str(ep), "--solve", "routine"], {})
            for ep in episodes or PORT_EPISODES
        ]
    if stage == "graph":
        env = {"ROBOSIMSTUDIO_SEGMENTATION_BACKEND": "palette"}
        return [
            (f"graph_{ep}", ["--task", "cable_route/port_easy", "--episodes", str(ep), *GRAPH_FLAGS], env)
            for ep in episodes or GRAPH_EPISODES
        ]
    raise ValueError(f"unknown stage {stage!r}")


def cmd_probe(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[str, subprocess.Popen]] = []
    episodes = tuple(int(e) for e in args.episodes.split(",")) if args.episodes else ()
    for candidate in args.candidates:
        for suffix, run_args, extra in stage_jobs(args.stage, episodes):
            name = f"{slug(candidate)}__{suffix}"
            if (out / name / "run_meta.json").exists() and not args.force:
                print(f"[probe] {name}: done already", flush=True)
                continue
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), **extra}
            env.setdefault("OMP_NUM_THREADS", "8")
            cmd = [PYTHON, str(Path(__file__).resolve()), "run", candidate, "--", *run_args]
            cmd += ["--out", str(out), "--name", name]
            log = open(logs / f"{name}.log", "w")  # noqa: SIM115 -- held by the child
            print(f"[probe] {name}: start on GPU {args.gpu}", flush=True)
            proc = subprocess.Popen(cmd, cwd=WORKTREE, env=env, stdout=log, stderr=subprocess.STDOUT)
            procs.append((name, proc))
            if not args.parallel:
                proc.wait()
                print(f"[probe] {name}: exit {proc.returncode}", flush=True)
                if proc.returncode and args.stop_on_failure:
                    return proc.returncode
    for name, proc in procs:
        if proc.returncode is None:
            proc.wait()
            print(f"[probe] {name}: exit {proc.returncode}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def summarize(run: Path) -> dict:
    """What one run directory says: speed, build, outcome, goal metrics."""
    meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").exists() else {}
    rows = []
    if (run / "episode_results.jsonl").exists():
        rows = [json.loads(line) for line in (run / "episode_results.jsonl").read_text().splitlines() if line.strip()]
    tag = json.loads((run / "fast_tune.json").read_text()) if (run / "fast_tune.json").exists() else {}
    prov = meta.get("provenance", {})
    builds = prov.get("builds", meta.get("builds", []))
    return {
        "run": run.name,
        "candidate": tag.get("candidate", ""),
        "physics": tag.get("physics", ""),
        "steps_per_s": prov.get("steps_per_s"),
        "build_s": [b.get("build_s") for b in builds],
        "substeps": [b.get("substeps") for b in builds],
        "captured": [b.get("captured") for b in builds],
        "wall_s": prov.get("wall_s"),
        "episodes": [
            {
                k: r.get(k)
                for k in ("episode_id", "success", "reward", "episode_length", "close_kind", "termination_reason")
                if k in r
            }
            | {
                "metrics": {
                    k: v
                    for k, v in (r.get("metrics") or {}).items()
                    if not k.startswith("goal.") or k.endswith(".fraction")
                }
            }
            for r in rows
        ],
    }


def cmd_table(args: argparse.Namespace) -> int:
    out = Path(args.out)
    runs = sorted(p for p in out.iterdir() if (p / "run_meta.json").exists())
    table = [summarize(p) for p in runs]
    if args.json:
        print(json.dumps(table, indent=1, default=str))
        return 0
    for row in table:
        eps = "; ".join(
            f"ep{e.get('episode_id')}: succ={e.get('success')} r={e.get('reward'):.3f} len={e.get('episode_length')}"
            f" ({e.get('close_kind')})"
            + "".join(
                f" {k.split('/')[-1].replace('.fraction', '')}={v:.4g}" if isinstance(v, float) else ""
                for k, v in e["metrics"].items()
                if k in ("distance", "straightness") or k.endswith(".fraction")
            )
            for e in row["episodes"]
        )
        sps = row["steps_per_s"]
        speed = f"{sps:.2f}" if isinstance(sps, (int, float)) else str(sps)
        print(f"{row['run']:<48} {speed:>7} steps/s  build_s={row['build_s']} substeps={row['substeps']}  {eps}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="install a candidate, then `vabar run` the rest in-process")
    p.add_argument("candidate")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("probe", help="run one probe stage over candidates")
    p.add_argument("candidates", nargs="+")
    p.add_argument("--stage", required=True, choices=("speed", "weave3", "port", "graph"))
    p.add_argument("--gpu", required=True, choices=("3", "7"))
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default="", help="comma-separated episode ids for the port/graph stages")
    p.add_argument("--parallel", action="store_true", help="start every run at once instead of one after another")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    p.set_defaults(func=cmd_probe)
    p = sub.add_parser("table", help="summarise every run directory under --out")
    p.add_argument("--out", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_table)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
