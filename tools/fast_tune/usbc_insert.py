# SPDX-License-Identifier: Apache-2.0
"""Re-runnable speed/fidelity sweep for the ``usbc_insert`` cable-insertion cells.

Every candidate is the shipped document (``insert_easy`` / ``insert_medium``)
with ONE knob changed (or a combination of survivors), derived in this process
from the lowered :class:`~robosimstudio.specs.TaskSpec` with
``dataclasses.replace`` and added to the suite under a new variant name
(``<base>__<candidate>``) with ``SUITES.replace``. Nothing shipped is renamed or
edited: the reference documents keep their names and their physics.

Three probes, cheapest first (docs/validation/fast_usbc_insert.md has the table
this produced):

``speed``  idle for 600 steps, no renderer: steps/s, build_s, per-episode reset_s.
``gt``     the privileged ``gt`` graph, no renderer, on the first five L1
           (insert_easy) and first three L2 (insert_medium) cell instances.
``mate``   the cell protocol (palette segmentation, vision ``mate`` graph,
           ``--no-privileged``, RTX, bus-timeout 600, settle 600).

Usage (from the VABAR checkout, GPU picked by CUDA_VISIBLE_DEVICES)::

    # one run, any `vabar run` flags after `--`
    python tools/fast_tune/usbc_insert.py run --cand nocable --base insert_easy -- \\
        --solve idle --episodes 10009 --max-steps 600 --out OUT --name x

    # a probe over a candidate list, N processes at once
    python tools/fast_tune/usbc_insert.py sweep --probe speed --cands ref,nocable,graph --out OUT -j 4
    python tools/fast_tune/usbc_insert.py sweep --probe gt --cands ref,fast --out OUT -j 16

    # the table
    python tools/fast_tune/usbc_insert.py table --out OUT

    # list candidates and the knobs each changes
    python tools/fast_tune/usbc_insert.py list

Each run directory gets a ``fast_tune.json`` sidecar beside ``run_meta.json``:
the candidate's knobs, the physics/entities actually built (read back from the
live environment, which is how an override is verified), every ``Env.reset``'s
wall time and whether it rebuilt, and the peak MuJoCo contact / constraint rows
(``nacon`` / ``nefc``, sampled every 50 steps) against ``nconmax`` / ``njmax``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Run as a script, sys.path[0] is this directory and `import vabar` would take
# whichever checkout the editable install points at. Put this one first, so the
# tasks a sweep measures are the ones beside this file.
sys.path.insert(0, str(REPO))
PYTHON = sys.executable
SUITE = "usbc_insert"
BASES = ("insert_easy", "insert_medium")

L1 = (10009, 10024, 10032, 10031, 10084)
"""First five cable_insertion/L1 instances (benchmark/vabar_v1.json order)."""
L2 = (10085, 10032, 10010)
"""First three cable_insertion/L2 instances."""
SPEED_EPISODE = {"insert_easy": 10009, "insert_medium": 10085}

MATE_FLAGS = [
    "--solve", "graph", "--graph", "mate", "--no-privileged", "--perception", "vision",
    "--renderer", "rtx", "--bus-timeout", "600", "--settle-steps", "600",
]  # fmt: skip
GT_FLAGS = ["--solve", "graph", "--graph", "gt", "--bus-timeout", "600"]
"""--bus-timeout only so a sweep sharing one GPU cannot time a converging tool call out."""

# --------------------------------------------------------------------------- candidates
#
# Knobs (all optional; absent = the shipped document's value):
#   cable        False drops the `cables` entity (plug cable; on medium also the bulkhead tail)
#   cuda_graph   ArrangementSpec.use_cuda_graph
#   voxel        sdf_voxel of the plug and the bulkhead [m]
#   hydro        hydroelastic on the plug and the bulkhead
#   substeps     ArrangementSpec.substeps (even)
#   iterations / ls_iterations / nconmax / njmax   MuJoCo backend params
#   rebuild      False re-poses instead of rebuilding when only object_xforms change
#                (model_key_fields drops object_xforms; the welded port's draw still rebuilds)
#   physics      a registered arrangement name to take wholesale (then the knobs above apply)

CANDIDATES: dict[str, dict] = {
    "ref": {},
    # --- one knob at a time from the reference -------------------------------------
    "nocable": {"cable": False},
    "graph": {"cuda_graph": True},
    "vox05": {"voxel": 5.0e-4},
    "point": {"hydro": False},
    "sub8": {"substeps": 8},
    "it50": {"iterations": 50, "ls_iterations": 20},
    "con1k": {"nconmax": 1024, "njmax": 2048},
    # --- combinations of survivors -------------------------------------------------
    "nc_graph": {"cable": False, "cuda_graph": True},
    "nc_graph_con": {"cable": False, "cuda_graph": True, "nconmax": 512, "njmax": 1024},
    "nc_graph_it": {"cable": False, "cuda_graph": True, "iterations": 50, "ls_iterations": 20},
    "nc_graph_sub8": {"cable": False, "cuda_graph": True, "substeps": 8},
    "nc_graph_sub12": {"cable": False, "cuda_graph": True, "substeps": 12},
    "nc_graph_vox": {"cable": False, "cuda_graph": True, "voxel": 5.0e-4},
    "nc_graph_point": {"cable": False, "cuda_graph": True, "hydro": False},
    "nc_graph_norebuild": {"cable": False, "cuda_graph": True, "rebuild": False},
    "nc_eager": {"cable": False, "cuda_graph": False},
    # The shipped fast documents (registered by tasks/usbc_insert, not derived here):
    # run them with `--cand doc`, which maps insert_easy -> insert_easy_nocable etc.
    "doc": {"document": True},
}

DOCUMENT_OF = {"insert_easy": "insert_easy_nocable", "insert_medium": "insert_medium_nocable"}


def variant_name(base: str, cand: str) -> str:
    if cand == "ref":
        return base
    if CANDIDATES[cand].get("document"):
        return DOCUMENT_OF[base]
    return f"{base}__{cand}"


def derive(task, cand: str, knobs: dict):
    """The shipped variant ``task`` with ``knobs`` applied, as a new TaskSpec."""
    from robosimstudio.specs import ObjectSpec
    from robosimstudio_bench.taskbuild.arrangements import get_arrangement

    physics = get_arrangement(knobs["physics"]) if "physics" in knobs else task.physics
    backend = physics.backend_for("*")
    params = {k: knobs[k] for k in ("iterations", "ls_iterations", "nconmax", "njmax") if k in knobs}
    if params:
        physics = physics.with_backend("*", backend.with_(**params))
    if "substeps" in knobs:
        physics = physics.with_(substeps=int(knobs["substeps"]))
    if "cuda_graph" in knobs:
        physics = physics.with_(use_cuda_graph=bool(knobs["cuda_graph"]))

    entities = []
    for spec in task.entities:
        name = getattr(spec, "name", "")
        if not knobs.get("cable", True) and name == "cables":
            continue
        if isinstance(spec, ObjectSpec) and name in ("plug", "bulkhead"):
            changes = {}
            if "voxel" in knobs:
                changes["sdf_voxel"] = float(knobs["voxel"])
            if "hydro" in knobs:
                changes["hydroelastic"] = bool(knobs["hydro"])
            spec = replace(spec, **changes) if changes else spec
        entities.append(spec)

    episodes = task.episodes
    if knobs.get("rebuild", True) is False:
        episodes = replace(
            episodes, model_key_fields=tuple(f for f in episodes.model_key_fields if f != "object_xforms")
        )
    return replace(
        task, name=variant_name(task.name, cand), physics=physics, entities=tuple(entities), episodes=episodes
    )


def register_candidates(cands: list[str] | None = None) -> list[str]:
    """Add ``<base>__<cand>`` variants to the usbc_insert suite. Returns the new full names."""
    from robosimstudio.specs import TaskSuite
    from robosimstudio_bench.taskbuild.registry import SUITES, get_suite

    suite = get_suite(SUITE)
    have = set(suite.variant_names)
    extra = []
    for cand in cands or list(CANDIDATES):
        knobs = CANDIDATES[cand]
        if cand == "ref" or knobs.get("document"):
            continue
        for base in BASES:
            name = variant_name(base, cand)
            if name not in have:
                extra.append(derive(suite.variant(base), cand, knobs))
                have.add(name)
    if extra:
        grown = TaskSuite(
            name=suite.name,
            variants=(*suite.variants, *extra),
            description=suite.description,
            provenance=suite.provenance,
            home=suite.home,
        )
        SUITES.replace(SUITE, grown, origin=__name__)
    return [f"{SUITE}/{t.name}" for t in extra]


# --------------------------------------------------------------------------- instrumentation


def _find_mjw(pipeline):
    """The MuJoCo-Warp data/solver behind a pipeline, wherever it is nested."""
    seen = set()
    stack = [getattr(pipeline, "solver", None)]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if hasattr(obj, "mjw_data"):
            return obj
        for attr in ("solvers", "entries", "islands", "solver", "inner"):
            sub = getattr(obj, attr, None)
            if isinstance(sub, (list, tuple)):
                stack.extend(getattr(s, "solver", s) for s in sub)
            elif sub is not None:
                stack.append(sub)
    return None


def instrument(sidecar: Path, cand: str) -> None:
    """Time every Env.reset, sample nacon/nefc, and write ``sidecar`` at exit."""
    import atexit

    from robosimstudio_bench.envs.env import Env

    record: dict = {"candidate": cand, "knobs": CANDIDATES.get(cand, {}), "resets": [], "peak": {}}
    orig_reset, orig_step, orig_rebuild = Env.reset, Env.step, Env._rebuild

    def describe(env) -> None:
        task = env.task
        phys = task.physics
        objs = {
            s.name: {"hydroelastic": s.hydroelastic, "sdf_voxel": s.sdf_voxel}
            for s in task.objects
            if s.name in ("plug", "bulkhead")
        }
        record["built"] = {
            "task": task.full_name,
            "entities": [getattr(s, "name", "") for s in task.entities],
            "objects": objs,
            "substeps": phys.substeps,
            "use_cuda_graph": phys.use_cuda_graph,
            "backend": dict(phys.backend_for("*").params),
            "model_key_fields": list(task.episodes.model_key_fields),
            "pipeline_substeps": int(env.pipeline.substeps),
            "body_count": int(env.model.body_count),
            "shape_count": int(env.model.shape_count),
        }
        solver = _find_mjw(env.pipeline)
        if solver is not None:
            record["built"]["naconmax"] = int(solver.mjw_data.naconmax)
            record["built"]["njmax"] = int(solver.mjw_data.njmax)

    def reset(self, episode_id=None):
        t0 = time.perf_counter()
        self._ft_rebuilt = False
        out = orig_reset(self, episode_id)
        record["resets"].append(
            {"episode": episode_id, "reset_s": round(time.perf_counter() - t0, 3), "rebuilt": self._ft_rebuilt}
        )
        record["captured"] = bool(self.pipeline.captured)
        if "built" not in record:
            describe(self)
        self._ft_steps = 0
        flush()
        return out

    def rebuild(self, episode):
        self._ft_rebuilt = True
        return orig_rebuild(self, episode)

    def step(self, action=None, mode=""):
        out = orig_step(self, action, mode)
        n = getattr(self, "_ft_steps", 0) + 1
        self._ft_steps = n
        if n % 50 == 0:
            solver = _find_mjw(self.pipeline)
            if solver is not None:
                peak = record["peak"]
                peak["nacon"] = max(peak.get("nacon", 0), int(solver.mjw_data.nacon.numpy().max()))
                peak["nefc"] = max(peak.get("nefc", 0), int(solver.mjw_data.nefc.numpy().max()))
        return out

    def flush() -> None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(record, indent=1, default=str))

    Env.reset, Env.step, Env._rebuild = reset, step, rebuild
    atexit.register(flush)


# --------------------------------------------------------------------------- commands


def cmd_run(args, passthrough: list[str]) -> int:
    os.chdir(REPO)
    from vabar import install

    install()
    register_candidates([args.cand])
    task = f"{SUITE}/{variant_name(args.base, args.cand)}"
    out = _flag(passthrough, "--out") or "outputs"
    name = _flag(passthrough, "--name") or f"{args.base}__{args.cand}"
    instrument(Path(out) / name / "fast_tune.json", args.cand)
    from vabar.run import main as run_main

    argv = ["--task", task, *passthrough]
    if not _flag(passthrough, "--name"):
        argv += ["--name", name]
    return run_main(argv)


def _flag(argv: list[str], flag: str) -> str:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return ""


def probe_jobs(probe: str, cands: list[str], bases: list[str]) -> list[tuple[str, list[str], dict]]:
    """(run name, argv after `run`, extra env) per process for a probe."""
    jobs = []
    for cand in cands:
        for base in bases:
            if probe == "speed":
                ep = SPEED_EPISODE[base]
                argv = ["--solve", "idle", "--episodes", str(ep), "--max-steps", "600"]
                jobs.append((f"speed__{base}__{cand}", ["--cand", cand, "--base", base, "--", *argv], {}))
                continue
            episodes = L1 if base == "insert_easy" else L2
            if probe == "gt":
                flags, env = GT_FLAGS, {}
            elif probe == "mate":
                flags, env = MATE_FLAGS, {"ROBOSIMSTUDIO_SEGMENTATION_BACKEND": "palette"}
                episodes = episodes[:3]
            else:
                raise SystemExit(f"unknown probe {probe!r}")
            for ep in episodes:
                argv = [*flags, "--episodes", str(ep)]
                jobs.append((f"{probe}__{base}__{cand}__{ep}", ["--cand", cand, "--base", base, "--", *argv], env))
    return jobs


def cmd_sweep(args) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cands = [c for c in args.cands.split(",") if c]
    bases = [b for b in args.bases.split(",") if b]
    jobs = probe_jobs(args.probe, cands, bases)
    if args.repeat > 1:
        jobs = [(f"{n}__r{k}", a, e) for k in range(args.repeat) for n, a, e in jobs]
    pending = [j for j in jobs if args.force or not (out / j[0] / "run_meta.json").is_file()]
    print(f"[sweep] {len(pending)}/{len(jobs)} runs to do, {args.jobs} at a time -> {out}")
    running: list[tuple[str, subprocess.Popen, float]] = []
    while pending or running:
        while pending and len(running) < args.jobs:
            name, argv, extra = pending.pop(0)
            env = {**os.environ, **extra}
            cmd = [PYTHON, str(Path(__file__).resolve()), "run", *argv, "--out", str(out), "--name", name]
            with open(out / f"{name}.log", "w") as log:  # the child keeps its own descriptor
                proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((name, proc, time.time()))
            time.sleep(args.stagger)
        for item in list(running):
            name, proc, t0 = item
            if proc.poll() is not None:
                running.remove(item)
                print(f"[sweep] {name}: exit {proc.returncode} after {(time.time() - t0) / 60:.1f} min", flush=True)
        time.sleep(2)
    return 0


def _row(run: Path) -> dict:
    meta = json.loads((run / "run_meta.json").read_text())
    prov = meta["provenance"]
    rows = [json.loads(line) for line in (run / "episode_results.jsonl").read_text().splitlines() if line.strip()]
    side = json.loads((run / "fast_tune.json").read_text()) if (run / "fast_tune.json").is_file() else {}
    resets = side.get("resets", [])
    ep_reset = sum(r["reset_s"] for r in resets[1:])
    steps = sum(int(r["episode_length"]) for r in rows)
    ep_wall = sum(float(r.get("wall_s", 0.0)) for r in rows)
    m = rows[0].get("metrics", {}) if rows else {}
    return {
        "run": run.name,
        "task": meta["task"],
        "success": [bool(r["success"]) for r in rows],
        "steps": [int(r["episode_length"]) for r in rows],
        "close": [r.get("close_kind") for r in rows],
        "depth_mm": round(1e3 * float(m.get("depth", float("nan"))), 2),
        "lat_mm": round(1e3 * float(m.get("lat", float("nan"))), 2),
        "tilt_deg": round(float(m.get("tilt", float("nan"))) * 57.2958, 2) if "tilt" in m else None,
        "gripped": m.get("gripped"),
        "build_s": prov["builds"][0]["build_s"],
        "captured": prov["cuda_graph_steps"],
        "steps_per_s": prov["steps_per_s"],
        "reset_s": round(ep_reset, 2),
        "stepping_per_s": round(steps / (ep_wall - ep_reset), 1) if ep_wall - ep_reset > 0 else None,
        "wall_s": prov["wall_s"],
        "peak": side.get("peak"),
        "built": side.get("built"),
    }


def cmd_table(args) -> int:
    out = Path(args.out)
    rows = [_row(p) for p in sorted(out.iterdir()) if (p / "run_meta.json").is_file()]
    if args.json:
        print(json.dumps(rows, indent=1, default=str))
        return 0
    for r in rows:
        peak = r["peak"] or {}
        print(
            f"{r['run']:<48} ok={''.join('Y' if s else 'n' for s in r['success'])} steps={r['steps']} "
            f"depth={r['depth_mm']} lat={r['lat_mm']} tilt={r['tilt_deg']} sps={r['steps_per_s']} "
            f"stepping={r['stepping_per_s']} build={r['build_s']} reset={r['reset_s']} "
            f"graph={r['captured'].get('steps_replayed')}/{r['captured'].get('steps_eager')} "
            f"nacon={peak.get('nacon')} nefc={peak.get('nefc')}"
        )
    return 0


def cmd_list(args) -> int:
    del args
    for name, knobs in CANDIDATES.items():
        print(f"{name:<22} {knobs}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1 :]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="one vabar run of one candidate; vabar run flags after --")
    p.add_argument("--cand", default="ref", choices=list(CANDIDATES))
    p.add_argument("--base", default="insert_easy", choices=BASES)
    p.add_argument("--out", default="")
    p.add_argument("--name", default="")
    p = sub.add_parser("sweep", help="run one probe over several candidates")
    p.add_argument("--probe", required=True, choices=("speed", "gt", "mate"))
    p.add_argument("--cands", required=True)
    p.add_argument("--bases", default=",".join(BASES))
    p.add_argument("--out", required=True)
    p.add_argument("-j", "--jobs", type=int, default=4)
    p.add_argument("--stagger", type=float, default=5.0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("table", help="summarise every run directory under --out")
    p.add_argument("--out", required=True)
    p.add_argument("--json", action="store_true")
    sub.add_parser("list", help="the candidates and their knobs")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        if args.out:
            passthrough += ["--out", args.out]
        if args.name:
            passthrough += ["--name", args.name]
        return cmd_run(args, passthrough)
    return {"sweep": cmd_sweep, "table": cmd_table, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
