# SPDX-License-Identifier: Apache-2.0
"""Compile the RTX shader cache and the Warp kernels once, before anything is timed.

    python -m vabar warmup                                # one task per suite, --renderer rtx
    python -m vabar warmup --suites tool_hanging,gear_mesh
    python -m vabar warmup --cell syringe_packing/L1      # every scene the cell renders, 8 at a time
    python -m vabar warmup --cell all --jobs 12           # every scene of every cell
    python -m vabar warmup --config fast --print          # the commands, without running them

The first RTX render on a machine compiles the path tracer's shader cache, and a
suite's first build compiles its Warp kernels; both are cached on disk afterwards.
A timed run that pays for them measures the machine's first day rather than the
benchmark. This runs one idle control step per scene, each in its own process,
under the config's physics and cameras (so the kernels compiled are the ones a
run uses) with the observations rendered every step, so the renderer actually
draws.

**Warm the cell, not just the suite.** The renderer compiles pipelines for each
scene's own materials, not once per machine: on syringe_packing/L1 a scene not
yet compiled took 251 s to build (21 s once compiled), and whichever pool worker
drew it stalled about 5 minutes while the others ran. ``--cell`` renders every
distinct task of the named cells (``--all-instances``: every instance), so a
benchmark run starts from a warm cache. **A compile is kept only if its process
exits cleanly**: a worker that is killed loses what it compiled, and the next run
compiles it again (measured: a scene compiled by a killed worker took 251 s
again; after one clean exit, 21 s).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from . import REPO
from . import config as cfg


def warmup_tasks(config: cfg.Config) -> dict[str, tuple[str, int]]:
    """Each suite's first benchmark instance, ``suite -> (task, episode)``, mapped onto the config's variant."""
    from .benchmark import load

    out: dict[str, tuple[str, int]] = {}
    for cell in load()["cells"].values():
        if cell["instances"] and cell["suite"] not in out:
            first = cell["instances"][0]
            out[cell["suite"]] = (config.map_task(first["task"]), int(first["episode"]))
    return out


def cell_scenes(cells: list[str], config: cfg.Config, *, every_instance: bool) -> dict[str, tuple[str, int]]:
    """``name -> (task, episode)``: each distinct task of ``cells`` (its first instance), or every instance."""
    from .benchmark import load

    manifest = load()["cells"]
    keys = list(manifest) if cells == ["all"] else cells
    unknown = sorted(set(keys) - set(manifest))
    if unknown:
        raise SystemExit(f"[warmup] unknown cell(s) {unknown}; see `vabar bench --list`")
    out: dict[str, tuple[str, int]] = {}
    for key in keys:
        for instance in manifest[key]["instances"]:
            task = config.map_task(instance["task"])
            episode = int(instance["episode"])
            name = f"{task}@{episode}" if every_instance else task
            out.setdefault(name, (task, episode))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vabar warmup", description=__doc__.splitlines()[0])
    parser.add_argument("--suites", default="", help="comma-separated suites; default every suite in the benchmark")
    parser.add_argument("--renderer", default="rtx", help="the renderer whose shaders to compile (default rtx)")
    parser.add_argument("--out", default="outputs/warmup", help="where the throwaway run directories go")
    parser.add_argument("--print", action="store_true", help="print the commands and exit")
    parser.add_argument("--config", default=None, help="experiment config whose physics and cameras to compile for")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one config key")
    parser.add_argument("--cell", action="append", default=[], help="warm every scene of this cell (repeatable; 'all')")
    parser.add_argument("--all-instances", action="store_true", help="with --cell: every instance, not each task once")
    parser.add_argument("--jobs", type=int, default=0, help="scenes at once (default: 8 with --cell, else 1)")
    args = parser.parse_args(argv)
    try:
        config = cfg.resolve(args.config, args.set)
    except cfg.ConfigError as exc:
        print(f"[warmup] {exc}", file=sys.stderr)
        return 2

    if args.cell:
        cells = [c.strip() for arg in args.cell for c in arg.split(",") if c.strip()]
        scenes = cell_scenes(cells, config, every_instance=args.all_instances)
        jobs = args.jobs or 8
    else:
        tasks = warmup_tasks(config)
        wanted = [s.strip() for s in args.suites.split(",") if s.strip()] or list(tasks)
        unknown = sorted(set(wanted) - set(tasks))
        if unknown:
            print(f"[warmup] no benchmark instance for {unknown}; the suites are {sorted(tasks)}", file=sys.stderr)
            return 2
        scenes = {suite: tasks[suite] for suite in wanted}
        jobs = args.jobs or 1
    out = Path(args.out).resolve()
    sets = [x for s in (*config.overrides, "runtime.obs_render=per-step", "runtime.clock=free") for x in ("--set", s)]

    def command(label: str, task: str, episode: int) -> tuple[str, list[str]]:
        name = "warmup_" + label.replace("/", "_").replace("@", "_")
        # One process per scene, from this checkout (cwd), so a scene that
        # fails to build does not take the others with it -- and each exits
        # cleanly, which is when the renderer writes what it compiled.
        cmd = [sys.executable, "-m", "vabar", "run", "--config", config.ref, *sets]
        cmd += ["--task", task, "--episodes", str(episode), "--solve", "idle", "--renderer", args.renderer]
        cmd += ["--max-steps", "1", "--out", str(out), "--name", name]
        return name, cmd

    if args.print:
        for label, (task, episode) in scenes.items():
            print("python " + shlex.join(command(label, task, episode)[1][1:]))
        return 0

    def warm(label: str) -> bool:
        task, episode = scenes[label]
        name, cmd = command(label, task, episode)
        log = out / f"{name}.log"
        started = time.perf_counter()
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as fh:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # a live log, for a step that can take minutes
            subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
        wall = time.perf_counter() - started
        meta = out / name / "run_meta.json"
        builds = json.loads(meta.read_text()).get("provenance", {}).get("builds", []) if meta.is_file() else []
        if builds:
            print(f"[warmup] {label}: ok in {wall:.0f} s (build {builds[0].get('build_s', 0):.0f} s)", flush=True)
            return True
        print(f"[warmup] {label}: FAILED after {wall:.0f} s; see {log}", flush=True)
        return False

    print(f"[warmup] {len(scenes)} scene(s), {jobs} at a time, --renderer {args.renderer} (logs: {out})", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(pool.map(warm, scenes))
    return 0 if all(results) else 1
