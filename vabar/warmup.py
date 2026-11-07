# SPDX-License-Identifier: Apache-2.0
"""Compile the RTX shader cache and the Warp kernels once, before anything is timed.

    python -m vabar warmup                                # one task per suite, on its own renderer, else rtx
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

**The pipelines compile once per install, not once per scene.** A first frame builds
the same ray-tracing pipelines whatever the scene, about 7 minutes of CPU, and the
driver keeps them in one 128 MiB file that only one process at a time may use. So the
first scene here runs alone and compiles, and that file is then copied for the workers
to come (:func:`compile_pipelines`). The compile counts only if its process exits
cleanly, because a process killed while compiling can leave a partial file.

``--cell`` renders each distinct task of the named cells (``--all-instances``: every
instance, ``--first-scene``: only the first). A cell's first run warms only its first
scene, for its Warp kernels; 396 RTX builds, every scene of every cell under all three
configs, compiled no pipelines beyond that one file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
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


def cell_scenes(
    cells: list[str], config: cfg.Config, *, every_instance: bool = False, first_only: bool = False
) -> dict[str, tuple[str, int]]:
    """``name -> (task, episode)``: each distinct task of ``cells`` (its first instance), or every instance, or one."""
    from .benchmark import load

    manifest = load()["cells"]
    keys = list(manifest) if cells == ["all"] else cells
    unknown = sorted(set(keys) - set(manifest))
    if unknown:
        raise SystemExit(f"[warmup] unknown cell(s) {unknown}; see `vabar bench --list`")
    out: dict[str, tuple[str, int]] = {}
    for key in keys:
        for instance in manifest[key]["instances"][: 1 if first_only else None]:
            task = config.map_task(instance["task"])
            episode = int(instance["episode"])
            name = f"{task}@{episode}" if every_instance else task
            out.setdefault(name, (task, episode))
    return out


SCENE_TIMEOUT_S = 900.0
"""How long one scene may take before its process is killed [s]; the pool watchdog's build budget.

**A build can hang for ever, burning 11 cores with its log silent**, and nothing
here would otherwise end it: a `vabar bench` sat 35 minutes on one gear scene,
with the other 29 warmed in 8. The slowest healthy cold compile measured is 713 s."""

SCENE_SUFFIXES = (".task", ".suite", ".py", ".json", ".xml", ".toml")
"""What a suite's scenes are built from. Meshes and textures rarely change without one of these changing too."""

SCENE_ATTEMPTS = 2
"""The pool watchdog's retries have cleared this hang each time so far, so a killed scene gets one more try."""


def run_scene(cmd: Sequence[str], log: Path, env: dict[str, str], timeout_s: float) -> bool:
    """Run one scene's process to its end, or kill it and everything it started; whether it ended by itself."""
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
            return True
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return False


def _render_cache() -> Path | None:
    """The renderer's cache directory, inside the ``ovrtx`` package; ``None`` without it.

    A new environment starts cold, and everything recorded here starts gone with it.
    """
    spec = importlib.util.find_spec("ovrtx")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent / "bin" / "cache"


def stamp_path(cell: str, config: cfg.Config) -> Path | None:
    """Where this install records that a cell's scenes are warmed; ``None`` without the RTX renderer."""
    cache = _render_cache()
    if cache is None:
        return None
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{config.ref}__{cell}")
    return cache / "vabar_warm" / name


def _pipelines_stamp() -> Path | None:
    """Beside the driver's files, so deleting them to reclaim the space deletes this too."""
    cache = _render_cache()
    return None if cache is None else cache / "nv_shadercache" / "vabar_compiled"


def _nvidia_smi() -> list[str]:
    """One ``name, driver`` line per GPU; none if there is no ``nvidia-smi``."""
    query = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    try:
        found = subprocess.run(query, capture_output=True, text=True, timeout=60, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in found.stdout.splitlines() if line.strip()]


def pipelines_compiled() -> bool:
    """Whether the driver's cache holds a whole compile from this GPU and driver."""
    stamp = _pipelines_stamp()
    if stamp is None or not stamp.is_file():
        return False
    gpus = sorted(set(_nvidia_smi()))
    return bool(gpus) and stamp.read_text() == "\n".join(gpus)


FILES_PER_GPU = 16
"""One more than the most workers any config runs on one RTX cell."""

MOST_FILES = 72
"""Enough for seven RTX cells at once, the most the benchmark has been run at, on 9.6 GB."""


def spread_pipelines(count: int) -> int:
    """Copy the driver's compiled file until ``count`` processes can each hold one; how many it added.

    **Warmed workers were compiling everything again, because each process needs its
    own file.** The driver names its cache files in a fixed sequence. A process takes
    a POSIX write lock on the first file nobody holds and uses only that one, and when
    the file is new it compiles and fills it. Every file holds the same 673 entries,
    whichever scene or cell wrote it: an install that had run all the RTX cells held
    69 files, 9 GB, of one 134 MB set. So a cell needs a complete file per worker, plus
    those of anything else running from the install, and warming 8 scenes at a time
    left 8. Measured with 15 workers on ``tool_sorting/L2``, on an idle machine for the
    first row, and under load for the others:

    ==============================  ==============================================
    Files at the start              Result
    ==============================  ==============================================
    69, written by compiling        15-35 s per build, no new files
    8, written by compiling         8 built; 7 compiled for 456-546 s; 7 new files
    15, copies of one               none compiled, no new files
    ==============================  ==============================================

    A copy under the next name works because a file does not record its own. If that
    ever stops holding, a worker compiles and writes its own file, as it would have
    without this. The driver's read-only switch, ``__GL_SHADER_DISK_CACHE_READ_ONLY``,
    also lets every process share one file, and builds as fast, but it keeps nothing:
    a scene that needed a new pipeline would compile it again on every run. Hard links
    cannot help, since a lock belongs to the file underneath.
    """
    cache = _render_cache()
    added = 0
    for directory in cache.glob("nv_shadercache/vulkan/GLCache/*/*") if cache else ():
        tables = sorted(directory.glob("*.toc"))
        if not tables:
            continue
        first = int(tables[0].stem, 16)
        for index in range(1, count):
            name = f"{first + index:016x}"
            if (directory / f"{name}.toc").exists():
                continue
            # Under a name no process looks for until it is whole, and the table last:
            # a worker starting on another GPU may reach this file at any moment.
            for suffix in (".bin", ".toc"):
                partial = directory / f"{name}{suffix}.partial"
                shutil.copyfile(tables[0].with_suffix(suffix), partial)
                partial.rename(directory / f"{name}{suffix}")
            added += 1
    return added


def compile_pipelines(compile_one: Callable[[], bool]) -> bool:
    """Run ``compile_one`` as this install's one compile, unless that has happened; whether it has now.

    Runs started together on a cold install wait here for the first one's compile.
    """
    stamp = _pipelines_stamp()
    if stamp is None:
        return True
    stamp.parent.mkdir(parents=True, exist_ok=True)
    with open(stamp.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if pipelines_compiled():
            return True
        if not compile_one():
            return False
        gpus = sorted(set(_nvidia_smi()))
        spread_pipelines(min(MOST_FILES, FILES_PER_GPU * max(1, len(_nvidia_smi()))))
        stamp.write_text("\n".join(gpus))
    return True


def _renderer(config: cfg.Config, cell: str, extra: Sequence[str]) -> str:
    """The renderer a run will use: its own ``--renderer`` flag, else what its config says of its suite."""
    from .benchmark import load

    for index, arg in enumerate(extra):
        if arg == "--renderer" and index + 1 < len(extra):
            return extra[index + 1]
        if arg.startswith("--renderer="):
            return arg.split("=", 1)[1]
    return config.renderer(load()["cells"][cell]["suite"])


def scene_digest(cell: str) -> str:
    """A hash of the files a cell's suite builds its scenes from, kept in the stamp.

    A stamp alone says a cell was compiled, not which scene. New materials after a
    ``git pull`` then compile during the first perception, which is the timeout
    warming exists to prevent. A stamp written before this held nothing, so it
    warms once more, and a cached scene takes about 20 s.
    """
    from .benchmark import load

    root = REPO / "tasks" / load()["cells"][cell]["suite"]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix in SCENE_SUFFIXES and path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def ensure_warm(cell: str, config: cfg.Config, extra: Sequence[str] = ()) -> None:
    """Compile a cell's scenes the first time this install runs its reference graph.

    **On a cold install a cell's first run does not just run slowly, it fails.**
    Under the cells' ``--bus-timeout 180`` a graph's first perception asks for a
    frame the renderer is still compiling for, and the episode aborts. Measured
    on ``tool_hanging/L1`` with ``--limit 6``, on two new environments: 0/6 both
    times, 4 episodes with ``BusTimeout: render was not answered within 180 s``.
    The same environment's second run scored 5/6, with no timeouts. A warmup
    that fails is not recorded, so the next run tries again.
    """
    renderer = _renderer(config, cell, extra)
    if renderer != "rtx":
        return
    stamp = stamp_path(cell, config)
    if stamp is None:
        return
    digest = scene_digest(cell)
    if stamp.exists() and stamp.read_text() == digest and pipelines_compiled():
        return
    print(f"[bench] first run of {cell} on this install: warming its scenes (once; --no-warmup skips)", flush=True)
    sets = [x for s in config.overrides for x in ("--set", s)]
    if main(["--cell", cell, "--config", config.ref, *sets, "--first-scene", "--renderer", renderer]) != 0:
        print(f"[bench] warming {cell} failed for some scenes: its first perception may time out", flush=True)
        return
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(digest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vabar warmup", description=__doc__.splitlines()[0])
    parser.add_argument("--suites", default="", help="comma-separated suites; default every suite in the benchmark")
    parser.add_argument("--renderer", default="", help="the renderer to compile for (default: the suite's, else rtx)")
    parser.add_argument("--out", default="outputs/warmup", help="where the throwaway run directories go")
    parser.add_argument("--print", action="store_true", help="print the commands and exit")
    parser.add_argument("--config", default=None, help="experiment config whose physics and cameras to compile for")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one config key")
    parser.add_argument("--cell", action="append", default=[], help="warm every scene of this cell (repeatable; 'all')")
    parser.add_argument("--all-instances", action="store_true", help="with --cell: every instance, not each task once")
    parser.add_argument("--first-scene", action="store_true", help="with --cell: only each cell's first instance")
    parser.add_argument("--jobs", type=int, default=0, help="scenes at once (default: 8 with --cell, else 1)")
    args = parser.parse_args(argv)
    try:
        config = cfg.resolve(args.config, args.set)
    except cfg.ConfigError as exc:
        print(f"[warmup] {exc}", file=sys.stderr)
        return 2

    if args.cell:
        cells = [c.strip() for arg in args.cell for c in arg.split(",") if c.strip()]
        scenes = cell_scenes(cells, config, every_instance=args.all_instances, first_only=args.first_scene)
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

    def renderer_of(task: str) -> str:
        return args.renderer or config.renderer(task) or "rtx"

    def command(label: str, task: str, episode: int) -> tuple[str, list[str]]:
        name = "warmup_" + label.replace("/", "_").replace("@", "_")
        # One process per scene, from this checkout (cwd), so a scene that
        # fails to build does not take the others with it.
        cmd = [sys.executable, "-m", "vabar", "run", "--config", config.ref, *sets]
        cmd += ["--task", task, "--episodes", str(episode), "--solve", "idle", "--renderer", renderer_of(task)]
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
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # a live log, for a step that can take minutes
        for attempt in range(1, SCENE_ATTEMPTS + 1):
            if run_scene(cmd, log, env, SCENE_TIMEOUT_S):
                break
            print(f"[warmup] {label}: killed after {SCENE_TIMEOUT_S:.0f} s (attempt {attempt})", flush=True)
        else:
            print(f"[warmup] {label}: FAILED, hung {SCENE_ATTEMPTS} times; see {log}", flush=True)
            return False
        wall = time.perf_counter() - started
        meta = out / name / "run_meta.json"
        builds = json.loads(meta.read_text()).get("provenance", {}).get("builds", []) if meta.is_file() else []
        if builds:
            print(f"[warmup] {label}: ok in {wall:.0f} s (build {builds[0].get('build_s', 0):.0f} s)", flush=True)
            return True
        print(f"[warmup] {label}: FAILED after {wall:.0f} s; see {log}", flush=True)
        return False

    renderer = args.renderer or "each suite's own, else rtx"
    print(f"[warmup] {len(scenes)} scene(s), {jobs} at a time, renderer: {renderer} (logs: {out})", flush=True)
    remaining = list(scenes)

    def compile_first() -> bool:
        first = next(label for label, (task, _) in scenes.items() if renderer_of(task) == "rtx")
        remaining.remove(first)
        return warm(first)

    rtx = any(renderer_of(task) == "rtx" for task, _ in scenes.values())
    compiled = not rtx or compile_pipelines(compile_first)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(pool.map(warm, remaining))
    return 0 if compiled and all(results) else 1
