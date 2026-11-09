# SPDX-License-Identifier: Apache-2.0
"""Run a solution over a VABAR task's episodes: RoboSimStudio's own headless runner.

    python -m vabar run --task tool_hanging/wrenches_level1 --solve routine --record
    python -m vabar run --task tool_hanging/wrenches_level1 --solve graph --graph gt --motion curobo
    python -m vabar run --task usbc_insert/insert_easy --solve graph --episodes 0:30 --workers 8
    python -m vabar run --config paper --task tool_hanging/wrenches_level1 --solve graph --graph gt
    python -m vabar run --set clock=free --set suites.tool_hanging.max_steps=9000 --task ...

Every flag is upstream's ``scripts/bench_run.py``, vendored unmodified as
:mod:`vabar._upstream.bench_run` because RoboSimStudio's wheel does not ship its
``scripts/``. What is added around it:

* :func:`vabar.install` before it parses, so the VABAR suites are registered and
  ``--solve routine`` exists; ``--task-root`` is therefore never needed.
* An experiment config (``--config NAME|path``, ``--set key=value``; see
  :mod:`vabar.config`). It supplies the runtime flags a command line leaves out
  (``--clock``, ``--obs-render``, ``--render-every``, a vision run's
  ``--renderer``, ``--bus-timeout``), its suite's ``flags``, its environment
  (tool servers kept warm, the planner cached), and maps the task onto the
  suite's configured variant. A flag on the command line always wins. Physics,
  cameras and budgets are rewritten in the registry itself, so they need no flag.
  A flag the vendored runner does not accept yet is dropped with one warning.
* ``vabar_config.json`` in the run directory: the resolved config, its sha256
  and the VABAR revision.
* ``--perception vision`` with no renderer named and none from the config gets
  ``--renderer auto`` (the machine's RTX path tracer, else the tiled one), since
  a vision graph needs cameras.

``python -m vabar run --help`` lists the rest.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import itertools
import sys
from pathlib import Path
from typing import Any

from . import config as cfg
from . import install

RENDER_FLAGS = ("--renderer", "--obs-renderer")

RUNTIME_FLAGS = {"clock": "--clock", "obs_render": "--obs-render", "render_every": "--render-every"}
"""The runner flags the interface contract adds, by the config key (and ``JobSpec`` field) each sets."""

CONTRACT_DEFAULTS = {"clock": "free", "obs_render": "per-step", "render_every": 1}
"""What the runner does when those flags are absent: v1.0's behaviour, so asking for it drops nothing."""


# ----------------------------------------------------------------------------- the runner's flags
class RunnerFlags:
    """What the vendored runner accepts -- its options and its ``JobSpec`` fields -- and argv surgery on them.

    Feature detection for the flags and fields the interface contract adds
    upstream (``--clock``, ``--obs-render``, ``--render-every``): a config asks
    for them, and whether they reach the run depends on the runner vendored here.
    """

    def __init__(self, parser: argparse.ArgumentParser, job_fields: set[str]) -> None:
        self.parser = parser
        self._actions = dict(parser._option_string_actions)
        self.job_fields = frozenset(job_fields)

    def accepts(self, flag: str) -> bool:
        return flag in self._actions

    def default(self, flag: str) -> Any:
        action = self._actions.get(flag)
        return None if action is None else action.default

    def _repeatable(self, dest: str) -> bool:
        return any(
            a.dest == dest and type(a).__name__ in ("_AppendAction", "_ExtendAction", "_CountAction")
            for a in self._actions.values()
        )

    def items(self, argv: list[str]) -> list[tuple[str | None, list[str]]]:
        """``argv`` as ``(dest, tokens)`` items: a flag with its value, a bare switch, or a lone token."""
        out: list[tuple[str | None, list[str]]] = []
        i = 0
        while i < len(argv):
            token = argv[i]
            name, eq, _ = token.partition("=") if token.startswith("--") else (token, "", "")
            action = self._actions.get(name)
            if action is None:
                out.append((None, [token]))
                i += 1
            elif eq or action.nargs == 0:
                out.append((action.dest, [token]))
                i += 1
            else:
                out.append((action.dest, argv[i : i + 2]))
                i += 2
        return out

    def dest(self, flag: str) -> str:
        return self._actions[flag].dest

    def value(self, argv: list[str], flag: str) -> str | None:
        """The value ``flag`` (or any of its spellings) has in ``argv``, the last one winning; ``None`` if absent."""
        if flag not in self._actions:
            return None
        dest, found = self.dest(flag), None
        for key, tokens in self.items(argv):
            if key == dest:
                head = tokens[0]
                found = head.partition("=")[2] if "=" in head else (tokens[1] if len(tokens) > 1 else "")
        return found

    def has(self, argv: list[str], flag: str) -> bool:
        return flag in self._actions and any(key == self.dest(flag) for key, _ in self.items(argv))

    def merge(self, base: list[str], extra: list[str], *, replace: bool, skip: set[str] = frozenset()) -> list[str]:
        """``extra``'s flags onto ``base``. A flag ``base`` has is replaced in place (``replace``) or kept.

        Flags whose ``dest`` is in ``skip`` are left out of ``extra``; a
        repeatable flag is always appended.
        """
        items = self.items(base)
        for key, tokens in self.items(extra):
            if key is not None and key in skip:
                continue
            where = [i for i, (k, _) in enumerate(items) if k is not None and k == key]
            if where and not self._repeatable(key):
                if replace:
                    items[where[-1]] = (key, tokens)
                continue
            items.append((key, tokens))
        return [token for _, tokens in items for token in tokens]


@functools.cache
def runner_flags() -> RunnerFlags:
    """The vendored runner's flags. Imports it, after :func:`vabar.install` as it must be."""
    install()
    from robosimstudio.specs import JobSpec

    from ._upstream import bench_run

    return RunnerFlags(bench_run.create_parser(), {f.name for f in dataclasses.fields(JobSpec)})


# ----------------------------------------------------------------------------- config -> flags
def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def config_flags(
    config: cfg.Config, task: str, *, solve: str, vision: bool, flags: RunnerFlags, bench: bool
) -> tuple[list[str], list[str], dict[str, Any]]:
    """The flags ``config`` asks of one ``vabar run`` of ``task``.

    Args:
        config: The config.
        task: ``suite/variant``, already mapped onto the configured variant.
        solve: The ``--solve`` value.
        vision: Whether the run observes through cameras (``--perception vision``).
        flags: The runner's flags, for feature detection.
        bench: A cell's job, which also states a suite's ``max_steps`` as a flag
            (replacing the cell's) rather than leaving it to the rewritten variant.

    Returns:
        The argv tokens to merge; the runtime flags the runner does not accept
        (``"--clock lockstep"``), dropped; and of those, the ones a ``JobSpec``
        can still carry (``{"clock": "lockstep"}``) -- which a job file sets.
    """
    table = config.table(task)
    runtime = config.runtime
    add = list(table.get("flags", []))
    if bench and "max_steps" in table:
        add += ["--max-steps", str(table["max_steps"])]
    if runtime["bus_timeout"] > 0 and solve == "graph":
        add += ["--bus-timeout", _number(runtime["bus_timeout"])]
    renderer = config.renderer(task)
    if renderer and vision:
        add += ["--renderer", renderer]
    dropped: list[str] = []
    fields: dict[str, Any] = {}
    for key, flag in RUNTIME_FLAGS.items():
        value = runtime[key]
        if value in ("", 0):
            continue
        if key == "obs_render" and value == "on-demand" and solve != "graph":
            # Only a graph asks for its renders. Every other policy -- a routine,
            # idle, your own class -- reads the cameras each step, and upstream
            # refuses on-demand for it rather than let it act on frames from reset.
            continue
        if flags.accepts(flag):
            if str(value) != str(flags.default(flag)):
                add += [flag, str(value)]
            continue
        if value == CONTRACT_DEFAULTS[key]:
            continue  # what the runner does without the flag
        if key in flags.job_fields:
            fields[key] = value
        dropped.append(f"{flag} {value}")
    return add, dropped, fields


def dropped_warning(config: cfg.Config, dropped: list[str], *, via_jobs: set[str] = frozenset()) -> str:
    """The one warning a run prints about runtime flags the vendored runner does not take yet."""
    unique = list(dict.fromkeys(dropped))
    carried = [d for d in unique if d.split()[0].lstrip("-").replace("-", "_") in via_jobs]
    lost = [d for d in unique if d not in carried]
    parts = []
    if lost:
        parts.append(
            f"[vabar] warning: config {config.name} asks for {', '.join(lost)}, which the vendored runner "
            "(vabar/_upstream/bench_run.py) does not accept yet; dropped -- this run uses the runner's own "
            "behaviour. Re-vendor bench_run from RoboSimStudio to apply them."
        )
    if carried:
        parts.append(
            f"[vabar] note: config {config.name}: {', '.join(carried)} is not a flag of the vendored runner; "
            "set on each JobSpec in the job file instead."
        )
    return "\n".join(parts)


def with_default_renderer(argv: list[str]) -> list[str]:
    """Upstream's runner builds no renderer unless one is named, and two kinds of run need one.

    A vision graph observes through cameras, and ``--record`` writes its video from
    the renderer: without one it keeps the state and no video at all.
    """
    wants_pictures = _has(argv, "--perception", "vision") or "--record" in argv
    if wants_pictures and not any(a.split("=")[0] in RENDER_FLAGS for a in argv):
        return [*argv, "--renderer", "auto"]
    return argv


def _has(argv: list[str], flag: str, value: str) -> bool:
    """Whether ``flag value`` (or ``flag=value``) is on the command line."""
    return f"{flag}={value}" in argv or any(a == flag and b == value for a, b in itertools.pairwise(argv))


def prepare(argv: list[str], config: cfg.Config, flags: RunnerFlags) -> tuple[list[str], list[str], list[str]]:
    """One ``vabar run`` command line with the config applied: the argv, notes to print, flags dropped."""
    notes: list[str] = []
    task = flags.value(argv, "--task") or ""
    if task:
        mapped = config.map_task(task)
        if mapped != task:
            argv = flags.merge(argv, ["--task", mapped], replace=True)
            notes.append(f"[vabar] config {config.name} runs {task} as {mapped}")
            task = mapped
    solve = flags.value(argv, "--solve") or "idle"
    vision = flags.value(argv, "--perception") == "vision"
    add, dropped, _ = config_flags(config, task, solve=solve, vision=vision, flags=flags, bench=False)
    if any(a.split("=")[0] in RENDER_FLAGS for a in argv):
        add = flags.merge([], add, replace=True, skip={flags.dest("--renderer")})
    argv = with_default_renderer(flags.merge(argv, add, replace=False))
    if dropped:
        notes.append(dropped_warning(config, dropped))
    return argv, notes, dropped


# ----------------------------------------------------------------------------- running
def run_dirs(argv: list[str]) -> list[Path]:
    """The run directories ``vabar run <argv>`` writes, without running it; empty if it would not run."""
    from ._upstream import bench_run

    try:
        args = bench_run.create_parser().parse_args(argv)
    except SystemExit:
        return []
    if args.list:
        return []
    if args.jobs:
        from robosimstudio_bench.run import load_jobs

        return [Path(args.out) / job.name for job in load_jobs(args.jobs)]
    if not args.task and not args.documents:
        return []
    return [Path(args.out) / bench_run.job_from_args(args).name]


def invoke(argv: list[str]) -> int:
    """The vendored runner over ``argv``, in this process."""
    from ._upstream import bench_run

    saved = sys.argv
    sys.argv = ["vabar run", *argv]
    try:
        return bench_run.main()
    finally:
        sys.argv = saved


def run(argv: list[str], config: cfg.Config) -> int:
    """``vabar run`` under ``config`` (already active)."""
    flags = runner_flags()
    if any(a in ("-h", "--help") for a in argv):
        print(
            f"vabar run also takes --config {{{','.join(cfg.names())}}}|FILE.toml and --set KEY=VALUE "
            "(repeatable): the experiment config, see vabar/config.py. Upstream's flags follow.\n"
        )
    if any(a in ("-h", "--help", "--list") for a in argv):
        return invoke(argv)
    dropped: list[str] = []
    if not flags.has(argv, "--jobs"):
        argv, notes, dropped = prepare(argv, config, flags)
        for note in notes:
            print(note)
    env = cfg.apply_env(config, flags.value(argv, "--task") or "")
    for run_dir in run_dirs(argv):
        cfg.write_provenance(run_dir, config, runner_argv=argv, env=env, dropped=dropped)
    return invoke(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[2:] if argv is None else argv)
    try:
        config, argv = cfg.from_argv(argv)
        cfg.activate(config)
    except cfg.ConfigError as exc:
        print(f"[vabar] {exc}", file=sys.stderr)
        return 2
    return run(argv, config)


def list_main(argv: list[str] | None = None) -> int:
    """Print every registered ``suite/variant``."""
    del argv
    install()
    from robosimstudio_bench.taskbuild.registry import list_tasks

    for name in list_tasks():
        print(name)
    return 0
