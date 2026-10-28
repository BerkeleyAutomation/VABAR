# SPDX-License-Identifier: Apache-2.0
"""Run a solution over a VABAR task's episodes: RoboSimStudio's own headless runner.

    python -m vabar run --task tool_hanging/wrenches_level1 --solve routine --record
    python -m vabar run --task tool_hanging/wrenches_level1 --solve graph --graph gt --motion curobo
    python -m vabar run --task usbc_insert/insert_easy --solve graph --episodes 0:30 --workers 8

Every flag is upstream's ``scripts/bench_run.py``, vendored unmodified as
:mod:`vabar._upstream.bench_run` because RoboSimStudio's wheel does not ship its
``scripts/``. The only thing added is :func:`vabar.install` before it parses,
so the VABAR suites are registered and ``--solve routine`` exists; ``--task-root``
is therefore never needed. One default differs: ``--perception vision`` without a
renderer named gets ``--renderer auto`` (the machine's RTX path tracer, else the
tiled one), since a vision graph needs cameras. ``python -m vabar run --help``
lists the rest.
"""

from __future__ import annotations

import itertools
import sys

from . import install

RENDER_FLAGS = ("--renderer", "--obs-renderer")


def _has(argv: list[str], flag: str, value: str) -> bool:
    """Whether ``flag value`` (or ``flag=value``) is on the command line."""
    return f"{flag}={value}" in argv or any(a == flag and b == value for a, b in itertools.pairwise(argv))


def main(argv: list[str] | None = None) -> int:
    install()
    # Imported after install(): the runner decides at import whether
    # `--solve routine` is on offer.
    from ._upstream import bench_run

    argv = list(sys.argv[2:] if argv is None else argv)
    # A vision graph observes through cameras, and upstream's runner builds no
    # renderer unless one is named ("empty is fastest"). Name the machine's own.
    if _has(argv, "--perception", "vision") and not any(a.split("=")[0] in RENDER_FLAGS for a in argv):
        argv += ["--renderer", "auto"]
    saved = sys.argv
    sys.argv = ["vabar run", *argv]
    try:
        return bench_run.main()
    finally:
        sys.argv = saved


def list_main(argv: list[str] | None = None) -> int:
    """Print every registered ``suite/variant``."""
    del argv
    install()
    from robosimstudio_bench.taskbuild.registry import list_tasks

    for name in list_tasks():
        print(name)
    return 0
