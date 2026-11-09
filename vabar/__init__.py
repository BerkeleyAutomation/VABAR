# SPDX-License-Identifier: Apache-2.0
"""VABAR: the Variational Automation Benchmark for Agentic Robotics, on RoboSimStudio.

The benchmark is the six suites under ``tasks/``; this package is only what the
upstream library does not ship. Three calls connect the two:

* :func:`install_routines` runs on import. Every suite's ``staging.py`` imports
  this package, so any process that scans the VABAR task root (a pool worker, a
  fleet, the studio, upstream's own ``scripts/bench_run.py --task-root``) can
  lower the ``(:routine ...)`` documents the suites ship.
* :func:`vabar.config.activate_from_env` runs on import too, for the same
  reason: the experiment config ``$VABAR_CONFIG`` names (``default`` when unset)
  rewrites the registered variants it overrides -- physics, cameras, budget --
  in every such process, workers included. See :mod:`vabar.config`.
* :func:`install` additionally points RoboSimStudio's task registry at
  ``tasks/``, and is what the CLI and the studio launcher call first.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

__all__ = ["REPO", "TASKS", "install", "install_routines", "skill_registries"]

REPO = Path(__file__).resolve().parents[1]
"""The checkout: ``tasks/``, ``assets/``, ``benchmark/`` and ``robot_skills/`` live here."""

TASKS = REPO / "tasks"
"""The VABAR task root: one directory per suite."""


def skill_registries() -> list[Path]:
    """The GaP skill registries a VABAR graph resolves bundles from, in precedence order.

    The cable bundles shipped in ``robot_skills/``, then an open-robot-skills
    checkout: ``$VABAR_OPEN_ROBOT_SKILLS``, else ``../open-robot-skills`` beside
    this checkout. Only directories that exist are named, because gap refuses a
    ``$GAP_SKILLS_PATH`` entry that does not.
    """
    candidates = [REPO / "robot_skills"]
    candidates.append(Path(os.environ.get("VABAR_OPEN_ROBOT_SKILLS") or REPO.parent / "open-robot-skills"))
    return [p.expanduser().resolve() for p in candidates if p.expanduser().is_dir()]


def install_routines() -> None:
    """Make :mod:`vabar.routines` the routine module RoboSimStudio looks for. Idempotent."""
    from robosimstudio_bench.solutions.declared import ROUTINES_MODULE

    if ROUTINES_MODULE in sys.modules:
        return
    module = importlib.import_module("vabar.routines")
    sys.modules[ROUTINES_MODULE] = module
    parent, _, leaf = ROUTINES_MODULE.rpartition(".")
    setattr(importlib.import_module(parent), leaf, module)


def install(*, only: bool = True) -> Path:
    """Register the VABAR task root with RoboSimStudio. Idempotent.

    Args:
        only: List VABAR's suites alone, without any root a RoboSimStudio
            checkout contributes. Set before the registry first loads.

    Returns:
        The task root.
    """
    if only:
        os.environ.setdefault("ROBOSIMSTUDIO_TASK_ROOTS_ONLY", "1")
    if not os.environ.get("GAP_SKILLS_PATH", "").strip():
        os.environ["GAP_SKILLS_PATH"] = os.pathsep.join(str(p) for p in skill_registries())
    install_routines()
    from robosimstudio.roots import add_task_root

    add_task_root(TASKS)
    return TASKS


install_routines()

from .config import activate_from_env  # after REPO and TASKS, which it reads

activate_from_env()
