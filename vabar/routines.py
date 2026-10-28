# SPDX-License-Identifier: Apache-2.0
"""Running a task's own shipped routine as a solution.

RoboSimStudio leaves its scripted-routine mechanism out of the public release
(:data:`robosimstudio_bench.solutions.declared.ROUTINES_MODULE`): there a task's
policy is a graph, and a document that names a ``(:routine ...)`` is refused at
lowering. Every VABAR suite ships a scripted groundtruth beside its graphs, so
this module is that mechanism, and :func:`vabar.install_routines` puts it where
upstream looks for it. Nothing in upstream is edited; ``routines_installed()``
simply starts answering yes.

A routine is any class the document names with ``(:routine "routine:Name")``,
constructed as ``Routine(env, **kwargs)`` and driven by ``act()`` once per step,
with an optional ``reset()``, ``MODE`` and ``row_extras()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from robosimstudio_bench.solutions.base import SolutionBase

__all__ = ["SolutionRoutine"]


class SolutionRoutine(SolutionBase):
    """Drives the environment with its task's shipped routine.

    Args:
        env: The environment. Its task must declare a ``routine``.
        **kwargs: Forwarded to the routine's constructor, so a run can select a
            grasp mode or an arm without a new task variant.
    """

    def __init__(self, env, **kwargs: Any) -> None:
        super().__init__(env, **kwargs)
        if env.task.routine is None:
            raise ValueError(f"task {env.task.full_name!r} ships no scripted routine")
        self.routine = env.task.routine(env, **kwargs)

    def reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        del obs, info
        if hasattr(self.routine, "reset"):
            self.routine.reset()

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        del obs
        return self.routine.act()

    @property
    def control_mode(self) -> str:
        """The routine's own ``MODE``, when it declares one."""
        return str(getattr(self.routine, "MODE", ""))

    @property
    def name(self) -> str:
        return f"scripted:{type(self.routine).__qualname__}"

    def row_extras(self) -> dict[str, Any]:
        """The routine's own telemetry, when it keeps any."""
        return dict(getattr(self.routine, "row_extras", dict)())
