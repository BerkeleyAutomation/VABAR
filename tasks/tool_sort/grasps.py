# SPDX-License-Identifier: Apache-2.0
"""Which face of a box a parallel jaw can close on, and how well.

Pure geometry: an oriented box, a hand's span and pad depth, and the six faces
that box offers. No scene, no simulator, no robot -- which is why it is its own
module, and why a test can exercise it on the export's measured half-extents
without building anything.

Split out of ``routine.py`` rather than living there because a routine is
choreography over a scene and this is a fact about a shape. The four constants
travel with it for the same reason: they are properties of a hand and a box,
not of a task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FIT_MARGIN", "MAX_TILT_DEG", "PAD_DEPTH", "POS_TOL", "Grasp", "face_grasps", "support"]

FIT_MARGIN = 0.012
"""How much narrower than the jaws' full span a grasp width has to be [m].

Six millimetres a side. The fingers have to get *around* the object before they
close on it, and a box collider offers no compliance to make up the difference:
at 0.080 m span the 0.087 m multimeter is not a tight grasp, it is a pair of
fingertips driving into the object on the way down.
"""

MAX_TILT_DEG = 68.0
"""How far off vertical the approach may lean [deg].

Past this the hand is reaching in sideways: the wrist runs out of room over a
bin whose walls stand 0.08 m above its floor, and the lower finger spends its
descent inside the pile rather than beside it. 68 degrees admits the leaning
multimeter (whose only fitting axis sits 30 degrees off vertical, so the
approach perpendicular to it sits 60 degrees off) and little else.
"""

PAD_DEPTH = 0.025
"""How far inside an object's near surface the pads are placed [m].

The bite the jaws take. Measured against the hand rather than chosen: the pads
are 8 mm boxes at the commanded tool centre and the fingers stand 45 mm back
from it, so a bite of 25 mm has the object well between the pads and the finger
bodies clear of everything the object is standing on. Anything thinner than
2 x :data:`PAD_DEPTH` along the approach is simply grasped at its centre.
"""

POS_TOL = 0.008
"""Position tolerance a leg counts as arrived at [m]."""


@dataclass(frozen=True)
class _Leg:
    """One leg of a move: a pose to fly to, and when to stop flying it."""

    label: str
    position: np.ndarray
    quat: np.ndarray
    grip: float
    max_steps: int
    converge: bool = True
    """Whether arriving ends the leg. False holds the full budget -- which is
    what a gripper command wants, since the arm is already where it should be
    and the thing being waited for is the fingers."""
    tol: float = POS_TOL
    """How near the target counts as arrived [m]."""


@dataclass(frozen=True)
class Grasp:
    """One proposed grasp of one object's box.

    Attributes:
        position: World point the jaws close around [m].
        approach: World direction the hand comes in along.
        close: World direction the jaws close across.
        width: The box's extent along :attr:`close` [m].
        score: Higher is better, within one search.
    """

    position: np.ndarray
    approach: np.ndarray
    close: np.ndarray
    width: float
    score: float

    @property
    def tilt_deg(self) -> float:
        """How far off straight down the hand comes in [deg]."""
        return float(np.degrees(np.arccos(np.clip(-self.approach[2], -1.0, 1.0))))


def support(axes: np.ndarray, half: np.ndarray, direction: np.ndarray) -> float:
    """Half a box's extent along an arbitrary world direction [m].

    Args:
        axes: The box's world axes, as columns.
        half: Its half-extents along those axes [m].
        direction: A unit world direction.
    """
    return float(np.sum(half * np.abs(axes.T @ direction)))


def face_grasps(
    centre: np.ndarray,
    axes: np.ndarray,
    half: np.ndarray,
    *,
    span: float,
    max_tilt_deg: float = MAX_TILT_DEG,
    pad_depth: float = PAD_DEPTH,
) -> list[Grasp]:
    """Every grasp of a box the jaws fit around, best first, with no scene.

    The geometric half of the search, kept free of handles and Warp so that what
    it claims can be checked in arithmetic: a battery lying flat proposes a
    top-down grasp across its 33 mm, a plate on edge proposes one across its
    thickness, the same plate lying flat proposes nothing at all, and no
    proposal ever asks the jaws to close across something wider than they open.
    The scene-dependent half -- is the floor in the way, is a neighbour, can this
    wrist be reached -- is :meth:`RoutineSortFaces._candidates`, which
    filters this list.

    Six closing directions (the box's three axes, each way) and five approaches
    for each: the four remaining axis directions, and the steepest *downward*
    direction perpendicular to the closing axis. That last one is the one that
    matters, because for a tilted plate none of the box's own axes points where
    the hand wants to come from.

    Args:
        centre: The box's centre in world coordinates [m].
        axes: Its world axes, as columns.
        half: Its half-extents along those axes [m].
        span: The widest the jaws may be asked to close across [m]. Pass the
            hand's span already reduced by whatever clearance the fingers need
            to get around an object -- see :data:`FIT_MARGIN`.
        max_tilt_deg: How far off vertical an approach may lean.
        pad_depth: How far inside the near surface the pads are placed [m].

    Returns:
        Candidates sorted by :attr:`Grasp.score`, highest first.
    """
    down = np.array([0.0, 0.0, -1.0])
    out: list[Grasp] = []
    for index in range(3):
        width = 2.0 * float(half[index])
        if width > span:
            continue
        for sign in (1.0, -1.0):
            close = sign * axes[:, index]
            approaches = [down - float(np.dot(down, close)) * close]
            approaches += [s * axes[:, j] for j in range(3) if j != index for s in (1.0, -1.0)]
            for raw in approaches:
                norm = float(np.linalg.norm(raw))
                if norm < 1.0e-9:
                    continue
                approach = raw / norm
                tilt = float(np.degrees(np.arccos(np.clip(-approach[2], -1.0, 1.0))))
                if tilt > max_tilt_deg:
                    continue
                # How deep to sit the pads, which is not the same question as
                # where the object's centre is. The tool centre this arm
                # commands *is* the pad plane, so aiming at the centre of a
                # 0.17 m object standing on end drives the fingers 60 mm past
                # its near face, through whatever is beside it, and stalls the
                # arm on contact with the object shoved out of the way. On
                # anything flatter than the bite the expression collapses to the
                # centre, which is the baseline's grasp unchanged.
                reach = support(axes, half, approach)
                position = centre - approach * max(0.0, reach - pad_depth)
                # Prefer coming straight down, then a firm fit, then a high
                # grasp: the first two are about whether the pick works and the
                # third about what it disturbs on the way.
                score = 3.0 * float(-approach[2]) + 1.5 * (1.0 - width / max(span, 1.0e-6)) + 2.0 * float(position[2])
                out.append(Grasp(position=position, approach=approach, close=close, width=width, score=score))
    out.sort(key=lambda grasp: -grasp.score)
    return out
