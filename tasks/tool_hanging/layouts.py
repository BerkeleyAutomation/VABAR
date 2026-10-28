# SPDX-License-Identifier: Apache-2.0
"""Seeded Level-1/2 layouts ported from RoboSimStudio's gap_perception branch."""

from __future__ import annotations

import numpy as np

from .scenes import TOOL_HANGING_BENCH as CELL

_FOOTPRINTS = {
    "wrench": (0.105, 0.025),
    "screwdriver": (0.110, 0.030),
    "scissors": (0.120, 0.060),
    "pliers": (0.120, 0.055),
}


def _kind(name: str) -> str:
    return next(
        kind
        for kind in _FOOTPRINTS
        if kind in name or (kind == "pliers" and name.startswith("plier_"))
    )


def _yaw_rotated(quat, yaw: float) -> tuple[float, ...]:
    x, y, z, w = map(float, quat)
    s, c = np.sin(yaw / 2.0), np.cos(yaw / 2.0)
    return (c * x - s * y, c * y + s * x, c * z + s * w, c * w - s * z)


def _corners(position, kind: str, yaw: float, clearance: float = 0.01) -> np.ndarray:
    a, b = _FOOTPRINTS[kind]
    a, b = a + clearance, b + clearance
    local = np.array(((a, b), (a, -b), (-a, -b), (-a, b)))
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array(((c, -s), (s, c))).T + position


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for polygon in (a, b):
        for edge in (np.roll(polygon, -1, axis=0) - polygon)[:2]:
            normal = np.array((-edge[1], edge[0]))
            normal /= np.linalg.norm(normal)
            if (a @ normal).max() < (b @ normal).min() or (b @ normal).max() < (a @ normal).min():
                return False
    return True


def _object_xform(obj, xy, yaw: float = 0.0) -> tuple[float, ...]:
    """An object transform in the task's work-surface frame."""
    return (
        float(xy[0]),
        float(xy[1]),
        float(obj.pose[2]),
        *_yaw_rotated(obj.quat, yaw),
    )


def sample_level1_layout(task, episode_id: int, rng, **_) -> dict:
    """Nominal episode zero; later episodes jitter the one tabletop tool."""
    if episode_id == 0:
        return {}
    tool = next(obj for obj in task.objects if not obj.static)
    xy = rng.uniform(-0.02, 0.02, size=2)
    return {"object_xforms": {tool.name: _object_xform(tool, xy)}}


def sample_level2_layout(task, episode_id: int, rng, **_) -> dict:
    """Jitter two targets/fixtures and activate a seeded subset of distractors."""
    if episode_id == 0:
        return {}
    targets = tuple(obj for obj in task.objects if not obj.name.startswith("distractor_"))
    distractors = tuple(obj for obj in task.objects if obj.name.startswith("distractor_"))
    objects: dict[str, tuple[float, ...]] = {}
    occupied: list[np.ndarray] = []
    centre = np.zeros(2, dtype=float)
    half = np.asarray(CELL.half[:2], dtype=float)
    lower, upper = centre - half, centre + half

    for obj in targets:
        nominal = centre + np.asarray(obj.pose[:2], dtype=float)
        for _attempt in range(4096):
            position = nominal + rng.uniform(-0.05, 0.05, size=2)
            yaw = float(rng.uniform(-np.deg2rad(5.0), np.deg2rad(5.0)))
            footprint = _corners(position, _kind(obj.name), yaw)
            if np.all(footprint.min(axis=0) >= lower) and np.all(footprint.max(axis=0) <= upper):
                if not any(_overlap(footprint, other) for other in occupied):
                    occupied.append(footprint)
                    objects[obj.name] = _object_xform(obj, position, yaw)
                    break
        else:
            raise RuntimeError(f"could not sample a collision-free Level-2 pose for {obj.name}")

    fixtures = {
        "wrenches_level2": ("wrench_hook_1", "wrench_hook_2"),
        "screwdrivers_level2": ("rack",),
        "scissors_level2": ("scissor_hook_1", "scissor_hook_2"),
        "pliers_level2": ("plier_support_left", "plier_support_right"),
    }[task.name]
    centre_y = float(rng.uniform(-0.01, 0.01))
    half_span = 0.1 + float(rng.uniform(-0.01, 0.01))
    dz = float(rng.uniform(-0.01, 0.01))
    sides = (0.0,) if len(fixtures) == 1 else (1.0, -1.0)
    static_xforms = {}
    for name, side in zip(fixtures, sides, strict=True):
        xform = np.asarray(next(item.xform for item in task.statics if item.name == name), dtype=float).copy()
        xform[1] += centre_y + side * (half_span - 0.1)
        xform[2] += dz
        static_xforms[name] = tuple(float(value) for value in xform)

    active = set(rng.choice(len(distractors), size=int(rng.integers(1, 4)), replace=False))
    for index, obj in enumerate(distractors):
        if index not in active:
            objects[obj.name] = (2.5 + 0.2 * index, 2.5, obj.pose[2], *obj.quat)
            continue
        for _attempt in range(4096):
            position = rng.uniform(lower, upper)
            yaw = float(rng.uniform(-np.pi, np.pi))
            footprint = _corners(position, _kind(obj.name), yaw)
            if np.all(footprint.min(axis=0) >= lower) and np.all(footprint.max(axis=0) <= upper):
                if not any(_overlap(footprint, other) for other in occupied):
                    occupied.append(footprint)
                    objects[obj.name] = _object_xform(obj, position, yaw)
                    break
        else:
            raise RuntimeError(f"could not sample a collision-free distractor pose for {obj.name}")
    return {"object_xforms": objects, "static_xforms": static_xforms}
