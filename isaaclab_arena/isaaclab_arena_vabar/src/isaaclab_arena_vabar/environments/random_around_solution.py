# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply ``random_around_solution`` on top of Arena's pooled placement reset.

Arena 0.3.1 places relation-solved objects through one ``placement_reset`` event
that writes the solved layout as is; neither the pooled nor the static path reads
the ``RandomAroundSolution`` marker, so a scene's declared spawn spread never
happens and every seed starts from the same pose. :func:`install` adds a reset
term to every benchmark graph that declares a marker. It runs after
``placement_reset`` and perturbs each marked object around the pose that event
(or, for an anchor, the object's own reset) just wrote, by the marker's own
half-ranges. Turns are about the centre of the object's bounding box, so its body
turns in place and moves only by the drawn offset.

Draws are rejection-sampled, as AUTOLab's own samplers are: a draw whose
bounding box would cut into another object's, where the two share height, is
drawn again, and after :data:`TRIES` refusals the object keeps its solved pose.
Stacked boxes (a part on its support) share no height and never veto a draw; a
pair whose boxes already overlap in the solved layout may keep, but not grow,
that overlap.

The draw is seeded from the experiment's ``placement_seed``, the environment id
and that environment's own reset count, so env 0 of a vector job draws what a
``num_envs=1`` job with the same placement seed draws, and two runs with one
placement seed start identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

EVENT_NAME = "random_around_solution"

TRIES = 64
"""Draws per object before it keeps its solved pose."""

HEIGHT_TOLERANCE_M = 1.0e-3
"""Vertical overlap below which two boxes are stacked, not side by side."""

FOOTPRINT_TOLERANCE_M = 1.0e-4
"""Each footprint shrinks by this much, so boxes that only touch are clear."""


class PlacementSeed:
    """Read-through handle to the builder's placement parameters.

    The builder writes ``placement_seed`` onto the environment's own
    ``ObjectPlacerParams`` after the scene is built, so the seed is read at reset.
    Shared, not copied, by ``copy.deepcopy`` and opaque to config validation, as
    Arena's ``PlacementPoolHandle`` is.
    """

    __slots__ = ("placer_params",)

    def __init__(self, placer_params: Any) -> None:
        self.placer_params = placer_params

    def __deepcopy__(self, memo: dict[int, object]) -> PlacementSeed:
        memo[id(self)] = self
        return self

    @property
    def seed(self) -> int:
        seed = getattr(self.placer_params, "placement_seed", None)
        return 0 if seed is None else int(seed)


def perturbed_pose(position, rotation_xyzw, marker, rng: np.random.Generator, pivot=(0.0, 0.0, 0.0)):
    """``position`` and ``rotation_xyzw`` moved by one uniform draw within ``marker``'s half-ranges.

    Rotations compose on the world side, so a yaw draw turns the object about the
    vertical whatever its base orientation. The turn is about ``pivot``, a point in
    the object's own frame, which therefore moves by the drawn offset alone.
    """
    half = np.array([marker.x_half_m, marker.y_half_m, marker.z_half_m])
    angles = np.array([marker.roll_half_rad, marker.pitch_half_rad, marker.yaw_half_rad])
    offset = rng.uniform(-1.0, 1.0, 3) * half
    turn = rng.uniform(-1.0, 1.0, 3) * angles
    base = Rotation.from_quat(rotation_xyzw)
    rotation = Rotation.from_euler("xyz", turn) * base
    swing = base.apply(pivot) - rotation.apply(pivot)
    return (
        tuple(float(v) for v in np.asarray(position) + offset + swing),
        tuple(float(v) for v in rotation.as_quat()),
    )


@dataclass(frozen=True)
class Solid:
    """A local bounding box placed in the environment: its footprint and height span."""

    footprint: Any
    z_min: float
    z_max: float

    @classmethod
    def of(cls, box, position, rotation_xyzw) -> Solid:
        """``box`` is the ``(min, max)`` corner pair in the object's own frame."""
        from shapely.geometry import MultiPoint

        lo, hi = (np.asarray(corner, dtype=float) for corner in box)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        world = Rotation.from_quat(rotation_xyzw).apply(corners) + np.asarray(position, dtype=float)
        footprint = MultiPoint([tuple(p) for p in world[:, :2]]).convex_hull.buffer(-FOOTPRINT_TOLERANCE_M)
        return cls(footprint, float(world[:, 2].min()), float(world[:, 2].max()))

    def overlap(self, other: Solid) -> float:
        """Footprint overlap area [m^2] of two boxes that share height, else 0."""
        shared_height = min(self.z_max, other.z_max) - max(self.z_min, other.z_min)
        if shared_height <= HEIGHT_TOLERANCE_M or not self.footprint.intersects(other.footprint):
            return 0.0
        # Touching footprints still count, with a floor, so a merely touching pair is not clear.
        return max(self.footprint.intersection(other.footprint).area, 1.0e-12)

    def collides(self, other: Solid) -> bool:
        return self.overlap(other) > 0.0


def draw_clear_pose(center, marker, rng: np.random.Generator, box, obstacles):
    """One draw around ``center`` that grows no overlap with any obstacle, else ``center``.

    ``obstacles`` pairs each obstacle ``Solid`` with the overlap the two solved
    poses already have (0 for a clear pair). Returns the pose and the number of
    draws it took; ``TRIES + 1`` means none cleared.
    """
    # Turn about the box centre: an asset whose origin sits off its body (the ring
    # wrench's is 22 cm away) would otherwise swing across the table on a yaw draw.
    pivot = 0.5 * (np.asarray(box[0], dtype=float) + np.asarray(box[1], dtype=float))
    for attempt in range(1, TRIES + 1):
        pose = perturbed_pose(center[0], center[1], marker, rng, pivot)
        solid = Solid.of(box, *pose)
        if all(solid.overlap(other) <= allowed + 1.0e-12 for other, allowed in obstacles):
            return pose, attempt
    return (tuple(center[0]), tuple(center[1])), TRIES + 1


def spread_layout(poses, boxes, markers, rng: np.random.Generator):
    """Draw every marked body of one layout, in name order, clear of the others.

    ``poses`` maps body names to ``(position, rotation_xyzw)`` centres, ``boxes``
    to local ``(min, max)`` corners and ``markers`` the marked names to their
    ``RandomAroundSolution``. Returns the new poses of the marked bodies.
    """
    solved = {name: Solid.of(boxes[name], *pose) for name, pose in poses.items()}
    solids = dict(solved)
    drawn = {}
    for name in sorted(markers):
        # Allowances come from the solved layout, so an overlap an earlier draw made is never inherited.
        obstacles = [
            (solid, solved[name].overlap(solved[other])) for other, solid in solids.items() if other != name
        ]
        drawn[name], _ = draw_clear_pose(poses[name], markers[name], rng, boxes[name], obstacles)
        solids[name] = Solid.of(boxes[name], *drawn[name])
    return drawn


def _layout_poses(assets, layout) -> dict[Any, tuple[tuple, tuple]]:
    """Every placed non-robot asset's pose this reset: solved, or its own fixed pose."""
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.relations.placement_events import get_pose_from_layout
    from isaaclab_arena.utils.pose import Pose

    poses = {}
    for asset in assets:
        if isinstance(asset, EmbodimentBase):
            continue
        if asset in layout.positions:
            pose = get_pose_from_layout(asset, layout)
        else:
            pose = asset.get_initial_pose()
            if not isinstance(pose, Pose):
                continue
        poses[asset] = (tuple(pose.position_xyz), tuple(pose.rotation_xyzw))
    return poses


def _local_box(asset) -> tuple[list[float], list[float]]:
    box = asset.get_bounding_box()
    return box.min_point.reshape(-1, 3)[0].tolist(), box.max_point.reshape(-1, 3)[0].tolist()


def randomize_around_solution(env, env_ids, placement_seed: PlacementSeed) -> None:
    """Perturb every ``RandomAroundSolution`` object around the pose just placed."""
    from isaaclab_arena.relations.placement_events import (
        get_placement_pool,
        get_reset_placement_results,
    )
    from isaaclab_arena.relations.relations import RandomAroundSolution, get_relation
    from isaaclab_arena.utils.pose import Pose

    if env_ids is None or len(env_ids) == 0:
        return
    results = get_reset_placement_results(env)
    assets = {asset.name: asset for asset in get_placement_pool(env).objects}
    markers = {
        name: marker
        for name, asset in assets.items()
        if (marker := get_relation(asset, RandomAroundSolution)) is not None
    }
    boxes: dict[str, tuple] = {}
    resets = env.__dict__.setdefault("_random_around_solution_resets", {})
    for env_id in env_ids.tolist():
        # Per env, so a partial reset of one env never shifts another's draws.
        count = resets.get(env_id, 0)
        resets[env_id] = count + 1
        poses = {asset.name: pose for asset, pose in _layout_poses(assets.values(), results[env_id]).items()}
        for name in poses:
            if name not in boxes:
                boxes[name] = _local_box(assets[name])
        rng = np.random.default_rng((placement_seed.seed, env_id, count))
        for name, (position, rotation) in spread_layout(poses, boxes, markers, rng).items():
            assets[name].write_layout_pose_to_sim(env, env_id, Pose(position_xyz=position, rotation_xyzw=rotation))


def event_term(placer_params: Any):
    """The reset ``EventTermCfg``; add it after ``placement_reset``."""
    from isaaclab.managers import EventTermCfg

    return EventTermCfg(
        func=randomize_around_solution, mode="reset", params={"placement_seed": PlacementSeed(placer_params)}
    )


def _add_event_term(env_cfg, *, placer_params):
    # The first benchmark callback, so this lands after Arena's placement_reset and
    # before any family term (USB-C's cable reset) that re-roots on the moved parts.
    setattr(env_cfg.events, EVENT_NAME, event_term(placer_params))
    return env_cfg


def declares_spread(arena_env) -> bool:
    """Whether any scene asset carries a ``RandomAroundSolution`` marker."""
    from isaaclab_arena.relations.relations import RandomAroundSolution

    return any(
        isinstance(relation, RandomAroundSolution)
        for asset in arena_env.scene.assets.values()
        for relation in (asset.get_relations() if hasattr(asset, "get_relations") else ())
    )


def install(arena_env) -> None:
    """Apply the graph's declared spawn spread at reset, if it declares one."""
    from isaaclab_arena_vabar.physics import append_env_cfg_callback

    if declares_spread(arena_env):
        append_env_cfg_callback(arena_env, partial(_add_event_term, placer_params=arena_env.placer_params))
