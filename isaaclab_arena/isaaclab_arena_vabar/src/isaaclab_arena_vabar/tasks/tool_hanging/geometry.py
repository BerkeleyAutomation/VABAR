"""Validated loop/rod geometry for tool-hanging goals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from isaaclab_arena_vabar.numeric import is_finite_real


_EPSILON = 1.0e-8


def _vector(name: str, values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3 or not all(is_finite_real(value) for value in values):
        raise ValueError(f"{name} must contain three finite numeric values")
    return tuple(float(value) for value in values)


def _radius(name: str, value: float, *, positive: bool) -> float:
    if not is_finite_real(value):
        raise ValueError(f"{name} must be finite")
    if (value <= 0.0) if positive else (value < 0.0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return float(value)


@dataclass
class LoopFeature:
    """A circular opening specified in the tool's local coordinate frame.

    ``enclosure_tolerance_m`` is AUTOLab's calibrated ``enclosure_tolerance``:
    extra radial room that only :func:`encloses` grants, as AUTOLab's own
    ``encloses`` atom does. ``threaded`` never reads it.
    """

    center_xyz: tuple[float, float, float]
    normal_xyz: tuple[float, float, float]
    radius_m: float
    enclosure_tolerance_m: float = 0.0

    def __post_init__(self) -> None:
        center = _vector("loop center", self.center_xyz)
        normal = _vector("loop normal", self.normal_xyz)
        magnitude = math.sqrt(sum(value * value for value in normal))
        if magnitude <= _EPSILON:
            raise ValueError("loop normal must be non-degenerate")
        object.__setattr__(self, "center_xyz", center)
        object.__setattr__(self, "normal_xyz", tuple(value / magnitude for value in normal))
        object.__setattr__(self, "radius_m", _radius("loop radius", self.radius_m, positive=True))
        object.__setattr__(
            self,
            "enclosure_tolerance_m",
            _radius("loop enclosure tolerance", self.enclosure_tolerance_m, positive=False),
        )


@dataclass
class RodFeature:
    """A cylindrical rod segment specified in the fixture's local frame."""

    start_xyz: tuple[float, float, float]
    end_xyz: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        start = _vector("rod start", self.start_xyz)
        end = _vector("rod end", self.end_xyz)
        if math.dist(start, end) <= _EPSILON:
            raise ValueError("rod segment must be non-degenerate")
        object.__setattr__(self, "start_xyz", start)
        object.__setattr__(self, "end_xyz", end)
        object.__setattr__(self, "radius_m", _radius("rod radius", self.radius_m, positive=False))


def _poses(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[-1] != 7:
        raise ValueError(f"{name} must be an [N, 7] torch tensor of XYZ + XYZW poses")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    quaternion = value[:, 3:]
    if torch.any(torch.linalg.vector_norm(quaternion, dim=-1) <= _EPSILON):
        raise ValueError(f"{name} must contain non-degenerate quaternions")
    return value


def _rotate(quaternion_xyzw: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion_xyzw / torch.linalg.vector_norm(quaternion_xyzw, dim=-1, keepdim=True)
    q_xyz, q_w = quaternion[:, :3], quaternion[:, 3:]
    return vector + 2.0 * torch.cross(q_xyz, torch.cross(q_xyz, vector, dim=-1) + q_w * vector, dim=-1)


def _world_point(local: tuple[float, float, float], poses: torch.Tensor) -> torch.Tensor:
    point = torch.as_tensor(local, device=poses.device, dtype=poses.dtype).expand(poses.shape[0], -1)
    return poses[:, :3] + _rotate(poses[:, 3:], point)


def threaded(
    loop: LoopFeature,
    rod: RodFeature,
    loop_poses: torch.Tensor,
    rod_poses: torch.Tensor,
) -> torch.Tensor:
    """Return where each world-space rod crosses the usable loop opening.

    The predicate is evaluated independently for every row in the batched pose
    tensors.  A rod's swept radius is subtracted from the loop radius, so a rod
    that only clips the opening's rim is not considered threaded.
    """

    loop_poses = _poses("loop_poses", loop_poses)
    rod_poses = _poses("rod_poses", rod_poses)
    if loop_poses.shape[0] != rod_poses.shape[0]:
        raise ValueError("loop_poses and rod_poses must have the same batch size")

    center = _world_point(loop.center_xyz, loop_poses)
    local_normal = torch.as_tensor(
        loop.normal_xyz, device=loop_poses.device, dtype=loop_poses.dtype
    ).expand(loop_poses.shape[0], -1)
    normal = _rotate(
        loop_poses[:, 3:],
        local_normal,
    )
    start = _world_point(rod.start_xyz, rod_poses)
    end = _world_point(rod.end_xyz, rod_poses)
    direction = end - start
    denominator = torch.sum(normal * direction, dim=-1)
    intersects_plane = torch.abs(denominator) > _EPSILON
    safe_denominator = torch.where(intersects_plane, denominator, torch.ones_like(denominator))
    fraction = torch.sum(normal * (center - start), dim=-1) / safe_denominator
    on_segment = (fraction >= 0.0) & (fraction <= 1.0)
    intersection = start + fraction.unsqueeze(-1) * direction
    radial_distance = torch.linalg.vector_norm(intersection - center, dim=-1)
    usable_radius = loop.radius_m - rod.radius_m
    return intersects_plane & on_segment & (usable_radius >= 0.0) & (radial_distance <= usable_radius)


def encloses(
    loop: LoopFeature,
    rod: RodFeature,
    loop_poses: torch.Tensor,
    rod_poses: torch.Tensor,
) -> torch.Tensor:
    """Return whether the finite rod centerline lies inside the loop opening.

    VABAR's ``encloses`` measures the closest point on the finite rod segment;
    unlike ``threaded``, it does not require crossing the loop plane and ignores
    the loop normal and rod radius. As in AUTOLab, the loop's
    ``enclosure_tolerance_m`` widens the accepted radius.
    """
    loop_poses = _poses("loop_poses", loop_poses)
    rod_poses = _poses("rod_poses", rod_poses)
    if loop_poses.shape[0] != rod_poses.shape[0]:
        raise ValueError("loop_poses and rod_poses must have the same batch size")

    center = _world_point(loop.center_xyz, loop_poses)
    start = _world_point(rod.start_xyz, rod_poses)
    direction = _world_point(rod.end_xyz, rod_poses) - start
    span_squared = torch.sum(direction * direction, dim=-1)
    if torch.any(span_squared <= _EPSILON * _EPSILON):
        raise ValueError("rod segment must be non-degenerate")
    fraction = torch.clamp(
        torch.sum((center - start) * direction, dim=-1) / span_squared,
        min=0.0,
        max=1.0,
    )
    closest = start + fraction.unsqueeze(-1) * direction
    offset = torch.linalg.vector_norm(closest - center, dim=-1)
    return offset <= loop.radius_m + loop.enclosure_tolerance_m
