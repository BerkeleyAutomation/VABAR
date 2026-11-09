"""Shared contracts and verdict logic for industrial-asset drop tests."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Resting means staying put, judged on pose rather than instantaneous velocity: MuJoCo
# contact chatter makes a resting object's velocity spike to cm/s while it moves < 0.1 mm.
MAX_REST_TRANSLATION_M = 0.001
MAX_REST_ROTATION_RAD = math.radians(1.0)


def pose_deviation(pose_a, pose_b) -> tuple[float, float]:
    """Return the translation (m) and rotation (rad) between two position+quaternion poses.

    Both quaternions must use the same component order; the angle does not depend on which.
    """
    translation = math.dist(pose_a[:3], pose_b[:3])
    dot = abs(sum(a * b for a, b in zip(pose_a[3:7], pose_b[3:7], strict=True)))
    return translation, 2.0 * math.acos(min(1.0, dot))


def settling_time(poses, dt: float) -> float | None:
    """Return the earliest time after which every pose stays at rest around the final pose."""
    if not poses or not all(math.isfinite(value) for pose in poses for value in pose):
        return None
    last_unsettled = max(
        (
            index
            for index, pose in enumerate(poses)
            if (deviation := pose_deviation(pose, poses[-1]))[0] > MAX_REST_TRANSLATION_M
            or deviation[1] > MAX_REST_ROTATION_RAD
        ),
        default=-1,
    )
    return (last_unsettled + 1) * dt


@dataclass(frozen=True)
class AssetDropObservation:
    finite_state: bool
    contact_count: int
    final_lowest_point_z: float
    tabletop_z: float
    inside_tabletop: bool
    tail_translations_m: tuple[float, ...]
    tail_rotations_rad: tuple[float, ...]

    @classmethod
    def tail_from_poses(cls, poses) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Each pose's deviation from the first pose of the window."""
        deviations = [pose_deviation(pose, poses[0]) for pose in poses]
        return tuple(d[0] for d in deviations), tuple(d[1] for d in deviations)


@dataclass(frozen=True)
class AssetDropVerdict:
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class AssetDropResult:
    registry_name: str
    physics_profile: str
    passed: bool
    failures: tuple[str, ...]
    final_lowest_point_z: float
    tabletop_z: float
    max_tail_translation_m: float
    max_tail_rotation_rad: float
    support_contact_samples: int
    settling_time_s: float | None
    mass_kg: float
    friction: str
    restitution: str
    collider: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["failures"] = list(self.failures)
        for key, item in value.items():
            if isinstance(item, float) and not math.isfinite(item):
                value[key] = None
        return value


def evaluate_asset_drop(
    observation: AssetDropObservation,
    *,
    max_penetration_m: float = 0.002,
    max_translation_m: float = MAX_REST_TRANSLATION_M,
    max_rotation_rad: float = MAX_REST_ROTATION_RAD,
) -> AssetDropVerdict:
    """Evaluate support and settling across the complete final one-second window."""

    failures: list[str] = []
    numeric_values = (
        observation.final_lowest_point_z,
        observation.tabletop_z,
        *observation.tail_translations_m,
        *observation.tail_rotations_rad,
    )
    if not observation.finite_state or not all(
        math.isfinite(v) for v in numeric_values
    ):
        failures.append("state became non-finite")
    if observation.contact_count <= 0:
        failures.append("asset never contacted the table")
    if not observation.inside_tabletop:
        failures.append("asset left the tabletop footprint")
    if observation.final_lowest_point_z < observation.tabletop_z - max_penetration_m:
        failures.append(
            f"asset penetrated the tabletop by more than {max_penetration_m} m"
        )
    if (
        not observation.tail_translations_m
        or max(observation.tail_translations_m) > max_translation_m
    ):
        failures.append(f"asset moved more than {max_translation_m} m in the final second")
    if (
        not observation.tail_rotations_rad
        or max(observation.tail_rotations_rad) > max_rotation_rad
    ):
        failures.append(
            f"asset turned more than {math.degrees(max_rotation_rad):g} deg in the final second"
        )
    return AssetDropVerdict(passed=not failures, failures=tuple(failures))


def validate_drop_evidence(
    result_path: Path, media_root: Path, expected_asset: str
) -> None:
    """Require one fresh drop process to have persisted complete report inputs."""
    payload = json.loads(result_path.read_text())
    result = payload.get("result")
    if (
        payload.get("asset") != expected_asset
        or not isinstance(result, dict)
        or result.get("registry_name") != expected_asset
    ):
        raise ValueError(f"incomplete result payload for asset {expected_asset}")

    for suffix in (".mp4", ".thumbnail.png"):
        media = media_root / f"{expected_asset}{suffix}"
        if not media.is_file() or media.stat().st_size == 0:
            raise ValueError(f"missing media for {expected_asset}: {media}")
