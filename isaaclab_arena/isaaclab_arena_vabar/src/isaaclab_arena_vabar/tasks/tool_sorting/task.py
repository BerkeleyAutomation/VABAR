"""Pivot-independent success criteria for sorting objects into compartments."""

from __future__ import annotations

import math

import numpy as np
import torch
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena_vabar.numeric import as_torch

Bounds = tuple[float, float, float, float, float, float]


def points_in_oriented_regions(
    object_positions_w: torch.Tensor,
    region_positions_w: torch.Tensor,
    region_quaternions_w: torch.Tensor,
    bounds_xyzxyz: torch.Tensor,
) -> torch.Tensor:
    """Return whether paired points lie inside inclusive region-local boxes."""
    object_pos_region, _ = subtract_frame_transforms(
        region_positions_w,
        region_quaternions_w,
        object_positions_w,
    )
    return (
        (object_pos_region >= bounds_xyzxyz[:, :3])
        & (object_pos_region <= bounds_xyzxyz[:, 3:])
    ).all(dim=-1)


def objects_centered_in_regions_and_released(
    env,
    object_names: list[str],
    region_names: list[str],
    bounds_xyzxyz: list[Bounds],
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
    robot_asset_name: str,
    gripper_joint_name: str,
    gripper_open_position_threshold: float,
    parked_bounds_xyzxyz: Bounds | None = None,
) -> torch.Tensor:
    """Return whether all objects are released, centered, and settled now.

    An object resting inside ``parked_bounds_xyzxyz`` (environment-local, axis-aligned) is not
    part of this environment's trial and counts as done: a layout with fewer objects than the
    scene parks the spare out of view and out of reach there.
    """
    objects = tuple(env.scene[name] for name in object_names)
    regions = tuple(env.scene[name] for name in region_names)
    robot = env.scene[robot_asset_name]
    try:
        gripper_joint_index = robot.data.joint_names.index(gripper_joint_name)
    except ValueError as error:
        raise ValueError(
            f"expected gripper joint {gripper_joint_name!r} was not found"
        ) from error

    pair_results: list[torch.Tensor] = []
    for object_instance, region_instance, bounds in zip(
        objects, regions, bounds_xyzxyz, strict=True
    ):
        center_positions_w = as_torch(object_instance.data.root_com_pos_w)
        if hasattr(region_instance, "data"):
            region_positions_w = as_torch(region_instance.data.root_pos_w)
            region_quaternions_w = as_torch(region_instance.data.root_quat_w)
        else:
            region_positions_w, region_quaternions_w = (
                as_torch(value) for value in region_instance.get_world_poses()
            )
        bounds_tensor = torch.as_tensor(
            bounds,
            dtype=center_positions_w.dtype,
            device=center_positions_w.device,
        )
        centered = points_in_oriented_regions(
            center_positions_w,
            region_positions_w,
            region_quaternions_w,
            bounds_tensor.unsqueeze(0).expand(env.num_envs, -1),
        )
        linear_speed = torch.linalg.vector_norm(
            as_torch(object_instance.data.root_lin_vel_w), dim=-1
        )
        angular_speed = torch.linalg.vector_norm(
            as_torch(object_instance.data.root_ang_vel_w), dim=-1
        )
        done = (
            centered
            & (linear_speed <= linear_velocity_threshold)
            & (angular_speed <= angular_velocity_threshold)
        )
        if parked_bounds_xyzxyz is not None:
            local = center_positions_w - as_torch(env.scene.env_origins).to(center_positions_w)
            parked_box = torch.as_tensor(
                parked_bounds_xyzxyz, dtype=local.dtype, device=local.device
            )
            done = done | ((local >= parked_box[:3]) & (local <= parked_box[3:])).all(dim=-1)
        pair_results.append(done)

    gripper_position = as_torch(robot.data.joint_pos)[:, gripper_joint_index]
    gripper_open = torch.abs(gripper_position) <= gripper_open_position_threshold
    return torch.stack(pair_results, dim=0).all(dim=0) & gripper_open


class ObjectsCenteredInRegionsTask(TaskBase):
    """Require every object's center of mass to rest inside its paired region."""

    def __init__(
        self,
        object_list: list[Asset],
        region_list: list[Asset],
        bounds_xyzxyz: list[Bounds],
        linear_velocity_threshold: float = 0.01,
        angular_velocity_threshold: float = 0.05,
        robot_asset_name: str = "robot",
        gripper_joint_name: str = "left_driver_joint",
        gripper_open_position_threshold: float = 0.1,
        consecutive_success_steps: int = 10,
        episode_length_s: float = 480.0,
        task_description: str | None = None,
        parked_bounds_xyzxyz: Bounds | None = None,
    ) -> None:
        for name, value in (
            ("linear_velocity_threshold", linear_velocity_threshold),
            ("angular_velocity_threshold", angular_velocity_threshold),
            ("gripper_open_position_threshold", gripper_open_position_threshold),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if not robot_asset_name or not gripper_joint_name:
            raise ValueError("robot asset and gripper joint names must be non-empty")
        if (
            isinstance(consecutive_success_steps, bool)
            or not isinstance(consecutive_success_steps, int)
            or consecutive_success_steps <= 0
        ):
            raise ValueError("consecutive_success_steps must be a positive integer")
        self._validate_regions(object_list, region_list, bounds_xyzxyz)
        if parked_bounds_xyzxyz is not None:
            self._validate_regions([None], [None], [parked_bounds_xyzxyz])
        self.parked_bounds = (
            None
            if parked_bounds_xyzxyz is None
            else tuple(float(value) for value in parked_bounds_xyzxyz)
        )
        self.objects = list(object_list)
        self.regions = list(region_list)
        self.bounds = [
            tuple(float(value) for value in bounds) for bounds in bounds_xyzxyz
        ]
        self.linear_velocity_threshold = float(linear_velocity_threshold)
        self.angular_velocity_threshold = float(angular_velocity_threshold)
        self.robot_asset_name = robot_asset_name
        self.gripper_joint_name = gripper_joint_name
        self.gripper_open_position_threshold = float(gripper_open_position_threshold)
        self.consecutive_success_steps = consecutive_success_steps
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description
            or "Place every object's center of mass inside its paired region and let it settle.",
        )
        self.events_cfg = None
        self.metrics = [SuccessRateMetric()]
        self.termination_cfg = self.make_termination_cfg()

    @staticmethod
    def _validate_regions(
        object_list: list[Asset],
        region_list: list[Asset],
        bounds_xyzxyz: list[Bounds],
    ) -> None:
        if (
            not object_list
            or len(object_list) != len(region_list)
            or len(object_list) != len(bounds_xyzxyz)
        ):
            raise ValueError(
                "objects, regions, and bounds must be non-empty lists of the same length"
            )
        for bounds in bounds_xyzxyz:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 6:
                raise ValueError(
                    "each region bounds entry must contain exactly six values"
                )
            try:
                values = tuple(float(value) for value in bounds)
            except (TypeError, ValueError) as error:
                raise ValueError("region bounds must be numeric") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError("region bounds must be finite")
            if any(
                lower > upper
                for lower, upper in zip(values[:3], values[3:], strict=True)
            ):
                raise ValueError(
                    "region bounds lower values must not exceed upper values"
                )

    def make_termination_cfg(self) -> TaskTerminationCfg:
        settled_now = TerminationTermCfg(
            func=objects_centered_in_regions_and_released,
            params={
                "object_names": [object_.name for object_ in self.objects],
                "region_names": [region.name for region in self.regions],
                "bounds_xyzxyz": self.bounds,
                "linear_velocity_threshold": self.linear_velocity_threshold,
                "angular_velocity_threshold": self.angular_velocity_threshold,
                "robot_asset_name": self.robot_asset_name,
                "gripper_joint_name": self.gripper_joint_name,
                "gripper_open_position_threshold": self.gripper_open_position_threshold,
                "parked_bounds_xyzxyz": self.parked_bounds,
            },
        )
        settled = TrueForConsecutiveStepsCfg(
            predicate=settled_now,
            required_steps=self.consecutive_success_steps,
        )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="objects_centered_in_regions_and_settled",
                    predicate_sequence=[settled],
                )
            ],
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, _arm_mode):
        return None

    def get_metrics(self) -> list[MetricBase]:
        return self.metrics

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.objects[0],
            offset=np.array([-1.5, -1.5, 1.5]),
        )
