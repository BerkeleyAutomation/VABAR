# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arena task configuration for USB-C insertion into a held bulkhead."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from isaaclab.managers import TerminationTermCfg
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from .geometry import Bore
from .terminations import UsbcGoalSpec, UsbcWorkHandSpec, usbc_insertion_seated


class UsbcInsertionTask(TaskBase):
    """Score a free plug against a free receptacle's declared bore."""

    def __init__(
        self,
        plug: Asset,
        bulkhead: Asset,
        bore: Mapping[str, Any],
        goal: Mapping[str, Any],
        work_hand: Mapping[str, Any],
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        super().__init__(
            episode_length_s=episode_length_s, task_description=task_description
        )
        self.plug = plug
        self.bulkhead = bulkhead
        self.goal_spec = UsbcGoalSpec(
            bore=Bore(
                mouth=tuple(bore["mouth"]),
                axis=tuple(bore["axis"]),
                depth=float(bore["depth"]),
                half_extents=tuple(bore["half_extents"]),
                tip_offset=float(bore["tip_offset"]),
            ),
            depth_min=float(goal["depth_min"]),
            lateral_max=float(goal["lateral_max"]),
            tilt_max=(
                float(goal["tilt_max"]) if goal.get("tilt_max") is not None else None
            ),
            speed_max=float(goal["speed_max"]),
            tcp_distance_min=float(goal["tcp_distance_min"]),
            require_released=bool(goal["require_released"]),
            plug_name=plug.name,
            port_name=bulkhead.name,
        )
        self.work_hand_spec = UsbcWorkHandSpec(
            robot_name=str(work_hand["robot_name"]),
            tcp_body_name=str(work_hand["tcp_body_name"]),
            tcp_offset_xyz=tuple(float(value) for value in work_hand["tcp_offset_xyz"]),
            gripper_action_name=str(work_hand["gripper_action_name"]),
            gripper_joint_name=str(work_hand["gripper_joint_name"]),
            span_m=float(work_hand["span_m"]),
            open_joint_m=float(work_hand["open_joint_m"]),
            grasp_width_m=float(work_hand["grasp_width_m"]),
        )
        self.termination_cfg = TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="usbc_insertion",
                    predicate_sequence=[
                        TerminationTermCfg(
                            func=usbc_insertion_seated,
                            params={
                                "goal": self.goal_spec,
                                "work_hand": self.work_hand_spec,
                            },
                        )
                    ],
                )
            ],
        )

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return self.termination_cfg

    def get_events_cfg(self) -> Any:
        return None

    def get_mimic_env_cfg(self, arm_mode) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
