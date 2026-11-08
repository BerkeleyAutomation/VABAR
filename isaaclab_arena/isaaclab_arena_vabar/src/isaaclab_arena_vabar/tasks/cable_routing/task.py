"""Arena task for routing a cable through its guides into the port."""

from __future__ import annotations

from typing import Any

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from .terminations import (
    CableWorkHand,
    TerminatedCableGoal,
    terminated_cable_route_success,
)


class CableRoutingTask(TaskBase):
    """Minimal task that keeps the RSS cable-routing scene steppable."""

    def __init__(
        self,
        *,
        scene_cfg: Any,
        events_cfg: Any,
        peg_names: tuple[str, ...],
        port_name: str,
        goal: TerminatedCableGoal,
        segment_half_lengths: tuple[float, ...],
        seat_frame_z: float,
        work_hand: CableWorkHand,
        episode_length_s: float,
        task_description: str,
        viewer_eye: tuple[float, float, float],
        viewer_lookat: tuple[float, float, float],
    ) -> None:
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description,
        )
        self._scene_cfg = scene_cfg
        self._events_cfg = events_cfg
        self._viewer_eye = viewer_eye
        self._viewer_lookat = viewer_lookat
        success = TerminationTermCfg(
            func=terminated_cable_route_success,
            params={
                "peg_names": peg_names,
                "port_name": port_name,
                "goal": goal,
                "segment_half_lengths": segment_half_lengths,
                "seat_frame_z": seat_frame_z,
                "work_hand": work_hand,
            },
        )
        self._terminations_cfg = TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="cable_routing",
                    predicate_sequence=[success],
                )
            ],
        )

    def get_scene_cfg(self):
        return self._scene_cfg

    def get_termination_cfg(self):
        return self._terminations_cfg

    def get_events_cfg(self):
        return self._events_cfg

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=self._viewer_eye, lookat=self._viewer_lookat)
