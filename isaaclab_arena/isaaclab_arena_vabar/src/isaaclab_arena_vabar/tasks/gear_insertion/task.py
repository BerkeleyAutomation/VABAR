"""Arena representation of the AUTOLab gear-mesh task families."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from .terminations import gear_mesh_success


@configclass
class EventsCfg:
    reset_all: EventTermCfg = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


class GearMeshTask(TaskBase):
    """Seat every sampled gear, press the button, and prove the train works."""

    def __init__(
        self,
        board: Asset,
        gear: Asset | None = None,
        gears: list[Asset] | None = None,
        gear_teeth: int = 20,
        target_offsets_xyz: Sequence[Sequence[float]] | None = None,
        episode_length_s: float = 100.0,
        task_description: str | None = None,
    ) -> None:
        gear_assets = tuple(gears or (() if gear is None else (gear,)))
        if not gear_assets:
            raise ValueError("gear mesh requires at least one loose gear")
        if len({asset.name for asset in gear_assets}) != len(gear_assets):
            raise ValueError("gear mesh requires unique loose gear names")
        offsets = tuple(
            tuple(float(value) for value in offset)
            for offset in (target_offsets_xyz or ((0.032, 0.0, 0.008),))
        )
        if len(offsets) != len(gear_assets) or any(len(offset) != 3 for offset in offsets):
            raise ValueError("gear mesh requires one 3D station offset per gear")
        teeth = tuple(int(gear_teeth) for _ in gear_assets)

        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description
            or "put the gear on its peg so it meshes with the pinion, then press the red button at the near end of the board to start it",
        )
        self.board = board
        self.gears = gear_assets
        self.gear = gear_assets[0]
        self.events_cfg = EventsCfg()
        self._success_cfg = TerminationTermCfg(
            func=gear_mesh_success,
            params={
                "board_asset_cfg": SceneEntityCfg(board.name),
                "gear_asset_cfgs": [SceneEntityCfg(asset.name) for asset in gear_assets],
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "tcp_body_name": "robotiq_base",
                "tcp_offset_xyz": (0.0, 0.0, 0.157),
                "target_offsets_xyz": offsets,
                "button_latch_m": 0.005,
                "drive_speed_rad_s": 4.0,
                "spin_fraction": 0.3,
                "gear_teeth": teeth,
                "spin_window_s": 0.5,
                "xy_threshold_m": 0.005348669,
                "z_threshold_m": 0.008,
                "hold_time_s": 1.0,
            },
        )

    def set_gear_teeth(self, gear_teeth: int) -> None:
        """Update the expected driven speed for the selected asset family."""
        self.configure_layout(
            (int(gear_teeth),),
            self._success_cfg.params["target_offsets_xyz"],
        )

    def configure_layout(
        self,
        gear_teeth: Sequence[int],
        target_offsets_xyz: Sequence[Sequence[float]],
    ) -> None:
        """Configure the selected board's station classes and local targets."""
        teeth = tuple(int(value) for value in gear_teeth)
        offsets = tuple(tuple(float(value) for value in offset) for offset in target_offsets_xyz)
        if len(teeth) != len(self.gears) or len(offsets) != len(self.gears):
            raise ValueError("selected gear-mesh layout does not match the scene gear count")
        if any(value not in (16, 20, 24) for value in teeth):
            raise ValueError("gear-mesh station teeth must be 16, 20, or 24")
        if any(len(offset) != 3 for offset in offsets):
            raise ValueError("gear-mesh station offsets must be 3D")
        self._success_cfg.params["gear_teeth"] = teeth
        self._success_cfg.params["target_offsets_xyz"] = offsets

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="gear_mesh",
                    predicate_sequence=[self._success_cfg],
                )
            ],
        )

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
