"""Episode-level fraction scoring for tool-hanging."""

from __future__ import annotations

import numpy as np
import torch
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg

from .terminations import HangingGoalSpec, tool_hanging_goal_completion


class ToolHangingFractionRecorder(RecorderTerm):
    """Record the current fraction of complete tool goals just before reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.goals = cfg.goals

    def record_pre_reset(self, env_ids):
        completed = tool_hanging_goal_completion(self._env, goals=self.goals)
        return self.name, completed[env_ids].to(torch.float32).mean(dim=-1)


@configclass
class ToolHangingFractionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ToolHangingFractionRecorder
    name: str = "tool_hanging_fraction"
    goals: tuple[HangingGoalSpec, ...] = ()


def compute_tool_hanging_fraction(recorded_metric_data: list[np.ndarray]) -> float:
    """Average the terminal per-episode completion fractions."""

    if not recorded_metric_data:
        return 0.0
    values = np.concatenate(recorded_metric_data)
    if values.ndim != 1:
        raise ValueError("tool-hanging fraction samples must be one-dimensional")
    return float(np.mean(values))


class ToolHangingFractionMetric(MetricBase):
    """Report the terminal fraction of individually complete hanging goals."""

    name = "tool_hanging_fraction"
    recorder_term_name = "tool_hanging_fraction"

    def __init__(self, goals: tuple[HangingGoalSpec, ...]):
        self.goals = goals

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return ToolHangingFractionRecorderCfg(name=self.recorder_term_name, goals=self.goals)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_tool_hanging_fraction,
            params={},
            recorder_term_name=self.recorder_term_name,
        )
