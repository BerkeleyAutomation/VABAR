"""A do-nothing hold policy for perf floor measurements.

Returns the robot's currently observed joint state every step (absolute
joint-position embodiments), so the episode is pure env stepping with zero
policy/bridge cost beyond the perf-trace timer itself. Used to measure
the Arena physics and runner cost floor; never part of an evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from isaaclab_arena_gap.perf_trace import PerfTrace


@dataclass
class CapHoldPolicyArgs(PolicyCfg):
    policy_device: str = "cuda"


@register_policy
class CapHoldPolicy(PolicyBase[CapHoldPolicyArgs]):
    name = "cap_hold"

    def __init__(self, config: CapHoldPolicyArgs):
        super().__init__(config)
        self._config = config
        # Runner-facing attributes every PolicyBase implementation in this
        # repo exposes (mirrors GapRemotePolicy / pi0_remote_policy).
        self.device = config.policy_device
        self.task_description: str | None = None
        self._perf = PerfTrace("hold")
        self._last_exit: float | None = None

    def get_action(self, env: Any, observation: dict[str, Any]):
        t_enter = time.perf_counter()
        sim_s = t_enter - self._last_exit if self._last_exit is not None else None
        # Build the hold row from the action manager's own terms rather than from
        # a scene entity named "robot". A bimanual embodiment has no such entity
        # -- it has left_robot and right_robot -- and its gripper is not the last
        # action element, so the single-arm assumptions do not survive it. The
        # term walk in arena_show already handles every layout the repo declares,
        # so share it instead of keeping a second copy that can drift.
        from isaaclab_arena_gap.arena_show import hold_action

        row = hold_action(env.unwrapped)
        t_exit = time.perf_counter()
        self._last_exit = t_exit
        self._perf.record("step", sim_s=sim_s, total_s=t_exit - t_enter)
        return row

    def reset(self, env_ids=None) -> None:
        self._perf.record("episode_reset", unix=time.time())
        self._last_exit = None

    def close(self) -> None:
        self._perf.close()
