"""Bimanual Arena transport."""

from __future__ import annotations

import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest


def test_transport_preserves_role_order_and_channel_validity() -> None:
    pytest.importorskip("gap")
    from isaaclab_arena_gap.cap_bimanual_yam_env import CapBimanualYamEnv

    env = object.__new__(CapBimanualYamEnv)
    env.arm_names = ("right", "left")
    env._target_lock = threading.Lock()
    env._arm_targets = [np.ones(6), np.arange(6, dtype=float)]
    env._gripper_targets = [0.2, 0.6]
    env._arm_valid = [True, False]
    env._gripper_valid = [True, False]

    with env._target_lock:
        wire = env._wire_action(*env._wire_inputs_locked())
    assert wire["right"]["joint_pos"] == [1.0] * 6
    assert wire["right"]["gripper"] == pytest.approx(0.2)
    assert wire["right"]["arm_valid"] and wire["right"]["gripper_valid"]
    assert wire["left"]["joint_pos"] == list(range(6))
    assert not wire["left"]["arm_valid"]
    assert not wire["left"]["gripper_valid"]


def test_factory_chooses_work_arm_without_constructing_ik(monkeypatch) -> None:
    pytest.importorskip("gap")
    from isaaclab_arena_gap import cap_bimanual_yam_env as runtime

    monkeypatch.setenv(
        "CAP_GAP_ARM_BASE_POSES",
        json.dumps([[[0.3025, y, 0.81], [0, 0, 0, 1]] for y in (0.17, -0.17)]),
    )
    monkeypatch.setenv("CAP_GAP_WORK_ARM", "right")
    monkeypatch.setattr(
        runtime, "CapBimanualYamEnv", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    env, config = runtime.make_env()
    assert env.arm_names == ("right", "left")
    assert not hasattr(env, "ik")
    assert config.control_freq == runtime._control_frequency()


def test_hold_action_clamps_measured_gripper_overshoot(monkeypatch):
    if sys.version_info < (3, 12):  # noqa: UP036 - GaP's test interpreter is 3.11.
        pytest.skip("IsaacLab Arena requires Python 3.12")
    pytest.importorskip("isaaclab_arena")
    from isaaclab_arena_gap.robot_adapter import YamAdapter

    adapter = YamAdapter()
    monkeypatch.setattr(
        adapter,
        "_state",
        lambda _env, name, _env_id: (
            np.zeros(6),
            1.001 if name == "left_robot" else -0.001,
        ),
    )
    monkeypatch.setattr(adapter, "action_dim_for", lambda *_: 14)
    action = adapter.hold_action(object())
    assert action[6] == 0.0
    assert action[13] == 1.0
