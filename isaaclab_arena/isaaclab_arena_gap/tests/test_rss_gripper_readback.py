"""The RSS shadow's gripper reading follows Arena's measured opening, as RoboSimStudio reads its own hand.

``robot.open_gripper`` / ``robot.close_gripper`` return ``1 - Gripper.closed_fraction``,
which RoboSimStudio measures off the shadow's finger coordinates. Arena's crank YAM
declares its driven finger the mimic of its twin -- the reverse of RSS's hand spec --
and the sync once rewrote the measured finger from the never-written twin, so every
reading came back 0.0137 whatever the hand did.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

FINGER_OPEN = 0.0475
"""RSS's ``yam_linear_arena`` finger travel [m]; its span is twice this."""


class _Array:
    def __init__(self, values, dtype=np.float32):
        self._values = np.asarray(values, dtype=dtype)

    def numpy(self):
        return self._values

    def assign(self, values):
        self._values = np.asarray(values, dtype=self._values.dtype).copy()


@pytest.fixture
def crank_yam_env(monkeypatch):
    """``ArenaEnv`` over a shadow with the Arena crank YAM's finger layout, one arm.

    Joint coordinates as Arena's USD builds them: six arm joints, the driven
    ``left_finger`` at 6 and its twin ``right_finger`` at 7, joined by the USD's
    mimic with the *driven* finger as follower: ``left = 0 - 1 * right``.
    """
    monkeypatch.setitem(sys.modules, "newton", SimpleNamespace(eval_fk=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "robosimstudio", ModuleType("robosimstudio"))
    module = "isaaclab_arena_gap.rss_adapter.arena_env"
    real = sys.modules.pop(module, None)
    arena_env = importlib.import_module(module)

    model = SimpleNamespace(
        joint_q_start=_Array(np.arange(8), np.int32),
        constraint_mimic_count=1,
        constraint_mimic_joint0=_Array([6], np.int32),
        constraint_mimic_joint1=_Array([7], np.int32),
        constraint_mimic_coef0=_Array([0.0]),
        constraint_mimic_coef1=_Array([-1.0]),
        constraint_mimic_enabled=_Array([True], bool),
    )
    # Whatever the twin last held; the wire never writes it.
    state = SimpleNamespace(joint_q=_Array([0.0] * 6 + [0.0, -0.00065]), joint_qd=None)
    arm = SimpleNamespace(arm_coords=tuple(range(6)), gripper_coords=(6,),
                          gripper_open_q=(FINGER_OPEN,), gripper_closed_q=(0.0,))
    sim = SimpleNamespace(model=model, state=state, handles=SimpleNamespace(arms={"left": arm}),
                          pipeline=SimpleNamespace(mark_dirty=lambda: None))
    env = object.__new__(arena_env.ArenaEnv)
    env._sim = sim
    env._arm_index = {"left": list(range(6))}
    env._grip_index = {"left": 6}
    env._passive = {}
    env._camera_names = ()
    yield env
    # The stubbed module must not outlive the stubs it was imported against.
    if real is None:
        sys.modules.pop(module, None)
    else:
        sys.modules[module] = real


def _rss_position(q) -> float:
    """What RSS's ``Gripper`` reports for this hand: the finger pair's gap over its span."""
    return float(np.clip((q[6] - q[7]) / (2 * FINGER_OPEN), 0.0, 1.0))


@pytest.mark.parametrize("opened", [1.0, 0.3, 0.0])
def test_crank_yam_reading_follows_arenas_opening(crank_yam_env, opened):
    env = crank_yam_env
    env._sync(SimpleNamespace(joint_positions=np.array([0.0] * 6 + [opened]), timestamp_s=0.0, cameras={}))
    q = env._sim.state.joint_q.numpy()
    assert q[6] == pytest.approx(opened * FINGER_OPEN)
    assert q[7] == pytest.approx(-opened * FINGER_OPEN)
    assert _rss_position(q) == pytest.approx(opened, abs=1e-6)


def test_an_unmeasured_follower_still_follows_its_leader(crank_yam_env):
    env = crank_yam_env
    model = env._sim.model
    # The other orientation: the measured finger leads, the twin follows.
    model.constraint_mimic_joint0, model.constraint_mimic_joint1 = _Array([7], np.int32), _Array([6], np.int32)
    env._sync(SimpleNamespace(joint_positions=np.array([0.0] * 6 + [0.6]), timestamp_s=0.0, cameras={}))
    assert _rss_position(env._sim.state.joint_q.numpy()) == pytest.approx(0.6, abs=1e-6)


@pytest.mark.parametrize(
    "variant, hand, kinematics",
    [("crank_4310_wide", None, None), ("linear_4310", "linear_4310", "yam_i2rt")],
)
@pytest.mark.parametrize("opened", [1.0, 0.3, 0.0])
def test_arena_yam_shadow_reads_the_gripper_as_robosimstudio_does(variant, hand, kinematics, opened):
    """On RoboSimStudio's own model of Arena's YAM USD (AUTOLab's v1.0 environment only)."""
    vabar = pytest.importorskip("vabar")
    pytest.importorskip("newton")

    vabar.install()
    from isaaclab_arena_gap.rss_adapter.arena_env import ArenaEnv
    from isaaclab_arena_gap.rss_adapter.embodiments import YAM_HAND, YAM_KINEMATICS, yam_bimanual

    mounts = {"left": (0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0), "right": (0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 1.0)}
    spec = yam_bimanual(mounts, variant, hand or YAM_HAND, kinematics or YAM_KINEMATICS)
    arms = {arm: [f"{arm}/yam_joint{i}" for i in range(1, 7)] for arm in mounts}
    grips = {arm: f"{arm}/gripper_open_fraction" for arm in mounts}
    io = SimpleNamespace(joint_names=[*arms["left"], *arms["right"], *grips.values()])
    task = SimpleNamespace(evaluator=lambda _task: None, cameras=(), instruction="")
    env = ArenaEnv(io, spec, task, arm_joints=arms, gripper_joint=grips, work_arm="left", support_arm="right")
    env._sync(SimpleNamespace(joint_positions=np.array([0.0] * 12 + [opened, opened]), timestamp_s=0.0, cameras={}))
    proprio = env._sim._observe_proprio()
    for arm in mounts:
        # GapEnvFacade's fraction, which robot.open_gripper / close_gripper return as ``position``.
        assert 1.0 - float(proprio[f"{arm}/gripper"]) == pytest.approx(opened, abs=1e-4)
