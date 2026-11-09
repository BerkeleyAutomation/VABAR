"""Camera scenes converge their RTX frames on the reset scene before the first observation."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("isaaclab")

from isaaclab.managers import EventTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_arena_vabar.camera_contract import (
    RESET_CAMERA_WARMUP_EVENT,
    RESET_CAMERA_WARMUP_FRAMES,
    configure_camera_contract,
    install_reset_camera_warmup,
    warm_up_rtx_cameras,
)


def _reset_term(**params):
    return EventTermCfg(func=lambda env, env_ids: None, mode="reset", params=params)


@configclass
class _Events:
    reset_robot_joints: EventTermCfg = _reset_term()
    placement_reset: EventTermCfg = _reset_term()


def _env_cfg(*, cameras: bool):
    scene = SimpleNamespace(robot=object())
    if cameras:
        scene.top_camera = CameraCfg(prim_path="/World/top_camera")
    return SimpleNamespace(scene=scene, events=_Events())


def _event_names(env_cfg):
    return list(vars(env_cfg.events))


def test_camera_scenes_warm_up_after_every_other_reset_event():
    env_cfg = _env_cfg(cameras=True)

    assert configure_camera_contract(env_cfg) == 1

    assert _event_names(env_cfg)[-1] == RESET_CAMERA_WARMUP_EVENT
    term = getattr(env_cfg.events, RESET_CAMERA_WARMUP_EVENT)
    assert term.mode == "reset"
    assert term.func is warm_up_rtx_cameras
    assert term.params == {"num_frames": RESET_CAMERA_WARMUP_FRAMES}


def test_reapplying_the_contract_moves_the_warm_up_behind_later_reset_events():
    # Tool sorting applies the contract inside its own build; the runner applies it
    # again once the environment config is complete.
    env_cfg = _env_cfg(cameras=True)
    configure_camera_contract(env_cfg)
    env_cfg.events.reset_cables = _reset_term()

    configure_camera_contract(env_cfg)

    names = _event_names(env_cfg)
    assert names[-1] == RESET_CAMERA_WARMUP_EVENT
    assert names.count(RESET_CAMERA_WARMUP_EVENT) == 1


def test_camera_free_scenes_get_no_warm_up():
    env_cfg = _env_cfg(cameras=False)

    assert configure_camera_contract(env_cfg) == 0

    assert not hasattr(env_cfg.events, RESET_CAMERA_WARMUP_EVENT)


@pytest.mark.parametrize("num_frames", (-1, 1.5, None, True))
def test_warm_up_frames_must_be_a_nonnegative_integer(num_frames):
    with pytest.raises(ValueError, match="nonnegative integer"):
        install_reset_camera_warmup(_env_cfg(cameras=True), num_frames)


class _Sim:
    def __init__(self, calls):
        self._calls = calls

    def render(self):
        self._calls.append("render")


@pytest.fixture
def rtx_pump(monkeypatch):
    """Record RTX frame requests; the real helper needs a running Kit app."""
    calls = []
    module = ModuleType("isaaclab_physx.renderers.isaac_rtx_renderer_utils")
    module.ensure_isaac_rtx_render_update = lambda force=False: calls.append(
        f"rtx_frame(force={force})"
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def test_each_warm_up_frame_syncs_poses_then_forces_an_rtx_frame(rtx_pump):
    env = SimpleNamespace(has_rtx_sensors=True, sim=_Sim(rtx_pump))

    warm_up_rtx_cameras(env, env_ids=[0], num_frames=3)

    # sim.render() alone renders nothing headless; each forced update is one frame.
    assert rtx_pump == ["render", "rtx_frame(force=True)"] * 3


@pytest.mark.parametrize(
    ("has_rtx_sensors", "num_frames"), ((False, RESET_CAMERA_WARMUP_FRAMES), (True, 0))
)
def test_warm_up_renders_nothing_without_rtx_cameras_or_frames(
    rtx_pump, has_rtx_sensors, num_frames
):
    env = SimpleNamespace(has_rtx_sensors=has_rtx_sensors, sim=_Sim(rtx_pump))

    warm_up_rtx_cameras(env, env_ids=[0], num_frames=num_frames)

    assert rtx_pump == []

