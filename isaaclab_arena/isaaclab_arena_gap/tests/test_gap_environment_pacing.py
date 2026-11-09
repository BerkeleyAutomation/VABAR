"""GaP environment steps stay synchronized with Arena observations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("gap")

from isaaclab_arena_gap.cap_franka_real_env import (
    _RSS_PANDA_HOME_JOINTS,
    CapIdentityMsgpackServer,
    _control_frequency,
)
from isaaclab_arena_gap.cap_franka_real_env import (
    _arm_base_position as _fr3_arm_base_position,
)
from isaaclab_arena_gap.cap_franka_real_env import (
    make_env as make_fr3_env,
)
from isaaclab_arena_gap.stale_observation import (
    DEFAULT_STARTUP_OBSERVATION_TIMEOUT_S,
    ObservationFreshnessGuard,
    StaleObservationError,
    resolve_startup_observation_timeout,
)


def test_slow_first_simulator_step_uses_startup_freshness_timeout() -> None:
    guard = ObservationFreshnessGuard(timeout_s=60.0, startup_timeout_s=180.0)

    assert guard.observe(10.0, now_s=0.0, stream_ready=True) is None
    assert guard.observe(10.0, now_s=118.0, stream_ready=True) is None
    assert guard.observe(11.0, now_s=119.0, stream_ready=True) is None

    assert guard.observe(11.0, now_s=178.9, stream_ready=True) is None
    failure = guard.observe(11.0, now_s=179.0, stream_ready=True)
    assert isinstance(failure, StaleObservationError)
    assert failure.timeout_s == 60.0


def test_startup_freshness_timeout_is_configurable() -> None:
    assert resolve_startup_observation_timeout({}) == (
        DEFAULT_STARTUP_OBSERVATION_TIMEOUT_S
    )
    assert resolve_startup_observation_timeout(
        {"CAP_STARTUP_OBSERVATION_TIMEOUT_S": "240"}
    ) == pytest.approx(240.0)
    with pytest.raises(ValueError, match="finite positive"):
        resolve_startup_observation_timeout(
            {"CAP_STARTUP_OBSERVATION_TIMEOUT_S": "nan"}
        )


def test_arm_base_position_is_read_from_the_managed_process_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAP_GAP_ARM_BASE_POSITION", "-0.5,-0.1,0.912")

    assert _fr3_arm_base_position() == (-0.5, -0.1, 0.912)


def test_fr3_control_frequency_is_pair_owned(monkeypatch) -> None:
    monkeypatch.delenv("CAP_GAP_CONTROL_FREQUENCY_HZ", raising=False)
    assert _control_frequency() == 50.0
    monkeypatch.setenv("CAP_GAP_CONTROL_FREQUENCY_HZ", "60")
    assert _control_frequency() == 60.0


def test_fr3_environment_exposes_arm_base_to_world_frame_connector(
    monkeypatch,
) -> None:
    fake_env = SimpleNamespace(camera_names=["left"])
    monkeypatch.setenv("CAP_GAP_ARM_BASE_POSITION", "-0.5,-0.1,0.912")
    monkeypatch.setattr(
        "isaaclab_arena_gap.cap_franka_real_env.CapFrankaRealEnv",
        lambda **_kwargs: fake_env,
    )

    env, config = make_fr3_env(camera_names=["left"])

    assert env is fake_env
    assert config.arm_bases == ((-0.5, -0.1, 0.912),)
    assert config.control_freq == 50.0
    assert config.home_joints == _RSS_PANDA_HOME_JOINTS


def test_fr3_scenes_start_where_the_connector_homes() -> None:
    """AUTOLab's runtime reads the resting tool height from the arm's pose when first asked."""
    from pathlib import Path

    import yaml

    scenes = Path(__file__).resolve().parents[2] / "isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments"
    for scene in sorted(scenes.glob("*/*.yaml")):
        embodiment = (yaml.safe_load(scene.read_text(encoding="utf-8")) or {}).get("embodiment", {})
        if embodiment.get("registry_name") == "vabar__fr3_robotiq_2f85":
            assert embodiment["params"]["initial_joint_pose"] == pytest.approx(_RSS_PANDA_HOME_JOINTS), scene


def test_cap_server_echoes_current_request_identity_not_stale_action_identity() -> None:
    request = {
        b"_isaac_cap": {
            b"session_id": b"fleet/env-003",
            b"episode_id": 2,
            b"request_id": 17,
            b"seed": 99,
        }
    }
    action = {
        "left": {"joint_pos": [0.0] * 7, "gripper": 1.0},
        "_isaac_cap": {"session_id": "stale", "request_id": 16},
    }

    reply = CapIdentityMsgpackServer.identity_reply(request, action)

    assert reply["_isaac_cap"] == {
        "session_id": "fleet/env-003",
        "episode_id": 2,
        "request_id": 17,
        "seed": 99,
    }
    assert action["_isaac_cap"]["session_id"] == "stale"
