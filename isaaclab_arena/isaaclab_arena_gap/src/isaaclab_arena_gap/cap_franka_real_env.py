"""Repository-owned FrankaRealEnv bridge and factory for Arena robot I/O.

The pinned GaP repos are READ-ONLY. Instead of patching
``gap.envs.franka_real_env``, this module subclasses it; the RSS adapter
(:mod:`isaaclab_arena_gap.rss_adapter.runner`) builds the bridge through
:func:`make_env` and drives it through
:class:`~isaaclab_arena_gap.bridge_robot_io.ArenaBridgeRobotIO`.

This subclass stamps per-channel command-valid flags so the Arena policy holds
observed state until the graph commands each channel. The robot model and FK
live above the wire driver, in RoboSimStudio.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import numpy as np
from gap.envs.franka_real_env import FrankaRealEnv
from gap.envs.msgpack_bridge import MsgpackNumpyServer, recv_framed, send_framed
from gap.envs.registry import EnvConfig

from isaaclab_arena_gap.fk_tcp import ROBOTIQ_TCP_OFFSET, ROBOTIQ_TCP_ROT_Z
from isaaclab_arena_gap.stale_observation import (
    BridgeIntegrationError,
    ObservationFreshnessGuard,
    StaleObservationError,
    resolve_stale_observation_timeout,
    resolve_startup_observation_timeout,
)

logger = logging.getLogger(__name__)

_RSS_PANDA_HOME_JOINTS = (
    0.0,
    0.19634954084936207,
    0.0,
    -2.617993877991494,
    0.0,
    2.941592653589793,
    0.7853981633974483,
)


def _control_frequency() -> float:
    """Read the pair's Arena control rate for Cartesian servo timing."""
    import os

    raw = os.environ.get("CAP_GAP_CONTROL_FREQUENCY_HZ", "50")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError("CAP_GAP_CONTROL_FREQUENCY_HZ must be numeric") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("CAP_GAP_CONTROL_FREQUENCY_HZ must be positive and finite")
    return value



def _arm_base_position() -> tuple[float, float, float]:
    """Read the Arena arm base expressed in GaP's shared world frame."""
    import os

    raw = os.environ.get("CAP_GAP_ARM_BASE_POSITION", "0,0,0")
    try:
        values = tuple(float(value) for value in raw.split(","))
    except ValueError as error:
        raise ValueError(
            "CAP_GAP_ARM_BASE_POSITION must contain three numbers"
        ) from error
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("CAP_GAP_ARM_BASE_POSITION must contain three finite numbers")
    return values



def _camera_names_from_env() -> list[str] | None:
    """Read the comma-separated bridge camera names, or None when unset/empty."""
    import os

    raw = os.environ.get("CAP_GAP_CAMERA_NAME", "")
    return [part.strip() for part in raw.split(",") if part.strip()] or None


class CapIdentityMsgpackServer(MsgpackNumpyServer):
    """Bridge server that correlates every reply with its current request."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Held while a request is recorded and its reply action chosen, so a
        # robot I/O exchange can publish a target atomically with its baseline.
        self.exchange_lock = threading.Lock()

    def accept(self, request: dict[str, Any]) -> dict[str, Any]:
        """Record ``request`` as the latest observation and build its reply."""
        with self.exchange_lock:
            self.latest_observation = request
            return self.identity_reply(request, self.latest_action)

    @staticmethod
    def identity_reply(
        request: dict[str, Any], action: dict[str, Any]
    ) -> dict[str, Any]:
        reply = dict(action)
        metadata = CapFrankaRealEnv._g(request, "_isaac_cap")
        if not isinstance(metadata, dict):
            return reply
        identity = {}
        for key in ("session_id", "episode_id", "request_id", "seed"):
            value = CapFrankaRealEnv._g(metadata, key)
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if value is not None:
                identity[key] = value
        if identity:
            reply["_isaac_cap"] = identity
        return reply

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                request = await recv_framed(reader)
                await send_framed(writer, self.accept(request))
        except asyncio.IncompleteReadError:
            logger.info("[CapMsgpackServer] client disconnected")
        except Exception:
            logger.exception("[CapMsgpackServer] error")


class CapFrankaRealEnv(FrankaRealEnv):
    """``FrankaRealEnv`` transport for measured joints, cameras, and targets."""

    # --- startup handshake: per-channel command-valid stamping --------------- #
    # The Arena policy must HOLD Arena's observed arm q + gripper until the graph
    # issues a real command for that channel — otherwise the hardcoded startup
    # seed (or a stale joint target during a gripper-only command) reaches the
    # robot and jolts it. We stamp the published action per-channel:
    #   arm_valid     := an arm target was set (_target_update_count > 0)
    #   gripper_valid := a gripper target was set (_cap_gripper_commanded)
    # Both are set by the robot I/O setters; the startup seed is NOT a command.

    def __init__(self, *args, **kwargs) -> None:
        # Initialize bridge flags BEFORE super(): the parent __init__ starts the
        # republisher thread, which reads them via _wire_inputs_locked.
        self._cap_gripper_commanded = False
        self._cap_republisher_failure: BridgeIntegrationError | None = None
        self._cap_terminating = threading.Event()
        stale_timeout_s = resolve_stale_observation_timeout()
        startup_timeout_s = resolve_startup_observation_timeout()
        self._cap_observation_guard = ObservationFreshnessGuard(
            stale_timeout_s,
            startup_timeout_s=startup_timeout_s,
        )
        self._cap_watch_period_s = min(1.0, max(0.01, stale_timeout_s / 10.0))
        self._cap_failure_logged = False
        import gap.envs.franka_real_env as parent_module

        original_server = parent_module.MsgpackNumpyServer
        parent_module.MsgpackNumpyServer = CapIdentityMsgpackServer
        try:
            super().__init__(*args, **kwargs)
        finally:
            parent_module.MsgpackNumpyServer = original_server
        self._cap_watch_thread = threading.Thread(
            target=self._cap_watch_observations,
            name="CapObservationFreshness",
            daemon=True,
        )
        self._cap_watch_thread.start()
        # Re-stamp the parent's hardcoded UNFLAGGED pre-seed so a read before the
        # first republish tick is correctly marked not-yet-commanded (=> hold).
        with self._target_lock:
            self._publish_action_locked()

    def _wire_stream_ready(self, wire: object) -> bool:
        if not isinstance(wire, dict):
            return False
        camera = self._find_camera_subdict(wire)
        if camera is None:
            return False
        images = self._g(camera, "images")
        if not isinstance(images, dict):
            return False
        return any(
            self._g(images, key) is not None for key in ("left_rgb", "rgb", "right_rgb")
        )

    def _check_observation_freshness(self) -> StaleObservationError | None:
        wire = self.server.latest_observation
        timestamp = self._g(wire, "timestamp") if isinstance(wire, dict) else None
        failure = self._cap_observation_guard.observe(
            timestamp,
            now_s=time.monotonic(),
            stream_ready=self._wire_stream_ready(wire),
            terminating=self._cap_terminating.is_set(),
        )
        if failure is not None and not self._cap_failure_logged:
            logger.error("[CapFrankaRealEnv] %s", failure)
            self._cap_failure_logged = True
        return failure

    def _cap_watch_observations(self) -> None:
        while not self._cap_terminating.wait(self._cap_watch_period_s):
            self._check_observation_freshness()

    def get_integration_failure(self) -> BridgeIntegrationError | None:
        """Return the sticky bridge failure for the execution wrapper."""
        return self._cap_republisher_failure or self._cap_observation_guard.failure()

    def set_robot_io_target(self, joints: np.ndarray, open_fraction: float) -> None:
        """Publish arm and gripper positions together for one control interval."""
        target = np.asarray(joints, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("robot I/O arm target must contain seven finite joints")
        if not np.isfinite(open_fraction) or not 0.0 <= open_fraction <= 1.0:
            raise ValueError("robot I/O gripper target must be in [0, 1]")
        with self._target_lock:
            self._current_joints = target.copy()
            self._target_joints = target.copy()
            self._gripper_fraction = float(open_fraction)
            self._target_gripper = float(open_fraction)
            self._target_update_count += 1
            self._cap_gripper_commanded = True
            self._publish_action_locked()
        self._last_target_update_t = time.time()

    def _wire_inputs_locked(self):
        """Return (target, gripper, arm_valid, gripper_valid); hold _target_lock.

        arm_valid := _target_update_count>0 (parent increments it AFTER
        assigning _target_joints under the lock, so count>0 implies the new
        target is already committed) — an OLD seed can never be read with a NEW
        count. gripper_valid := _cap_gripper_commanded (set under this lock)."""
        return (
            self._target_joints,
            self._target_gripper,
            bool(self._target_update_count > 0),
            bool(self._cap_gripper_commanded),
        )

    def _publish_action_locked(self) -> bool:
        """Publish the committed targets as the wire action; hold _target_lock."""
        target, gripper, arm_valid, gripper_valid = self._wire_inputs_locked()
        if target is None:
            return False
        self.server.latest_action = self._wire_action(
            target, gripper, arm_valid, gripper_valid
        )
        return True

    def _wire_action(self, target, gripper, arm_valid, gripper_valid) -> dict:
        arm_valid = bool(arm_valid)
        gripper_valid = bool(gripper_valid)
        return {
            "timestamp": time.time(),
            "left": {
                "joint_pos": np.asarray(target, dtype=np.float32).tolist(),
                "gripper": float(gripper),
            },
            "arm_valid": arm_valid,
            "gripper_valid": gripper_valid,
            "command_valid": arm_valid and gripper_valid,
        }

    def _republish_loop(self) -> None:
        # As FrankaRealEnv._republish_loop, but publishes target + both validity
        # flags under _target_lock, the same lock the robot I/O setters publish
        # under, so an older snapshot can never overwrite a newer target.
        try:
            while not self._republish_stop.is_set():
                with self._target_lock:
                    published = self._publish_action_locked()
                if published:
                    self._republish_count += 1
                time.sleep(self._republish_period)
        except Exception as error:
            logger.exception("[CapFrankaRealEnv] REPUBLISHER CRASHED")
            failure = BridgeIntegrationError(
                f"Arena action republisher stopped: {type(error).__name__}: {error}"
            )
            failure.__cause__ = error
            self._cap_republisher_failure = failure

    def close(self) -> None:
        self._cap_terminating.set()
        super().close()
        watch_thread = getattr(self, "_cap_watch_thread", None)
        if watch_thread is not None and watch_thread is not threading.current_thread():
            watch_thread.join(timeout=max(0.1, self._cap_watch_period_s * 2.0))


def make_env(
    suite_name: str = "franka_real",
    task_id: int = 0,
    camera_names: list[str] | None = None,
    enable_render: bool = False,
    *,
    port: int = 9000,
    host: str = "127.0.0.1",
    **extra,
):
    """This repository's ``franka_real`` factory: same contract as
    ``gap.envs.franka_real_env:make_env`` but builds a :class:`CapFrankaRealEnv`.

    ``EnvConfig`` uses the RSS simulator's absolute joints at 60 Hz,
    Panda home, and Robotiq TCP 0.157 m / pi/4. ``task_id``/``enable_render`` are accepted
    for registry-contract compatibility and ignored (no task switch on hardware).

    The msgpack server port honors the ``GAP_PORT`` env var when set (``gap run``
    exposes no --port flag), so GapRemotePolicy can spawn one GaP process per
    episode on a unique port.
    """
    import os as _os

    port = int(_os.environ.get("GAP_PORT", port))
    names = camera_names
    if names is None:
        names = _camera_names_from_env()
    env = CapFrankaRealEnv(camera_names=names, port=port, host=host)
    arm_base = _arm_base_position()
    config = EnvConfig(
        arm_dof=7,
        num_arms=1,
        action_mode="absolute_joints",
        control_freq=_control_frequency(),
        home_joints=_RSS_PANDA_HOME_JOINTS,
        tcp_offset=tuple(float(value) for value in ROBOTIQ_TCP_OFFSET),
        tcp_rotation_z=float(ROBOTIQ_TCP_ROT_Z),
        arm_bases=(arm_base,),
        robot_urdf_path="panda_description",
        default_cameras=tuple(env.camera_names),
        # Simulated: `is_real` only stops Connector.go_home from moving, and graphs
        # rely on that retract to clear the overhead view before perceiving again.
        is_real=False,
    )
    return env, config

