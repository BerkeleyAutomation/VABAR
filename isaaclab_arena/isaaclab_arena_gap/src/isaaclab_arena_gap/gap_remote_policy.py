"""GaP as an Isaac Lab Arena `PolicyBase` remote policy.

This lets Arena's `eval_runner`/`policy_runner` drive, score, record, and sweep GaP
like any other policy. The managed policy starts one GaP process per episode on
an ephemeral loopback port and exchanges one observation/action per Arena step:

    AUTOLab graph via the RSS adapter (managed, `scripts/run_rss_graph.sh`)
        ▲  obs frame (joints+grip, rgb, depth, K, pose_mat)   │  action {joint_pos, gripper}
        │                                                      ▼
    GapRemotePolicy.get_action(env, obs)  ── one send/recv per env.step ──▶ action tensor

Mirrors `isaaclab_arena_openpi/policy/pi0_remote_policy.py` (a remote PolicyBase
talking to an out-of-process inference server), but GaP returns ONE action per
step (no open-loop chunk replay), so there is no per-env chunk cache.

This module owns the Arena-facing config and the fleet of per-environment
sessions. ``gap_session`` owns one GaP process and episode lifecycle,
``gap_process`` spawns it, ``gap_wire`` holds the adapter interface and reply
validation, and ``msgpack_transport`` the socket protocol. The concrete FR3 and
YAM observation/action adapters live in ``robot_adapter``.
"""

from __future__ import annotations

import atexit
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

# Arena is only importable inside the Arena venv (where eval_runner runs).
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from isaaclab_arena_gap.gap_process import apply_adapter_tool_env as _apply_adapter_tool_env
from isaaclab_arena_gap.gap_process import default_spawn_gap as _default_spawn_gap
from isaaclab_arena_gap.gap_session import GapSession, derive_session_seed
from isaaclab_arena_gap.gap_wire import GapObsAdapter
from isaaclab_arena_gap.msgpack_transport import GapMsgpackClient
from isaaclab_arena_gap.robot_profiles import BIMANUAL_YAM_PROFILES, ROBOT_PROFILES

__all__ = [
    "GapObsAdapter",
    "GapRemotePolicy",
    "GapRemotePolicyArgs",
    "_apply_adapter_tool_env",
    "_default_spawn_gap",
]


@dataclass
class GapRemotePolicyArgs(PolicyCfg):
    gap_host: str = "127.0.0.1"
    gap_port: int = 9000  # used only when auto_spawn=False (connect to external GaP)
    policy_device: str = "cuda"
    robot_profile: str = "fr3"
    gap_control_frequency_hz: float = 50.0
    # Optional physical-arm order for a multi-arm GaP wire. The first entry is
    # wire ``left`` / arm 0, the second is wire ``right`` / arm 1.
    gap_arms: tuple[str, ...] = ()
    # Arena camera keys and the aliases announced to the graph.
    gap_camera: str = ""
    gap_bridge_camera: str = ""
    gap_wrist_camera: str = ""
    gap_wrist_bridge_camera: str = "eye_in_hand"
    gap_aux_camera: str = ""
    gap_aux_bridge_camera: str = ""
    gap_extra_cameras: dict[str, str] = field(default_factory=dict)

    # Pair-owned ``rss_*`` graph adaptation (graph_adaptations/) and workspace measurements.
    gap_adaptation: str = ""
    gap_inputs: dict[str, Any] = field(default_factory=dict)
    gap_workspace: dict[str, float] = field(default_factory=dict)
    gap_work_arm: str = ""
    # Static fixtures the graph's task document names, mapped to the Arena scene
    # assets standing for them (``{peg0: peg_0, ...}``). Their poses are part of
    # the static scene layout, which AUTOLab's compiled scene hands its tools
    # even under ``--no-privileged``; only kinematic fixtures are accepted.
    gap_scene_statics: dict[str, str] = field(default_factory=dict)

    # Lazy, policy-owned per-episode subprocess lifecycle:
    # Default False preserves the external-launch flow (lazy connect to a fixed
    # port); policy-owned jobs set auto_spawn=true explicitly.
    auto_spawn: bool = False
    # This bridge always stamps arm/gripper validity. Missing flags => HOLD
    # (never apply an unflagged hardcoded seed). Set false only for legacy
    # graphs that don't stamp (explicit opt-out).
    strict_command_valid: bool = True
    gap_graph: str = ""
    # Launcher, relative to the repository root or absolute; "" runs
    # scripts/run_rss_graph.sh (gap_process.DEFAULT_LAUNCHER), the only GaP runtime.
    gap_launcher: str = ""
    # Cold cuRobo initialization can block a reply for longer than the generic
    # transport default, especially on a freshly provisioned worker.
    gap_io_timeout_s: float | None = 120.0
    # Optional mapper-owned path for lifecycle attribution. Arena's episode
    # JSONL remains authoritative for benchmark success.
    bridge_status_path: str = ""
    graph_result_path: str = ""
    # Let Arena's own success termination fire before ending a graph-complete
    # episode. This is measured in policy/env steps, not wall-clock time.
    graph_success_settle_s: float = 2.0
    # Some tasks score the settled scene even when the graph declares failure.
    graph_failure_settle_s: float = 0.0
    # Consumed by Arena's standalone policy runner during shutdown.
    remote_kill_on_exit: bool = False

    # Fleet coordination. Each Arena slot owns an independent GaP process,
    # socket, graph session, and episode lifecycle.
    num_envs: int = 1
    gap_fleet_id: str = ""
    gap_base_seed: int = 0
    one_episode_per_env: bool = False

    # Resolved per-session values. Fleet construction fills these before a
    # child is spawned; keeping them on the config makes spawn_fn an observable
    # test seam without changing its established two-argument interface.
    gap_env_id: int = 0
    gap_session_id: str = ""
    gap_episode_id: int = 0
    gap_session_seed: int = 0
    gap_trace_dir: str = ""
    gap_fleet_session: bool = False
    # Keep this OmegaConf-compatible: nested tuple element annotations are not
    # supported by Arena's typed experiment loader. Runtime values still carry
    # the same ``[position, quaternion]`` pair for each arm.
    gap_arm_base_poses: list[list[list[float]]] = field(default_factory=list)
    gap_arm_base_position: list[float] | None = None
    gap_arm_base_positions: list[list[float]] | None = None
    # ``gap_scene_statics``' measured poses, by fixture name: [x, y, z, qx, qy, qz, qw].
    gap_scene_static_poses: dict[str, list[float]] = field(default_factory=dict)


def _validate_config(config: GapRemotePolicyArgs) -> dict[str, float]:
    """Reject an unusable config, normalize numeric fields, return the workspace."""
    if config.auto_spawn and config.gap_host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            f"auto_spawn=True spawns GaP locally, but gap_host={config.gap_host!r} is not "
            f"loopback. Use a loopback host for auto_spawn, or set auto_spawn=False to "
            f"connect to an externally-launched GaP."
        )
    if config.auto_spawn and not config.gap_graph:
        raise ValueError("auto_spawn=True requires an explicit gap_graph")
    for name in ("graph_success_settle_s", "graph_failure_settle_s"):
        value = getattr(config, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if isinstance(config.num_envs, bool) or int(config.num_envs) < 1:
        raise ValueError("num_envs must be a positive integer")
    if config.num_envs > 1 and not config.auto_spawn:
        raise ValueError("multi-env GaP requires auto_spawn=True")
    if config.robot_profile not in ROBOT_PROFILES:
        raise ValueError(
            f"unsupported robot_profile {config.robot_profile!r}; expected one of {ROBOT_PROFILES}"
        )
    if not isinstance(config.gap_workspace, dict):
        raise ValueError("gap_workspace must be a mapping")
    if not isinstance(config.gap_scene_statics, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and k and v
        for k, v in config.gap_scene_statics.items()
    ):
        raise ValueError("gap_scene_statics must map fixture names to Arena asset names")
    config.gap_control_frequency_hz = float(config.gap_control_frequency_hz)
    if (
        not np.isfinite(config.gap_control_frequency_hz)
        or config.gap_control_frequency_hz <= 0.0
    ):
        raise ValueError("gap_control_frequency_hz must be positive and finite")
    try:
        workspace = {
            str(key): float(value) for key, value in config.gap_workspace.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("gap_workspace values must be numeric") from exc
    if not all(np.isfinite(value) for value in workspace.values()):
        raise ValueError("gap_workspace values must be finite")
    return workspace


@register_policy
class GapRemotePolicy(PolicyBase[GapRemotePolicyArgs]):
    """GaP driven through Arena's eval harness as a remote PolicyBase.

    Canonical jobs let the policy start GaP per episode; legacy configurations
    may connect to an already-running server. Each Arena environment owns one
    independent ``GapSession`` (GaP process and graph session); with
    ``num_envs > 1`` only their request/reply waits are coordinated
    concurrently.
    """

    name = "gap_remote"
    config_class = GapRemotePolicyArgs

    def __init__(
        self,
        config: GapRemotePolicyArgs,
        adapter: GapObsAdapter | None = None,
        *,
        spawn_fn=None,
        client_factory=None,
    ) -> None:
        super().__init__(config)
        from isaaclab_arena_gap.external_policy_termination import (
            configure_external_policy_termination,
        )

        configure_external_policy_termination(config.num_envs)
        workspace = _validate_config(config)
        self._config = config
        self._fleet = config.num_envs > 1
        self._fleet_id = config.gap_fleet_id or uuid.uuid4().hex
        env_id = int(config.gap_env_id)
        config.gap_fleet_session = self._fleet
        config.gap_fleet_id = self._fleet_id
        config.gap_session_id = (
            config.gap_session_id or f"{self._fleet_id}/env-{env_id:03d}"
        )
        config.gap_session_seed = derive_session_seed(
            config.gap_base_seed, env_id, int(config.gap_episode_id)
        )
        adapter = adapter if adapter is not None else _resolve_robot_adapter(config)
        self.device = config.policy_device
        self.task_description: str | None = None
        # Injectable for tests; defaults spawn a real subprocess + msgpack client.
        if spawn_fn is None:
            spawn_fn = _default_spawn_gap
        if client_factory is None:

            def client_factory(host, port):
                return GapMsgpackClient(
                    host, port, io_timeout_s=config.gap_io_timeout_s
                )

        self._atexit_registered = False

        session_configs = [config] + [
            replace(
                config,
                num_envs=1,
                gap_env_id=child_env_id,
                gap_session_id=f"{self._fleet_id}/env-{child_env_id:03d}",
                gap_episode_id=0,
                gap_session_seed=derive_session_seed(
                    config.gap_base_seed, child_env_id, 0
                ),
                gap_fleet_session=True,
            )
            for child_env_id in range(1, config.num_envs)
        ]
        # Shared by the sessions so concurrently assigned ports stay distinct.
        self._fleet_allocated_ports: set[int] = set()
        self._sessions = [
            GapSession(
                session_config,
                adapter=adapter,
                workspace=workspace,
                spawn_fn=spawn_fn,
                client_factory=client_factory,
                identity_required=self._fleet,
                allocated_ports=self._fleet_allocated_ports,
            )
            for session_config in session_configs
        ]
        self._fleet_retired: set[int] = set()
        self._fleet_initialized = False
        self._fleet_executor: ThreadPoolExecutor | None = None
        if self._fleet:
            self._fleet_executor = ThreadPoolExecutor(
                max_workers=config.num_envs,
                thread_name_prefix="gap-session",
            )
        # Perf knob: spawn+connect the GaP process at construction instead of
        # on the first policy step, overlapping GaP startup with the
        # remaining Arena env preparation. The runner's initial fleet reset
        # preserves these unexchanged sessions; later episode resets still
        # spawn fresh graphs lazily.
        eager_requested = (
            os.environ.get("CAP_EAGER_GAP_SPAWN", "0") == "1" and config.auto_spawn
        )
        # Arm bases and scene fixtures are measured from the Arena env on the
        # first step; a child spawned before then would read defaults.
        if eager_requested and not _has_explicit_arm_bases(config):
            self._set_eager_spawn("deferred")
            print(
                "[GapRemotePolicy] eager spawn deferred to the first step: "
                "no explicit arm bases configured"
            )
        elif eager_requested and config.gap_scene_statics:
            self._set_eager_spawn("deferred")
            print(
                "[GapRemotePolicy] eager spawn deferred to the first step: "
                "scene fixtures are measured from the env"
            )
        elif eager_requested:
            # Best-effort: an eager-spawn failure must not fail policy
            # construction — fall back to the lazy first-step spawn, which
            # keeps the original error surface (get_action's handlers).
            try:
                if self._fleet:
                    self._assign_fleet_ports()
                    assert self._fleet_executor is not None
                    futures = [
                        self._fleet_executor.submit(session.ensure_started)
                        for session in self._sessions
                    ]
                    for future in futures:
                        future.result()
                else:
                    self._sessions[0].ensure_started()
            except Exception as exc:  # noqa: BLE001
                for session in self._sessions:
                    session.teardown()
                self._set_eager_spawn("deferred")
                print(
                    "[GapRemotePolicy] eager spawn failed; falling back to "
                    f"lazy spawn: {exc}"
                )
            else:
                # Until the first get_action, no runner code path owns this
                # process; a crash during the (long) Arena env build would
                # otherwise orphan the whole GaP process group.
                self._register_atexit_teardown()

    def _set_eager_spawn(self, state: str) -> None:
        for session in self._sessions:
            session.eager_spawn = state

    def _register_atexit_teardown(self) -> None:
        if self._atexit_registered:
            return
        atexit.register(self._atexit_teardown)
        self._atexit_registered = True

    def _unregister_atexit_teardown(self) -> None:
        if not self._atexit_registered:
            return
        atexit.unregister(self._atexit_teardown)
        self._atexit_registered = False

    def _atexit_teardown(self) -> None:
        """Last-resort reap of policy-owned processes (never raises)."""
        for session in self._sessions:
            try:
                session.teardown()
            except Exception:  # noqa: BLE001
                pass

    def _assign_fleet_ports(self) -> None:
        """Pre-assign distinct ports before concurrent child startup."""
        for env_id, session in enumerate(self._sessions):
            if (
                env_id in self._fleet_retired
                or session.started
                or session.assigned_port is not None
            ):
                continue
            session.assign_port()

    @property
    def is_remote(self) -> bool:
        return True

    def _episode_termination_requests(self) -> list[bool]:
        """Return graph-completion requests for the active Arena environments."""
        return [
            env_id not in self._fleet_retired and session.termination_requested()
            for env_id, session in enumerate(self._sessions)
        ]

    def _publish_episode_termination(self) -> None:
        from isaaclab_arena_gap.external_policy_termination import (
            request_external_policy_termination,
        )

        request_external_policy_termination(self._episode_termination_requests())

    def get_action(self, env, observation: dict[str, Any]):
        import torch

        num_envs = env.unwrapped.num_envs
        if not self._fleet:
            assert num_envs == 1, (
                f"GapRemotePolicy is single-env (GaP+cuRobo is one host-side instance); "
                f"got num_envs={num_envs}. Run the eval with num_envs=1 (sequential)."
            )
            action = self._sessions[0].get_action(env, observation, 0)
            self._publish_episode_termination()
            return action
        if num_envs != len(self._sessions):
            raise ValueError(
                f"Arena num_envs={num_envs} does not match GaP fleet "
                f"num_envs={len(self._sessions)}"
            )
        assert self._fleet_executor is not None
        self._assign_fleet_ports()
        self._fleet_initialized = True
        results = {}
        futures = {}
        for env_id, session in enumerate(self._sessions):
            if env_id in self._fleet_retired:
                results[env_id] = session.hold_action(env, env_id)
            else:
                futures[env_id] = self._fleet_executor.submit(
                    session.get_action, env, observation, env_id
                )
        for env_id, future in futures.items():
            try:
                results[env_id] = future.result()
            except Exception as exc:  # isolate one graph/session failure
                session = self._sessions[env_id]
                session.fail(exc)
                results[env_id] = session.hold_action(env, env_id)
        action = torch.cat([results[env_id] for env_id in range(num_envs)], dim=0)
        self._publish_episode_termination()
        return action

    @staticmethod
    def _normalize_env_ids(env_ids: Any, count: int) -> list[int]:
        if env_ids is None:
            return list(range(count))
        values = (
            env_ids.detach().cpu().tolist() if hasattr(env_ids, "detach") else env_ids
        )
        if isinstance(values, (int, np.integer)):
            values = [int(values)]
        ids = [int(value) for value in values]
        if len(set(ids)) != len(ids) or any(
            value < 0 or value >= count for value in ids
        ):
            raise ValueError(f"env_ids must be unique values within [0, {count})")
        return ids

    def reset(self, env_ids=None) -> None:
        from isaaclab_arena_gap.external_policy_termination import (
            clear_external_policy_termination,
        )

        clear_external_policy_termination(env_ids)
        if not self._fleet:
            self._sessions[0].reset(increment_episode=True)
            return
        ids = self._normalize_env_ids(env_ids, len(self._sessions))
        initial_reset = not self._fleet_initialized
        for env_id in ids:
            session = self._sessions[env_id]
            if initial_reset and session.started and not session.exchanged_any:
                continue
            if (
                self._config.one_episode_per_env
                and not initial_reset
                and session.episode_started
            ):
                session.reset(increment_episode=False)
                self._fleet_retired.add(env_id)
            elif env_id not in self._fleet_retired:
                session.reset(increment_episode=not initial_reset)
        self._fleet_initialized = True

    def close(self) -> None:
        for session in self._sessions:
            session.close()
        if self._fleet_executor is not None:
            self._fleet_executor.shutdown(wait=True)
            self._fleet_executor = None
        self._unregister_atexit_teardown()

    def shutdown_remote(self, kill_server: bool = False) -> None:
        """Arena policy_runner teardown hook (policy_runner.py calls
        ``policy.shutdown_remote(kill_server=...)`` on exit). A policy-owned
        per-episode spawn is always reaped here; for an external GaP we own only
        the client (kill_server can't reach a process we didn't spawn). Provided
        so this policy works under the standalone policy_runner, not just
        eval_runner."""
        self.close()


def _resolve_robot_adapter(config: GapRemotePolicyArgs) -> GapObsAdapter:
    from isaaclab_arena_gap.robot_adapter import make_robot_adapter

    return make_robot_adapter(
        config.robot_profile,
        arms=config.gap_arms,
        camera=config.gap_camera,
        bridge_camera=config.gap_bridge_camera,
        wrist_camera=config.gap_wrist_camera,
        wrist_bridge_camera=config.gap_wrist_bridge_camera,
        aux_camera=config.gap_aux_camera,
        aux_bridge_camera=config.gap_aux_bridge_camera,
        extra_cameras=config.gap_extra_cameras,
    )


def _has_explicit_arm_bases(config: GapRemotePolicyArgs) -> bool:
    """Whether the config already carries every arm base its profile reads."""
    if config.robot_profile in BIMANUAL_YAM_PROFILES:
        return len(config.gap_arm_base_poses) == 2
    arm_count = len(config.gap_arms) or 1
    bases = config.gap_arm_base_positions
    return (bases is not None and len(bases) >= arm_count) or (
        arm_count == 1 and config.gap_arm_base_position is not None
    )
