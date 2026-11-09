"""One GaP graph session: a single Arena environment slot's GaP process.

A session owns one GaP process, socket, graph session, and episode lifecycle.
``GapRemotePolicy`` holds one session per Arena environment and coordinates
them; sessions never run in lockstep.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from isaaclab_arena_gap import gripper_collision_fix
from isaaclab_arena_gap.bridge_contract import (
    BRIDGE_STATUS_SCHEMA,
    ContractError,
    atomic_write_json,
    graph_result_reference,
    load_json,
    utc_now,
    validate_graph_result,
)
from isaaclab_arena_gap.gap_process import free_port as _free_port
from isaaclab_arena_gap.gap_process import session_artifact_path
from isaaclab_arena_gap.gap_wire import (
    GapObsAdapter,
    validate_reply_identity,
    validated_action_row,
    validated_command_slices,
    validity_flag,
)
from isaaclab_arena_gap.msgpack_transport import GapProtocolError
from isaaclab_arena_gap.perf_trace import PerfTrace
from isaaclab_arena_gap.process_group import terminate_process_group as _terminate_proc

if TYPE_CHECKING:
    from isaaclab_arena_gap.gap_remote_policy import GapRemotePolicyArgs


def derive_session_seed(base_seed: int, env_id: int, episode_id: int) -> int:
    """Derive a stable 31-bit seed without process-randomized ``hash()``."""
    identity = f"{int(base_seed)}:{int(env_id)}:{int(episode_id)}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big") & 0x7FFFFFFF


def _row(values: Any, env_id: int) -> np.ndarray:
    """One environment's row of an Isaac Lab data array, as float64."""
    values = getattr(values, "torch", values)
    return np.asarray(values[env_id].detach().cpu().numpy(), dtype=np.float64)


def scene_static_poses(
    env: Any, env_id: int, statics: dict[str, str]
) -> dict[str, list[float]]:
    """Environment-local root poses of the scene fixtures ``statics`` names.

    ``statics`` maps a task document's fixture name to the Arena asset standing
    for it. Only kinematic rigid objects are fixtures: anything the simulator
    moves is object state, which a graph must perceive, so it is refused here.
    Returns ``{fixture: [x, y, z, qx, qy, qz, qw]}``.
    """
    unwrapped = env.unwrapped
    origins = getattr(unwrapped.scene, "env_origins", None)
    poses = {}
    for fixture, asset_name in statics.items():
        asset = unwrapped.scene[asset_name]
        rigid_props = getattr(getattr(asset.cfg, "spawn", None), "rigid_props", None)
        if not getattr(rigid_props, "kinematic_enabled", False):
            raise ValueError(
                f"scene static {fixture!r} -> {asset_name!r} is not a kinematic fixture"
            )
        position = _row(asset.data.root_pos_w, env_id)
        if origins is not None:
            position = position - _row(origins, env_id)
        # Isaac Lab 3 stores quaternions xyzw, as RoboSimStudio does.
        quat = _row(asset.data.root_quat_w, env_id)
        poses[fixture] = [float(v) for v in (*position, *quat)]
    return poses


#: Behavior-changing performance knobs stamped into every bridge status so
#: knob-altered runs are distinguishable in recorded evidence. Values are the
#: RESOLVED scalars ("0"/"1"); tracing is recorded as enabled/disabled rather
#: than by its (run-specific) directory path.
_PERF_KNOB_KEYS = ("CAP_EAGER_GAP_SPAWN",)


def _perf_knob_snapshot(eager_spawn: str | None = None) -> dict[str, str]:
    snapshot = {
        key: ("1" if os.environ.get(key, "0") == "1" else "0")
        for key in _PERF_KNOB_KEYS
    }
    if eager_spawn is not None:
        snapshot["CAP_EAGER_GAP_SPAWN"] = eager_spawn
    snapshot["CAP_PERF_TRACE"] = "1" if os.environ.get("CAP_PERF_TRACE_DIR") else "0"
    # From the resolved gate, not os.environ: this knob defaults to ON, so its
    # absence is not its value. Without it, corrected-gripper and stock-gripper
    # runs are indistinguishable in the recorded evidence.
    snapshot["CAP_FIX_GRIPPER_LINKAGE_COLLISION"] = (
        "1" if gripper_collision_fix.linkage_fix_enabled() else "0"
    )
    return snapshot


class GapSession:
    """One Arena slot's GaP process and graph session across episodes.

    ``config`` is the policy config itself for environment 0 and a per-env
    copy for every other environment, so environment 0's writes also land on
    the policy config. The resolved per-session identity (``gap_env_id``,
    ``gap_episode_id``, ``gap_session_seed``, arm bases) is kept on it so
    ``spawn_fn(config, port)`` receives everything the child needs.
    """

    def __init__(
        self,
        config: GapRemotePolicyArgs,
        *,
        adapter: GapObsAdapter,
        workspace: dict[str, float],
        spawn_fn: Callable[[GapRemotePolicyArgs, int], Any],
        client_factory: Callable[[str, int], Any],
        identity_required: bool,
        allocated_ports: set[int],
    ) -> None:
        self.config = config
        self.env_id = int(config.gap_env_id)
        self.session_id = config.gap_session_id
        self._episode_id = int(config.gap_episode_id)
        self._request_id = 0
        self._identity_required = identity_required
        self._gap_workspace = workspace
        self._adapter = adapter
        self.device = config.policy_device
        self._spawn_fn = spawn_fn
        self._client_factory = client_factory
        # Shared with sibling sessions so concurrently assigned ports stay distinct.
        self._allocated_ports = allocated_ports
        # Lazy lifecycle: configure ONLY — nothing spawned or connected here.
        self._proc = None
        self._pgid = None  # spawned process-group id (leader pid via start_new_session)
        self._client = None
        self._port = None
        self.assigned_port: int | None = None
        self.started = False
        self._last_action = None
        self._gap_closed = False
        self._logged_hold = False
        self._status_terminal_written = False
        self._perf = PerfTrace(
            f"bridge-{hashlib.sha256(self.session_id.encode()).hexdigest()[:12]}"
            if identity_required
            else "bridge"
        )
        self._perf_last_exit = None
        self._graph_result = None
        self._graph_result_ref = None
        self._graph_process_disagreement = False
        self._episode_termination_countdown: int | None = None
        self.episode_started = False
        self.exchanged_any = False
        self.eager_spawn: str | None = None

    def _process_exit_code(self, wait_timeout_s: float = 0.0) -> int | None:
        if self._proc is None:
            return None
        try:
            return_code = self._proc.poll()
            if return_code is None and wait_timeout_s > 0:
                return_code = self._proc.wait(timeout=wait_timeout_s)
            return return_code
        except Exception:
            return None

    def _write_bridge_status(
        self,
        *,
        state: str,
        close_kind: str | None = None,
        error: BaseException | None = None,
        terminal: bool = False,
    ) -> None:
        path = session_artifact_path(self.config.bridge_status_path, self.config)
        if not path:
            return
        identity = self._identity_required
        payload = {
            "schema_version": BRIDGE_STATUS_SCHEMA,
            "timestamp": utc_now(),
            "campaign_id": os.environ.get("CAP_CAMPAIGN_ID"),
            "sample_id": os.environ.get("CAP_SAMPLE_ID"),
            "attempt_id": os.environ.get("CAP_ATTEMPT_ID"),
            "perf_knobs": _perf_knob_snapshot(self.eager_spawn),
            "state": state,
            "close_kind": close_kind,
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error)[:2000] if error is not None else None,
            "process_exit_code": self._process_exit_code(),
            "process_group_id": self._pgid,
            "graph": self.config.gap_graph,
            "auto_spawn": self.config.auto_spawn,
            "host": self.config.gap_host,
            "port": self._port,
            "session_id": self.session_id if identity else None,
            "episode_id": self._episode_id if identity else None,
            "env_id": self.env_id if identity else None,
            "seed": self.config.gap_session_seed if identity else None,
            "graph_result": self._graph_result_ref,
            "graph_process_disagreement": self._graph_process_disagreement,
        }
        try:
            atomic_write_json(path, payload)
            self._status_terminal_written = terminal
        except Exception as exc:
            print(f"[GapRemotePolicy] could not write bridge status to {path}: {exc}")

    def _read_final_graph_result(
        self, process_exit_code: int | None
    ) -> tuple[str | None, BaseException | None]:
        """Validate GaP's final sidecar and retain its immutable bridge ref."""
        raw_path = session_artifact_path(self.config.graph_result_path, self.config)
        if not raw_path:
            return None, None
        path = Path(raw_path)
        deadline = time.monotonic() + 2.0
        result = None
        outcome = None
        while True:
            if path.is_file():
                try:
                    result = load_json(path)
                    outcome = validate_graph_result(
                        result,
                        campaign_id=os.environ.get("CAP_CAMPAIGN_ID", ""),
                        plan_sha256=os.environ.get("CAP_PLAN_SHA256", ""),
                        image_ref=os.environ.get("CAP_IMAGE_REF", ""),
                        sample_id=os.environ.get("CAP_SAMPLE_ID", ""),
                        attempt_id=os.environ.get("CAP_ATTEMPT_ID", ""),
                    )
                except ContractError as exc:
                    return "invalid", exc
                if outcome != "started" or time.monotonic() >= deadline:
                    break
            elif time.monotonic() >= deadline:
                return "missing", FileNotFoundError(f"No graph result at {path}")
            time.sleep(0.05)
        try:
            assert result is not None and outcome is not None
            bridge_parent = (
                Path(
                    session_artifact_path(self.config.bridge_status_path, self.config)
                ).parent
                if self.config.bridge_status_path
                else path.parent
            )
            relative_path = os.path.relpath(path, bridge_parent)
            self._graph_result = result
            self._graph_result_ref = graph_result_reference(
                result, relative_path=relative_path
            )
            if outcome == "started":
                return "started", RuntimeError(
                    "Graph result did not reach a terminal outcome"
                )
            self._graph_process_disagreement = (
                process_exit_code == 0 and outcome != "success"
            ) or (process_exit_code not in {None, 0} and outcome == "success")
            return outcome, None
        except (ContractError, OSError) as exc:
            return "invalid", exc

    def ensure_started(self) -> None:
        """Spawn + connect ONE GaP process for this episode, lazily on first use.
        Never restarts within an episode — only reset()/close() tear down (one
        graph per Arena episode)."""
        if self.started:
            return
        cfg = self.config
        # Failure-atomic over the whole bring-up: on any failure, close a
        # half-open client and reap the spawned process group.
        stage = "spawn" if cfg.auto_spawn else "connect"
        try:
            if cfg.auto_spawn:
                port = self.assigned_port or _free_port()
                self._proc = self._spawn_fn(cfg, port)
                # start_new_session makes the child its own group leader, so its
                # pid IS the pgid. Store it now — os.getpgid(pid) raises after the
                # leader exits, so we must not rely on it at teardown.
                self._pgid = getattr(self._proc, "pid", None)
            else:
                port = cfg.gap_port  # connect to an externally-launched GaP
                self._proc = None
                self._pgid = None
            self._port = port
            stage = "connect"
            print(
                f"[GapRemotePolicy] starting episode: connecting to GaP "
                f"{cfg.gap_host}:{port} (auto_spawn={cfg.auto_spawn}) ..."
            )
            self._client = self._client_factory(cfg.gap_host, port)
            self._client.connect()
        except BaseException as exc:
            self._write_bridge_status(
                state="failed",
                close_kind="spawn_failed" if stage == "spawn" else "connect_failed",
                error=exc,
                terminal=True,
            )
            self.teardown()
            raise
        print("[GapRemotePolicy] connected.")
        self.started = True
        self._write_bridge_status(state="connected")

    def teardown(self) -> None:
        """Close the client and terminate+reap the GaP process (idempotent)."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._proc is not None or self._pgid is not None:
            _terminate_proc(self._proc, self._pgid)
            self._proc = None
            self._pgid = None
        self._port = None
        if self.assigned_port is not None:
            self._allocated_ports.discard(self.assigned_port)
        self.assigned_port = None
        self.started = False

    def assign_port(self) -> None:
        """Reserve a port no sibling session holds, before concurrent startup."""
        for _ in range(1000):
            port = _free_port()
            if port not in self._allocated_ports:
                self._allocated_ports.add(port)
                self.assigned_port = port
                return
        raise RuntimeError("could not allocate a distinct GaP fleet port")

    def termination_requested(self) -> bool:
        return not (
            self._episode_termination_countdown is None
            or self._episode_termination_countdown > 0
        )

    def schedule_episode_termination(self, close_kind: str, env: Any) -> None:
        if close_kind == "graph_clean_close":
            settle_s = self.config.graph_success_settle_s
        elif close_kind in {"graph_declared_failure", "graph_exit_nonzero"}:
            settle_s = self.config.graph_failure_settle_s
        else:
            self._episode_termination_countdown = 0
            return
        step_dt = float(getattr(env.unwrapped, "step_dt", 0.02))
        if not np.isfinite(step_dt) or step_dt <= 0:
            raise RuntimeError(
                f"Arena step_dt must be positive and finite, got {step_dt!r}"
            )
        self._episode_termination_countdown = math.ceil(settle_s / step_dt)

    def get_action(self, env: Any, observation: dict[str, Any], env_id: int):
        import torch

        self.episode_started = True
        if self._gap_closed:
            if (
                self._episode_termination_countdown is not None
                and self._episode_termination_countdown > 0
            ):
                self._episode_termination_countdown -= 1
            return self.hold_action(env, env_id)
        perf = self._perf
        t_enter = time.perf_counter()
        last_exit = self._perf_last_exit
        sim_side_s = t_enter - last_exit if last_exit is not None else None
        if not self.started:
            bases = self._adapter.arm_base_positions(env, env_id)
            self.config.gap_arm_base_positions = bases
            self.config.gap_arm_base_position = bases[0] if bases else None
            if hasattr(self._adapter, "base_poses"):
                self.config.gap_arm_base_poses = self._adapter.base_poses(env, env_id)
            statics = getattr(self.config, "gap_scene_statics", None)
            if statics:
                self.config.gap_scene_static_poses = scene_static_poses(env, env_id, statics)
        self.ensure_started()  # lazy spawn+connect on first action of the episode
        t_started = time.perf_counter()
        frame = self._adapter.build_obs_frame(env, observation, env_id=env_id)
        workspace = self._gap_workspace
        if workspace or self._identity_required:
            metadata = frame.setdefault("_isaac_cap", {})
            if not isinstance(metadata, dict):
                raise GapProtocolError("Arena observation _isaac_cap must be a mapping")
            if workspace:
                metadata["workspace"] = workspace
            if self._identity_required:
                self._request_id += 1
                metadata.update(
                    {
                        "session_id": self.session_id,
                        "episode_id": self._episode_id,
                        "request_id": self._request_id,
                        "seed": self.config.gap_session_seed,
                    }
                )
        t_built = time.perf_counter()
        try:
            action = self._client.round_trip(frame)
            if self._identity_required:
                validate_reply_identity(action, frame["_isaac_cap"])
            row = self._decode_action_row(action, env, env_id)
            self.exchanged_any = True
            t_exit = time.perf_counter()
            self._perf_last_exit = t_exit
            perf.record(
                "step",
                sim_s=sim_side_s,
                spawn_s=t_started - t_enter,
                build_s=t_built - t_started,
                send_s=getattr(self._client, "last_send_s", None),
                recv_s=getattr(self._client, "last_recv_s", None),
                total_s=t_exit - t_enter,
            )
        except GapProtocolError as e:
            self._gap_closed = True
            self.schedule_episode_termination("protocol_error", env)
            self._write_bridge_status(
                state="failed",
                close_kind="protocol_error",
                error=e,
                terminal=True,
            )
            self.teardown()
            return self.hold_action(env, env_id)
        except (BrokenPipeError, ConnectionError, TimeoutError, OSError) as e:
            print(
                f"[GapRemotePolicy] GaP channel ended ({type(e).__name__}: {e}); "
                "holding so Arena can terminate and write its result."
            )
            # Latch closed FIRST (get_action returns hold at the top, so it can
            # never respawn this episode), then reap the dead/dying process.
            self._gap_closed = True
            if isinstance(e, TimeoutError):
                process_exit_code = self._process_exit_code()
                close_kind = "io_timeout"
                graph_result_error = None
            else:
                # GaP closes the bridge before its tool clients have
                # necessarily finished process teardown. The episode is over,
                # so a longer bounded wait avoids misclassifying a clean CUDA
                # teardown as a transport failure.
                process_exit_code = self._process_exit_code(wait_timeout_s=30.0)
                graph_outcome, graph_result_error = self._read_final_graph_result(
                    process_exit_code
                )
                if graph_outcome == "success":
                    close_kind = "graph_clean_close"
                elif graph_outcome == "failure":
                    close_kind = "graph_declared_failure"
                elif graph_outcome == "runtime_error":
                    close_kind = "graph_runtime_error"
                elif graph_outcome in {"missing", "started", "invalid"}:
                    close_kind = f"graph_result_{graph_outcome}"
                elif process_exit_code == 0:
                    close_kind = "graph_clean_close"
                elif process_exit_code is not None:
                    close_kind = "graph_exit_nonzero"
                elif isinstance(e, BrokenPipeError):
                    close_kind = "channel_closed"
                else:
                    close_kind = "connection_error"
            print(
                f"[GapRemotePolicy] GaP process exit code={process_exit_code}; "
                f"close_kind={close_kind}."
            )
            self.schedule_episode_termination(close_kind, env)
            ended = close_kind in {"graph_clean_close", "graph_declared_failure"}
            self._write_bridge_status(
                state="ended" if ended else "failed",
                close_kind=close_kind,
                error=graph_result_error if ended else graph_result_error or e,
                terminal=True,
            )
            self.teardown()
            return self.hold_action(env, env_id)
        batch = row[None, :]
        self._last_action = torch.from_numpy(batch).to(
            dtype=torch.float32, device=self.device
        )
        return self._last_action

    def fail(self, error: BaseException) -> None:
        """Fail this fleet session closed without affecting sibling sessions."""
        self._gap_closed = True
        self._episode_termination_countdown = 0
        self._write_bridge_status(
            state="failed",
            close_kind="session_error",
            error=error,
            terminal=True,
        )
        self.teardown()

    def _decode_action_row(
        self, action: dict[str, Any], env: Any, env_id: int = 0
    ) -> np.ndarray:
        # Per-channel startup hold: until GaP's graph commands a channel, HOLD
        # Arena's observed state for it (never apply GaP's hardcoded startup
        # seed). arm_valid: an arm move was issued; gripper_valid: a gripper
        # command was issued (both stamped per-channel by CapFrankaRealEnv).
        # Holding per-channel means a gripper-only first command applies the
        # gripper WITHOUT reintroducing the seed joints (arm stays at observed q).
        # STRICT (default): missing validity => INVALID => hold — an unflagged
        # reply is the parent's hardcoded pre-seed (e.g. during URDF load) and
        # must NOT be applied. strict_command_valid=False opts out (legacy graphs
        # that never stamp => absent treated as valid/decoded).
        if not isinstance(action, dict):
            raise GapProtocolError(
                f"GaP action must be a mapping, got {type(action).__name__}"
            )

        if hasattr(self._adapter, "decode_validated_action"):
            return validated_action_row(
                self._adapter.decode_validated_action(
                    action, env, env_id, strict=self.config.strict_command_valid
                ),
                self._adapter.action_dim_for(env, env_id),
                "decoded action",
            )

        dim = self._adapter.action_dim_for(env, env_id)
        command_slices = validated_command_slices(self._adapter, env, env_id, dim)
        hold = validated_action_row(
            self._adapter.hold_action(env, env_id), dim, "hold action"
        )

        # Invalid channels may contain GaP's startup seed or no value at all.
        # Give the adapter well-formed placeholders, then replace those channels
        # with the observed hold below. Valid channels remain untouched and must
        # pass the adapter's strict wire validation.
        default_valid = not self.config.strict_command_valid
        validity: dict[str, tuple[bool, bool]] = {}
        decode_input = dict(action)
        for wire_arm, action_slice in command_slices.items():
            arm_valid = validity_flag(
                action, "arm_valid", default_valid, wire_arm=wire_arm
            )
            grip_valid = validity_flag(
                action, "gripper_valid", default_valid, wire_arm=wire_arm
            )
            validity[wire_arm] = (arm_valid, grip_valid)

            raw_block = action.get(wire_arm)
            block = dict(raw_block) if isinstance(raw_block, dict) else {}
            if not arm_valid:
                block["joint_pos"] = hold[
                    action_slice.start : action_slice.stop - 1
                ].tolist()
            if not grip_valid:
                block["gripper"] = 1.0
            decode_input[wire_arm] = block

        row = validated_action_row(
            self._adapter.decode_action(decode_input, env, env_id),
            dim,
            "decoded action",
        )
        if any(
            not (arm_valid and grip_valid)
            for arm_valid, grip_valid in validity.values()
        ):
            row = row.copy()
            for wire_arm, action_slice in command_slices.items():
                arm_valid, grip_valid = validity[wire_arm]
                if not arm_valid:
                    row[action_slice.start : action_slice.stop - 1] = hold[
                        action_slice.start : action_slice.stop - 1
                    ]
                if not grip_valid:
                    row[action_slice.stop - 1] = hold[action_slice.stop - 1]
            if not self._logged_hold:
                status = ", ".join(
                    f"{wire_arm}:arm={arm_valid},gripper={grip_valid}"
                    for wire_arm, (arm_valid, grip_valid) in validity.items()
                )
                print(
                    f"[GapRemotePolicy] startup hold: {status} "
                    f"(holding observed q/gripper "
                    f"until the graph commands each channel)"
                )
                self._logged_hold = True
        return validated_action_row(row, dim, "final action")

    def hold_action(self, env, env_id: int = 0):
        import torch

        # Hold the CURRENT OBSERVED arm q (never replay a possibly-stale commanded
        # arm target — that could fling the arm). Preserve the LAST COMMANDED
        # gripper so a grasp stays closed; if no command yet, hold observed gripper.
        dim = self._adapter.action_dim_for(env, env_id)
        row = validated_action_row(
            self._adapter.hold_action(env, env_id), dim, "hold action"
        ).copy()
        if self._last_action is not None:
            last = self._last_action.detach().cpu().numpy().reshape(-1)
            for index in self._adapter.gripper_action_indices(env, env_id):
                row[index] = last[index]
        return torch.from_numpy(row[None, :]).to(
            dtype=torch.float32, device=self.device
        )

    def reset(self, *, increment_episode: bool) -> None:
        # End of episode: terminate+reap the GaP process and reset state so the
        # NEXT get_action spawns a FRESH graph (one graph per Arena episode).
        self._perf.record("episode_reset", unix=time.time())
        self._perf_last_exit = None
        if self.started and self.exchanged_any and not self._status_terminal_written:
            self._write_bridge_status(
                state="ended",
                close_kind="arena_termination",
                terminal=True,
            )
        self.teardown()
        self.exchanged_any = False
        self._last_action = None
        self._gap_closed = False
        self._logged_hold = False
        self._status_terminal_written = False
        self._graph_result = None
        self._graph_result_ref = None
        self._graph_process_disagreement = False
        self._episode_termination_countdown = None
        self.episode_started = False
        self._request_id = 0
        if increment_episode:
            self._episode_id += 1
            self.config.gap_episode_id = self._episode_id
            self.config.gap_session_seed = derive_session_seed(
                self.config.gap_base_seed,
                self.env_id,
                self._episode_id,
            )

    def close(self) -> None:
        if self.started and not self._status_terminal_written:
            self._write_bridge_status(
                state="ended",
                close_kind="policy_close",
                terminal=True,
            )
        self.teardown()
        self._perf.close()
