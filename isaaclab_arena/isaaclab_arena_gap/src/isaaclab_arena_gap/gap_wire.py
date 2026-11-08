"""GaP wire format: the Arena-side adapter interface and reply validation.

``GapObsAdapter`` translates between Arena observations/actions and GaP's
msgpack frames; concrete FR3 and YAM adapters live in ``robot_adapter``. The
helpers below validate replies before any value reaches the Arena action
tensor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from isaaclab_arena_gap.msgpack_transport import GapProtocolError


class GapObsAdapter(ABC):
    """Translate between Arena's env/observation and GaP's msgpack frames.

    The concrete adapter is Arena-API-coupled (reads the robot/camera off
    `env.unwrapped.scene`, computes the camera `pose_mat` = T_wb^-1 @ T_wc, and
    encodes GaP's joint+gripper reply into the embodiment's action-term layout).
    Concrete adapters live in ``isaaclab_arena_gap.robot_adapter``.
    """

    action_dim: int  # width of the env action tensor (e.g. 7 joints + 1 gripper)

    def action_dim_for(self, env: Any, env_id: int = 0) -> int:
        """Return the complete Arena action width for the active environment."""
        del env, env_id
        return int(self.action_dim)

    def command_action_dim_for(self, env: Any, env_id: int = 0) -> int:
        """Return the action width controlled by the GaP wire command."""
        return self.action_dim_for(env, env_id)

    def command_action_slices_for(self, env: Any, env_id: int = 0) -> dict[str, slice]:
        """Map each GaP wire arm to its complete Arena arm/gripper slice."""
        width = self.command_action_dim_for(env, env_id)
        return {"left": slice(0, width)}

    def arm_base_position(
        self, env: Any, env_id: int = 0
    ) -> tuple[float, float, float] | None:
        """Robot base position in the graph's environment-local world frame."""
        del env, env_id
        return None

    def gripper_action_indices(self, env: Any, env_id: int = 0) -> tuple[int, ...]:
        slices = validated_command_slices(
            self, env, env_id, self.action_dim_for(env, env_id)
        )
        return tuple(block.stop - 1 for block in slices.values())

    def arm_base_positions(
        self, env: Any, env_id: int = 0
    ) -> tuple[tuple[float, float, float], ...] | None:
        """Robot bases ordered like the GaP wire arms."""
        base = self.arm_base_position(env, env_id)
        return None if base is None else (base,)

    @abstractmethod
    def build_obs_frame(
        self, env: Any, observation: dict[str, Any], env_id: int
    ) -> dict[str, Any]:
        """Assemble GaP's obs frame for one env:
        {timestamp, left:{joint_pos:[q1..q7, gripper_frac]}, <cam>:{images:{rgb},
         depth_data, intrinsics:{left:{intrinsics_matrix}}, pose_mat}}."""

    @abstractmethod
    def decode_action(
        self, action: dict[str, Any], env: Any, env_id: int = 0
    ) -> np.ndarray:
        """Map GaP's reply {left:{joint_pos:[7], gripper}} to a (action_dim,)
        row for the embodiment's action term (joint order + gripper encoding)."""

    @staticmethod
    def parse_wire_action(
        action: dict[str, Any], *, expected_arm_dof: int = 7
    ) -> tuple[np.ndarray, float]:
        """Validate and return GaP arm joints and gripper open fraction."""
        if not isinstance(action, dict):
            raise GapProtocolError(
                f"GaP action must be a mapping, got {type(action).__name__}"
            )
        left = action.get("left")
        if not isinstance(left, dict):
            raise GapProtocolError("GaP action must contain a 'left' mapping")

        try:
            raw_joints = np.asarray(left.get("joint_pos"))
        except (TypeError, ValueError) as exc:
            raise GapProtocolError(
                f"GaP joint_pos must contain {expected_arm_dof} numeric values"
            ) from exc
        if raw_joints.shape != (expected_arm_dof,):
            raise GapProtocolError(
                f"GaP joint_pos must have shape ({expected_arm_dof},), got {raw_joints.shape}"
            )
        if raw_joints.dtype.kind not in "iuf":
            raise GapProtocolError("GaP joint_pos must contain real numeric values")
        joints = raw_joints.astype(np.float32, copy=False)
        if not np.isfinite(joints).all():
            raise GapProtocolError("GaP joint_pos contains a non-finite value")

        raw_gripper = left.get("gripper")
        if raw_gripper is None:
            raise GapProtocolError("GaP action must contain a gripper open fraction")
        try:
            gripper_value = np.asarray(raw_gripper)
        except (TypeError, ValueError) as exc:
            raise GapProtocolError(
                "GaP gripper must be a numeric open fraction"
            ) from exc
        if gripper_value.shape != () or gripper_value.dtype.kind not in "iuf":
            raise GapProtocolError("GaP gripper must be a numeric open fraction")
        try:
            gripper = float(gripper_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GapProtocolError(
                "GaP gripper must be a numeric open fraction"
            ) from exc
        if not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
            raise GapProtocolError(
                f"GaP gripper must be finite and within [0, 1], got {gripper!r}"
            )
        return joints, gripper

    def hold_action(self, env: Any, env_id: int = 0) -> np.ndarray:
        """Action row that HOLDS the robot at its currently-observed arm q — the
        startup/pre-ready hold used by GapRemotePolicy until GaP marks
        ``command_valid`` (so no hardcoded GaP startup seed ever reaches the
        robot). The default holds arm q with a zero gripper term; embodiment
        adapters override to also hold the observed gripper state."""
        robot = env.unwrapped.scene["robot"]
        dim = int(self.action_dim)
        row = np.zeros(dim, dtype=np.float32)
        q = robot.data.joint_pos[env_id, : dim - 1].cpu().numpy()
        row[: len(q)] = q
        return row


def validated_action_row(value: Any, expected_dim: int, name: str) -> np.ndarray:
    """Return one finite float32 action row or reject the adapter output."""
    try:
        row = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise GapProtocolError(f"{name} must be numeric") from exc
    if row.shape != (expected_dim,):
        raise GapProtocolError(
            f"{name} must have shape ({expected_dim},), got {row.shape}"
        )
    if not np.isfinite(row).all():
        raise GapProtocolError(f"{name} contains a non-finite value")
    return row


def validated_command_slices(
    adapter: GapObsAdapter, env: Any, env_id: int, action_dim: int
) -> dict[str, slice]:
    """Return disjoint, in-bounds Arena slices for every GaP wire arm."""
    command_slices = adapter.command_action_slices_for(env, env_id)
    if not isinstance(command_slices, dict) or not command_slices:
        raise GapProtocolError(
            "adapter command action slices must be a non-empty mapping"
        )

    occupied: set[int] = set()
    validated: dict[str, slice] = {}
    for wire_arm, action_slice in command_slices.items():
        if wire_arm not in ("left", "right") or not isinstance(action_slice, slice):
            raise GapProtocolError(
                "adapter command action slices must map 'left'/'right' to slices"
            )
        start = action_slice.start
        stop = action_slice.stop
        if (
            not isinstance(start, int)
            or not isinstance(stop, int)
            or action_slice.step not in (None, 1)
            or start < 0
            or stop <= start
            or stop > action_dim
        ):
            raise GapProtocolError(
                f"adapter command slice for {wire_arm!r} must fit within width "
                f"{action_dim}, got {action_slice!r}"
            )
        indices = set(range(start, stop))
        if occupied & indices:
            raise GapProtocolError("adapter command action slices must not overlap")
        occupied.update(indices)
        validated[wire_arm] = action_slice
    return validated


def validity_flag(
    action: dict[str, Any], channel: str, default: bool, *, wire_arm: str = "left"
) -> bool:
    """Read one scalar or per-wire-arm validity flag."""
    if channel in action:
        value = action[channel]
    elif "command_valid" in action:
        value = action["command_valid"]
    else:
        value = default
    if isinstance(value, dict):
        value = value.get(wire_arm, default)
    if not isinstance(value, (bool, np.bool_)):
        raise GapProtocolError(
            f"GaP {channel}[{wire_arm!r}] must be a boolean, got {value!r}"
        )
    return bool(value)


def validate_reply_identity(action: Any, expected_metadata: dict[str, Any]) -> None:
    """Reject a reply that belongs to another session, episode, or request."""
    if not isinstance(action, dict):
        raise GapProtocolError(
            f"GaP action must be a mapping, got {type(action).__name__}"
        )
    metadata = action.get("_isaac_cap")
    if not isinstance(metadata, dict):
        raise GapProtocolError("GaP fleet reply is missing _isaac_cap identity")
    for key in ("session_id", "episode_id", "request_id"):
        if metadata.get(key) != expected_metadata.get(key):
            raise GapProtocolError(
                f"GaP fleet reply has stale {key}: expected "
                f"{expected_metadata.get(key)!r}, got {metadata.get(key)!r}"
            )
