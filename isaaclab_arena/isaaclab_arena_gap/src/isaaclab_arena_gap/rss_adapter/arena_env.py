"""An Env for RoboSimStudio's policy agent whose world is Arena.

RoboSimStudio's ``SolutionGraph``, ``GapEnvFacade``, servo, IK and tools run
unmodified against this object. It is RoboSimStudio's own robot-only ``Sim``
(Arena's robot asset, AUTOLab's registered hand), with three things replaced:

* **Time.** ``step`` decodes the action with RoboSimStudio's own controllers,
  sends the resulting arm and gripper targets to Arena, and blocks for Arena's
  next sample. One ``step`` is one Arena control period.
* **State.** The robot's joints, including the hand's passive linkage, are
  written from each Arena sample and forward kinematics fills ``body_q``.
* **Cameras.** Frames, intrinsics and extrinsics are Arena's.

World positions (arm mounts, camera poses, fixtures, the work surface) are
Arena's, translated by the binding's ``frame_offset`` where AUTOLab's task code
holds world constants of its own (see ``GraphBinding.task_base_midpoint``).
The translation is rigid, so kinematics, IK and perception are unchanged by
it; joint commands carry no frame.

It is strict by construction: the scene holds the robot and nothing else, so
a tool that reaches for a simulated object finds none. Task identity and
instruction come from AUTOLab's own task document (``doc``, as RoboSimStudio's
bench env exposes it: what the task declares, never where anything is);
geometry never does, because AUTOLab's world frame is not Arena's. Two
exceptions to "robot only", neither of them simulated object state:

* a hand calibration a task declares as an entity (USB-C's ``rig``): it is
  built on AUTOLab's own robot and carries no world state, and is found by name
  as the task's tools expect;
* the static fixtures a task declares (cable's pegs, anchor and port), as
  handles at Arena's poses. AUTOLab's compiled scene hangs every welded
  fixture's pose on ``handles.statics`` under any perception mode, and its
  tools read it there when they register; nothing that moves is offered.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import newton
import numpy as np
import robosimstudio as rss
from scipy.spatial.transform import Rotation

from isaaclab_arena_gap.bridge_robot_io import JointPositionCommand


def translated(pose, offset) -> tuple[float, ...]:
    """A world pose ``(x, y, z, *rest)`` moved by ``offset``; orientation unchanged."""
    values = tuple(float(v) for v in pose)
    return (*(v + float(d) for v, d in zip(values[:3], offset, strict=True)), *values[3:])


def static_handles(task, poses: dict[str, Any], frame_offset=(0.0, 0.0, 0.0)) -> dict[str, Any]:
    """Handles for the task's declared static fixtures, at Arena's ``poses`` (xyz, xyzw).

    Each pose is moved by ``frame_offset`` into the frame the shadow runs in.

    As RoboSimStudio's compiler makes them for a welded fixture: bodiless, with
    the pose on ``xform``. Only names the task declares as statics are accepted;
    a pose for anything else (an object, the cable) is refused rather than
    offered to the graph.
    """
    from robosimstudio.scene.handles import EntityHandle
    from robosimstudio.specs import StaticSpec

    declared = {spec.name for spec in task.entities if isinstance(spec, StaticSpec)}
    unknown = sorted(set(poses) - declared)
    if unknown:
        raise KeyError(f"{task.full_name} declares no static fixture {unknown}; statics {sorted(declared)}")
    handles = {}
    for name, pose in poses.items():
        xform = tuple(float(v) for v in pose)
        if len(xform) != 7 or not np.all(np.isfinite(xform)):
            raise ValueError(f"static {name!r} needs a finite (x, y, z, qx, qy, qz, qw) pose, got {pose!r}")
        handles[name] = EntityHandle(name=name, category="static", static=True,
                                     xform=translated(xform, frame_offset))
    return handles


class ArenaCamera:
    """The fields ``GapEnvFacade`` and task tools read off a camera spec, from Arena's calibration."""

    def __init__(self, name: str, env: "ArenaEnv", parent_body: int = -1) -> None:
        self.name = name
        self._env = env
        self.parent_body = int(parent_body)
        """The shadow body the camera rides, as AUTOLab's rig mounts it (``-1``: world-fixed)."""

    @property
    def intrinsics(self) -> np.ndarray:
        return np.asarray(self._env.sample.cameras[self.name].intrinsics, dtype=np.float64)

    def pose_on_body(self, body_q: Any = None) -> np.ndarray:
        del body_q  # Arena's camera pose is already in the world frame
        pose = np.asarray(self._env.sample.cameras[self.name].pose, dtype=np.float64).copy()
        pose[:3, 3] += self._env.frame_offset
        return pose

    @property
    def world_matrix(self) -> np.ndarray:
        """As ``CameraSpec.world_matrix`` states it: the world pose, or for a mounted camera its pose
        on the body it rides -- here Arena's measured mount, read off the current sample."""
        pose = self.pose_on_body()
        if self.parent_body < 0:
            return pose
        px, py, pz, qx, qy, qz, qw = self._env.state.body_q.numpy()[self.parent_body]
        body = np.eye(4)
        body[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        body[:3, 3] = (px, py, pz)
        return np.linalg.inv(body) @ pose


class ArenaEnv:
    """RoboSimStudio's Env surface over Arena: a robot-only shadow driven by RobotIO samples."""

    def __init__(self, io, embodiment, task, *, arm_joints: dict[str, list[str]], gripper_joint: dict[str, str],
                 passive_joints: dict[str, str] | None = None, cameras: tuple[str, ...] = (), episode_id: int = 0,
                 surface_z: float | None = None, work_arm: str = "", support_arm: str = "",
                 calibrations: dict[str, Any] | None = None, statics: dict[str, Any] | None = None,
                 frame_offset=(0.0, 0.0, 0.0)):
        self.frame_offset = np.asarray(frame_offset, dtype=np.float64).reshape(3)
        """Translation from Arena's world to the shadow's; ``embodiment`` is already mounted in it."""
        scene = rss.Scene()
        scene.add(rss.robot(embodiment))
        self._sim = scene.build()
        # Found by name, as a task's tools look them up; never stepped or synced.
        self._sim.scene.by_name.update(calibrations or {})
        self._sim.handles.statics.update(static_handles(task, statics or {}, self.frame_offset))
        if work_arm:
            # RoboSimStudio counts arm_id 0 as the work arm, whatever order the arms are declared in.
            self._sim.handles.work_arm = work_arm
            self._sim.handles.support_arm = support_arm
        if surface_z is not None:
            # The work surface the Arena task declares, in the shadow's frame. RSS derives
            # transport heights and hover clearances from it and the hand's envelope.
            self._sim.handles.surface_z = float(surface_z) + float(self.frame_offset[2])
        self.io = io
        self.task = task
        # As RoboSimStudio's bench env answers it: the evaluator's document, or None.
        self.doc = getattr(task.evaluator(task), "doc", None)
        self.episode = SimpleNamespace(
            episode_id=int(episode_id),
            instruction=str(getattr(task, "instruction", "") or getattr(task, "language", "") or ""),
        )
        self.max_steps = 0  # Arena ends the episode
        self.last_result = SimpleNamespace(success=False)
        self.termination_request: str | None = None
        self.renderer = None
        self.obs_renderer = None
        self.obs_render_enabled = True
        self.frames: dict[str, dict[str, Any]] = {}
        self.video_frames: dict[str, dict[str, Any]] = {}
        self._camera_names = tuple(cameras)
        # AUTOLab's rig says which arm a camera rides; RoboSimStudio mounts it on that arm's end body.
        mounted = {spec.name: spec.parent_frame for spec in (getattr(task, "cameras", None) or ()) if spec.parent_frame}
        self.cameras = [
            ArenaCamera(name, self, int(self._sim.handles.arm(mounted[name]).ee_body) if name in mounted else -1)
            for name in self._camera_names
        ]
        names = list(io.joint_names)
        self._arm_index = {arm: [names.index(n) for n in joints] for arm, joints in arm_joints.items()}
        self._grip_index = {arm: names.index(n) for arm, n in gripper_joint.items()}
        # Arena's measured passive hand joints, by the shadow's own joint label.
        labels = [label.rsplit("/", 1)[-1] for label in self._sim.model.joint_label]
        q_start = self._sim.model.joint_q_start.numpy()
        self._passive = {}
        for io_name, model_name in (passive_joints or {}).items():
            if io_name in names and model_name in labels:
                self._passive[names.index(io_name)] = int(q_start[labels.index(model_name)])
        self.sample = None
        self.step_count = 0
        self.sim_time = 0.0

    # ------------------------------------------------------------ delegation
    def __getattr__(self, name: str) -> Any:
        return getattr(self._sim, name)

    @property
    def state(self):
        return self._sim.state

    @property
    def control(self):
        return self._sim.control

    # ------------------------------------------------------------ Arena sync
    def _apply_mimic(self, q: np.ndarray, measured: set[int]) -> None:
        """Coordinates the wire does not carry, through the asset's own mimic constraints.

        A constraint runs from whichever side was measured. Arena's crank YAM declares its
        driven finger the mimic of its twin, the reverse of the hand spec RSS drives, so
        writing the follower from its unwritten leader would overwrite the measured opening
        with the twin's stale coordinate (``robot.close_gripper`` then reads 0.0137 open,
        whatever the hand does). The measured side is solved for its twin instead.
        """
        model = self._sim.model
        if not getattr(model, "constraint_mimic_count", 0):
            return
        starts = model.joint_q_start.numpy()
        for follower, leader, offset, scale, enabled in zip(
            model.constraint_mimic_joint0.numpy(), model.constraint_mimic_joint1.numpy(),
            model.constraint_mimic_coef0.numpy(), model.constraint_mimic_coef1.numpy(),
            model.constraint_mimic_enabled.numpy(), strict=True,
        ):
            if not enabled:
                continue
            f, lead = int(starts[follower]), int(starts[leader])
            if f in measured and lead not in measured:
                if abs(float(scale)) > 1.0e-12:
                    q[lead] = (q[f] - offset) / scale
            elif f not in measured:
                q[f] = offset + scale * q[lead]

    def _sync(self, sample) -> None:
        self.sample = sample
        sim = self._sim
        q = np.asarray(sim.state.joint_q.numpy(), dtype=np.float32).copy()
        measured: set[int] = set()
        for arm, idx in self._arm_index.items():
            handle = sim.handles.arms[arm]
            q[list(handle.arm_coords)] = sample.joint_positions[idx]
            measured.update(int(c) for c in handle.arm_coords)
            # Arena reports the hand as an open fraction; the hand spec's own open/closed
            # finger coordinates turn it back into the drivers RSS's Gripper measures.
            if handle.gripper_coords and handle.gripper_open_q and handle.gripper_closed_q:
                closed = 1.0 - float(sample.joint_positions[self._grip_index[arm]])
                opened, shut = np.asarray(handle.gripper_open_q), np.asarray(handle.gripper_closed_q)
                q[list(handle.gripper_coords)] = opened + closed * (shut - opened)
                measured.update(int(c) for c in handle.gripper_coords)
        self._apply_mimic(q, measured)
        # Measured passive joints win over the conversion where Arena reports them.
        for io_i, coord in self._passive.items():
            q[coord] = sample.joint_positions[io_i]
        sim.state.joint_q.assign(q)
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
        # The pipeline never steps, so its host read-back (``make_context``'s body poses and servo
        # targets, which grip and pose tools read) is re-read only when told the state moved.
        sim.pipeline.mark_dirty()
        self.sim_time = float(sample.timestamp_s)
        self.frames = {
            name: {"rgb": cam.rgb, **({"depth": cam.depth_m} if cam.depth_m is not None else {})}
            for name, cam in sample.cameras.items()
            if name in self._camera_names
        }

    def _command(self) -> np.ndarray:
        """Arena's target vector from RoboSimStudio's decoded targets: arm joints and an open fraction."""
        sim = self._sim
        target = np.asarray(self.sample.joint_positions, dtype=np.float64).copy()
        joint_target = sim.control.joint_target_q.numpy()
        for arm, idx in self._arm_index.items():
            target[idx] = joint_target[list(sim.handles.arms[arm].arm_coords)]
            gripper = sim.scene.embodiment.grippers.get(arm)
            if gripper is not None:
                # RSS's command is 1.0 closed; Arena's wire is 1.0 open.
                target[self._grip_index[arm]] = float(np.clip(1.0 - float(gripper.command), 0.0, 1.0))
        return target

    # ------------------------------------------------------------ Env surface
    def reset(self, *args, **kwargs):
        del args, kwargs
        self._sync(self.io.observe())
        # Targets start where Arena's robot is, so the first hold does not move it.
        sim = self._sim
        jt = sim.control.joint_target_q.numpy().copy()
        jt[:] = sim.state.joint_q.numpy()
        sim.control.joint_target_q.assign(jt)
        for arm, i in self._grip_index.items():
            gripper = sim.scene.embodiment.grippers.get(arm)
            if gripper is not None:
                gripper.command = 1.0 - float(self.sample.joint_positions[i])
        self.step_count = 0
        self.termination_request = None
        return self.observe(), {"success": False}

    def apply_action(self, action, mode: str = "") -> None:
        self._sim.apply_action(action, mode)

    def use_control(self, mode: str):
        return self._sim.use_control(mode)

    def step(self, action=None, mode: str = ""):
        if action is not None:
            self._sim.apply_action(action, mode)
        self._sync(self.io.exchange(JointPositionCommand(self._command())))
        self.step_count += 1
        obs = self.observe()
        terminated = self.termination_request is not None
        info = {"success": False, "reward": 0.0}
        if terminated:
            info["termination_reason"] = self.termination_request
        return obs, 0.0, terminated, False, info

    def observe(self) -> dict[str, Any]:
        return {
            "proprio": self._sim._observe_proprio(),
            "cameras": dict(self.frames),
            "instruction": self.episode.instruction,
            "step": self.step_count,
            "sim_time": self.sim_time,
        }

    def evaluate(self):
        return self.termination_request is not None, False, {"success": False, "reward": 0.0}

    def request_termination(self, reason: str) -> None:
        self.termination_request = str(reason)
