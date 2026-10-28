# SPDX-License-Identifier: Apache-2.0
"""The tool-hanging suite's scripted routines: one state machine, four tool types.

Every routine here is the same shape -- :class:`TwoArmRoutine`, a
class-attribute-driven state machine that plans once against the live object
pose and then smoothsteps between converged joint waypoints -- specialised by a
frozen job table of **measured mesh-local landmarks**. The wrench tracks its
ring centre, the screwdriver its tip, the scissors their handle bow, the plier
its handle-gap axis; the gripper is commanded so that the *landmark* lands on
the target, which is what lets two differently-authored meshes share a motion.

One file rather than the branch-era ``routines/`` package: under
a task directory, only ``routine.py``, ``tools.py`` and ``task.py`` may be
Python, and ``tests/test_layout.py`` enforces it so that "a task is a
document" cannot erode one convenience file at a time.

The landmark constants double as the goals' feature payloads: the ring centres
and hook geometry pinned in this task's ``assets/records.py`` are the
same measured numbers these jobs steer by, so the routine and the evaluator
cannot disagree about where "hung" is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_multiply, quat_to_mat
from robosimstudio.robots.embodiments import top_down_grasp_site_quat

from .assets.records import HOOK_TIP_WORLD_X, HOOK_TIP_WORLD_Z, LEVEL1_Y

__all__ = [
    "PLIER_HANG_JOBS",
    "PLIER_HANG_JOBS_LEVEL1",
    "SCISSOR_HANG_JOBS",
    "SCISSOR_HANG_JOBS_LEVEL1",
    "SCREWDRIVER_INSERT_JOBS",
    "SCREWDRIVER_INSERT_JOBS_LEVEL1",
    "WRENCH_HANG_JOBS",
    "WRENCH_HANG_JOBS_LEVEL1",
    "HangPlierRoutine",
    "HangPlierRoutineLevel1",
    "HangScissorRoutine",
    "HangScissorRoutineLevel1",
    "HangWrenchRoutine",
    "HangWrenchRoutineLevel1",
    "InsertScrewdriverRoutine",
    "InsertScrewdriverRoutineLevel1",
    "TwoArmRoutine",
]


# ---------------------------------------------------------------- shared tuning
# One value serves every tool where the branch modules stated the same number;
# a constant that genuinely differs per tool carries the tool's prefix below.
# LEVEL1_Y, HOOK_TIP_WORLD_X and HOOK_TIP_WORLD_Z are the records' -- the
# board's centre station and the shared hook's measured tip -- imported above
# so the job tables steer by the same numbers the fixtures are placed with.
APPROACH_ABOVE_GRASP = 0.120
PRE_HOOK_DISTANCE_IN_FRONT = 0.045
INSERT_DISTANCE_PAST_HOOK_TIP = 0.018
LOWER_ONTO_HOOK_DISTANCE = 0.012
RETRACT_AWAY_FROM_BOARD = 0.100
RETRACT_UPWARD = 0.050
POSITION_ERROR_LIMIT = 0.045
ROTATION_ERROR_LIMIT = np.radians(20.0)

# ------------------------------------------------------------------------ base


class TwoArmRoutine:
    """Run two preplanned arm jobs sequentially with smooth joint commands."""

    MODE = "joint_position"
    JOBS: ClassVar[tuple] = ()
    PHASES: ClassVar[tuple[str, ...]] = ()
    DURATIONS: ClassVar[dict[str, int]] = {}
    GRIP_PHASES: ClassVar[frozenset[str]] = frozenset()
    POSITION_LIMIT = 0.045
    ROTATION_LIMIT = np.radians(20.0)
    IK_STEPS = 60
    DONE_MESSAGE = "routine complete"
    TELEMETRY_PREFIX = "task"

    def __init__(self, env, *, arm: str = "") -> None:
        del arm
        self.env = env
        self.reset()

    @property
    def job(self):
        return self.JOBS[self.job_index]

    @property
    def ik(self):
        return self.env.arm(self.job.arm_name)

    @property
    def tool(self):
        return self.env.handles.objects[self.job.object_name]

    def reset(self) -> None:
        self.job_index = 0
        self.phase = "idle"
        self.frame = 0
        self.message = ""
        self.waypoints: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.errors: dict[str, tuple[float, float]] = {}
        self.targets: dict[str, np.ndarray] = {}
        self.home_q: np.ndarray | None = None
        self.grip = dict.fromkeys(self.env.scene.embodiment.iks, 0.0)

    @staticmethod
    def world_point(pose: np.ndarray, local: np.ndarray) -> np.ndarray:
        """Transform one mesh-local landmark into the world frame."""
        return np.asarray(pose[:3], dtype=np.float64) + quat_to_mat(pose[3:7]) @ local

    def solve(self, seed: np.ndarray, phase: str, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Solve and record one full-pose IK waypoint."""
        q = self.ik.solve_q_full(seed, pos, quat, iterations=self.IK_STEPS)
        pos_err, rot_err = self.ik.pose_errors(q, pos, quat)
        self.errors[f"{self.job.object_name}/{phase}"] = (float(pos_err), float(rot_err))
        self.targets[phase] = np.asarray(pos, dtype=np.float64)
        if pos_err > self.POSITION_LIMIT or rot_err > self.ROTATION_LIMIT:
            raise RuntimeError(
                f"{self.job.object_name} {self.job.arm_name}-arm IK failed at {phase}: "
                f"position {pos_err * 1000:.0f} mm, rotation {np.degrees(rot_err):.1f} deg"
            )
        return q

    def home_from(self, q: np.ndarray) -> np.ndarray:
        """Return a waypoint with only the active arm restored to its home joints."""
        home = q.copy()
        for coord in self.ik.arm_coords:
            home[coord] = self.home_q[coord]
        return home

    def start_plan(self, waypoints: dict[str, tuple[np.ndarray, np.ndarray]], message: str) -> None:
        self.waypoints = waypoints
        self.grip[self.job.arm_name] = 0.0
        self.phase = self.PHASES[0]
        self.frame = 0
        self.message = message

    def act(self) -> np.ndarray:
        if self.phase == "idle":
            try:
                self.plan()
            except Exception as exc:
                self.phase = "done"
                self.message = str(exc)
                self.env.request_termination(self.message)
                return self.hold_action()
        if self.phase == "done":
            return self.hold_action()

        duration = self.DURATIONS[self.phase]
        t = min(1.0, self.frame / max(1, duration - 1))
        alpha = t * t * (3.0 - 2.0 * t)
        q0, q1 = self.waypoints[self.phase]
        command = q0 + alpha * (q1 - q0)

        if self.phase == "close":
            self.grip[self.job.arm_name] = t
        elif self.phase in self.GRIP_PHASES:
            self.grip[self.job.arm_name] = 1.0
        elif self.phase == "open":
            self.grip[self.job.arm_name] = 1.0 - t
        else:
            self.grip[self.job.arm_name] = 0.0

        self.frame += 1
        if self.frame >= duration:
            self.frame = 0
            self.advance()
        return self.action_for(command)

    def advance(self) -> None:
        previous = self.phase
        index = self.PHASES.index(previous)
        if index + 1 < len(self.PHASES):
            self.phase = self.PHASES[index + 1]
        elif self.job_index + 1 < len(self.JOBS):
            completed = self.job
            self.job_index += 1
            self.phase = "idle"
            self.message = self.job_done_message(completed)
            return
        else:
            self.phase = "done"
            self.message = self.DONE_MESSAGE
            self.env.request_termination(self.message)
            return
        self.phase_message(previous)

    def phase_message(self, previous: str) -> None:
        if previous == "descend":
            actual, _ = self.ik.ee_pose(self.env.state)
            error = float(np.linalg.norm(actual - self.targets["grasp"]))
            self.message = f"{self.job.object_name} at grasp ({error * 1000:.0f} mm error); closing"

    def job_done_message(self, completed) -> str:
        return f"finished {completed.object_name}; starting {self.job.object_name}"

    def action_for(self, q_full: np.ndarray) -> np.ndarray:
        parts = []
        for arm, ik in self.env.scene.embodiment.iks.items():
            joints = np.array([q_full[c] for c in ik.arm_coords], dtype=np.float32)
            parts.append(np.append(joints, np.float32(self.grip[arm])))
        return np.concatenate(parts).astype(np.float32)

    def hold_action(self) -> np.ndarray:
        return self.action_for(np.asarray(self.env.state.joint_q.numpy(), dtype=np.float32))

    def row_extras(self) -> dict[str, object]:
        active = self.job.object_name if self.phase != "done" else "complete"
        key = self.TELEMETRY_PREFIX
        return {f"{key}_phase": self.phase, f"{key}_object": active, f"{key}_message": self.message}

    def get_action(self, obs) -> np.ndarray:
        del obs
        return self.act()


# ------------------------------------------------------------------------ the wrenches

# The shafts' broad, nearly uniform middle sections are stable pinch points,
# close to each mesh's measured mass centre and safely away from both jaws and
# ring.  Ring centres are the exact feature landmarks declared in records.py.
WRENCH_150_GRASP_CENTER_LOCAL = np.array([0.066, 0.0, 0.0], dtype=np.float64)
WRENCH_150_RING_CENTER_LOCAL = np.array([0.11011492, 0.0, 0.0], dtype=np.float64)
WRENCH_200_GRASP_CENTER_LOCAL = np.array([0.070, 0.0, 0.0], dtype=np.float64)
WRENCH_200_RING_CENTER_LOCAL = np.array([0.13076606, 0.0, 0.0], dtype=np.float64)

LEFT_WRENCH_HOOK_CENTER_WORLD_Y = 0.060
RIGHT_WRENCH_HOOK_CENTER_WORLD_Y = -0.060
WRENCH_LIFT_ABOVE_GRASP = 0.180
WRENCH_RING_CLEARANCE_ABOVE_TIP = 0.0
WRENCH_PASS_HOOK_TIP = 0.008
WRENCH_LOWER_ONTO_HOOK = 0.010
# Bias the mathematical grasp site toward the top face of the shaft so the hand
# retains clearance from the table while the fingers still close on the steel.
WRENCH_GRASP_SITE_ABOVE_SHAFT_CENTER = 0.002
WRENCH_IK_ITERATIONS = 100


@dataclass(frozen=True)
class WrenchHangJob:
    """One wrench's arm, hook, pinch point, and measured ring centre."""

    object_name: str
    arm_name: str
    hook_center_world_y: float
    grasp_center_local: np.ndarray
    ring_center_local: np.ndarray


WRENCH_HANG_JOBS = (
    WrenchHangJob(
        "ring_spanner_19",
        "left",
        LEFT_WRENCH_HOOK_CENTER_WORLD_Y,
        WRENCH_150_GRASP_CENTER_LOCAL,
        WRENCH_150_RING_CENTER_LOCAL,
    ),
    WrenchHangJob(
        "ring_spanner_17",
        "right",
        RIGHT_WRENCH_HOOK_CENTER_WORLD_Y,
        WRENCH_200_GRASP_CENTER_LOCAL,
        WRENCH_200_RING_CENTER_LOCAL,
    ),
)


class HangWrenchRoutine(TwoArmRoutine):
    """Hang the left wrench on the left hook, then the right on the right."""

    JOBS = WRENCH_HANG_JOBS
    PHASES = (
        "approach",
        "descend",
        "close",
        "lift",
        "orient_vertical",
        "move_to_pre_hook",
        "correct_pre_hook",
        "insert_ring_over_tip",
        "lower_onto_tip",
        "open",
        "retract_from_hook",
        "return_home",
        "settle",
    )
    DURATIONS: ClassVar[dict[str, int]] = {
        "approach": 70,
        "descend": 60,
        "close": 60,
        "lift": 90,
        "orient_vertical": 150,
        "move_to_pre_hook": 120,
        "correct_pre_hook": 100,
        "insert_ring_over_tip": 90,
        "lower_onto_tip": 70,
        "open": 55,
        "retract_from_hook": 90,
        "return_home": 90,
        "settle": 120,
    }
    GRIP_PHASES = frozenset(
        ("lift", "orient_vertical", "move_to_pre_hook", "correct_pre_hook", "insert_ring_over_tip", "lower_onto_tip")
    )
    POSITION_LIMIT = POSITION_ERROR_LIMIT
    ROTATION_LIMIT = ROTATION_ERROR_LIMIT
    IK_STEPS = WRENCH_IK_ITERATIONS
    LIFT_ABOVE_GRASP: ClassVar[float] = WRENCH_LIFT_ABOVE_GRASP
    DONE_MESSAGE = "both wrenches hung; both arms returned home"
    TELEMETRY_PREFIX = "hang"

    def reset(self) -> None:
        super().reset()
        self.live_hook_replan_pending = False
        self.live_pre_hook_correction_pending = False

    def act(self) -> np.ndarray:
        # ``advance`` runs while producing the final vertical command; the
        # environment applies that command after ``act`` returns.  Replan on
        # the following control tick, when both the hand and wrench poses
        # therefore reflect the completed physical rotation.
        if self.live_hook_replan_pending and self.phase == "move_to_pre_hook":
            self.replan_hook_from_live_pose()
            self.live_hook_replan_pending = False
        if self.live_pre_hook_correction_pending and self.phase == "correct_pre_hook":
            self.correct_pre_hook_from_live_pose()
            self.live_pre_hook_correction_pending = False
        return super().act()

    def advance(self) -> None:
        previous = self.phase
        super().advance()
        if previous == "orient_vertical" and self.phase == "move_to_pre_hook":
            self.live_hook_replan_pending = True
        elif previous == "move_to_pre_hook" and self.phase == "correct_pre_hook":
            self.live_pre_hook_correction_pending = True

    def correct_pre_hook_from_live_pose(self) -> None:
        """Cancel loaded-controller tracking error before moving inward."""
        q_now = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float32)
        object_pose = self.tool.pose(self.env.state.body_q.numpy())
        ring_world = self.world_point(object_pose, self.job.ring_center_local)
        ring_error = self.live_ring_targets["pre"] - ring_world
        tcp_world, _ = self.ik.ee_pose(self.env.state)
        # ``tcp_world + ring_error`` is the nominal waypoint already being
        # commanded; the loaded controller sits below it by ``ring_error``.
        # Over-command by that measured residual to cancel the steady sag.
        corrected_tcp = tcp_world + 2.0 * ring_error
        q_corrected = self.solve(q_now, "correct_pre_hook", corrected_tcp, self.upright_quat)

        # Re-measure the held transform at the pre-hook configuration and use
        # it for the remaining short insertion and lowering motions.
        _, tcp_quat = self.ik.ee_pose(self.env.state)
        ring_from_tcp = quat_to_mat(tcp_quat).T @ (ring_world - tcp_world)

        def tcp_for_ring(ring_target: np.ndarray) -> np.ndarray:
            return ring_target - quat_to_mat(self.upright_quat) @ ring_from_tcp + ring_error

        insert_grasp = tcp_for_ring(self.live_ring_targets["insert"])
        loaded_grasp = tcp_for_ring(self.live_ring_targets["loaded"])
        retracted = loaded_grasp + np.array([-RETRACT_AWAY_FROM_BOARD, 0.0, RETRACT_UPWARD])
        q_insert = self.solve(q_corrected, "insert_ring_over_tip", insert_grasp, self.upright_quat)
        q_loaded = self.solve(q_insert, "lower_onto_tip", loaded_grasp, self.upright_quat)
        q_retracted = self.solve(q_loaded, "retract_from_hook", retracted, self.upright_quat)
        q_home = self.home_from(q_retracted)
        self.waypoints.update(
            {
                "correct_pre_hook": (q_now, q_corrected),
                "insert_ring_over_tip": (q_corrected, q_insert),
                "lower_onto_tip": (q_insert, q_loaded),
                "open": (q_loaded, q_loaded),
                "retract_from_hook": (q_loaded, q_retracted),
                "return_home": (q_retracted, q_home),
                "settle": (q_home, q_home),
            }
        )
        self.message = (
            f"correcting {self.job.object_name} pre-hook tracking by {np.linalg.norm(ring_error) * 1000:.1f} mm"
        )

    def replan_hook_from_live_pose(self) -> None:
        """Replace hook waypoints using the ring's post-rotation held pose."""
        q_now = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float32)
        object_pose = self.tool.pose(self.env.state.body_q.numpy())
        ring_world = self.world_point(object_pose, self.job.ring_center_local)
        tcp_world, tcp_quat = self.ik.ee_pose(self.env.state)
        ring_from_tcp = quat_to_mat(tcp_quat).T @ (ring_world - tcp_world)

        def tcp_for_ring(ring_target: np.ndarray) -> np.ndarray:
            return ring_target - quat_to_mat(self.upright_quat) @ ring_from_tcp

        pre_grasp = tcp_for_ring(self.live_ring_targets["pre"])
        insert_grasp = tcp_for_ring(self.live_ring_targets["insert"])
        loaded_grasp = tcp_for_ring(self.live_ring_targets["loaded"])
        retracted = loaded_grasp + np.array([-RETRACT_AWAY_FROM_BOARD, 0.0, RETRACT_UPWARD])

        q_pre = self.solve(q_now, "move_to_pre_hook", pre_grasp, self.upright_quat)
        q_insert = self.solve(q_pre, "insert_ring_over_tip", insert_grasp, self.upright_quat)
        q_loaded = self.solve(q_insert, "lower_onto_tip", loaded_grasp, self.upright_quat)
        q_retracted = self.solve(q_loaded, "retract_from_hook", retracted, self.upright_quat)
        q_home = self.home_from(q_retracted)
        self.waypoints.update(
            {
                "move_to_pre_hook": (q_now, q_pre),
                "insert_ring_over_tip": (q_pre, q_insert),
                "lower_onto_tip": (q_insert, q_loaded),
                "open": (q_loaded, q_loaded),
                "retract_from_hook": (q_loaded, q_retracted),
                "return_home": (q_retracted, q_home),
                "settle": (q_home, q_home),
            }
        )
        correction = ring_world - (tcp_world + quat_to_mat(self.upright_quat) @ self.planned_ring_from_tcp)
        self.message = (
            f"measured {self.job.object_name} after rotation; replanned ring path "
            f"({np.linalg.norm(correction) * 1000:.1f} mm grasp correction)"
        )

    def plan(self) -> None:
        object_pose = self.tool.pose(self.env.state.body_q.numpy())
        grasp_world = self.world_point(object_pose, self.job.grasp_center_local)
        # The hand's collision geometry rides slightly above its mathematical
        # grasp site when it reaches the tabletop. Aim slightly above the shaft
        # centre so the fingers close on the steel while retaining clearance.
        grasp_world[2] += WRENCH_GRASP_SITE_ABOVE_SHAFT_CENTER
        seed = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float32)
        if self.home_q is None:
            self.home_q = seed.copy()

        hand = self.env.scene.embodiment.hands[self.job.arm_name]
        tabletop_quat = np.asarray(top_down_grasp_site_quat(hand, 90.0), dtype=np.float64)
        upright_rotation = quat_from_axis_angle((0.0, 1.0, 0.0), -np.pi / 2.0)
        upright_quat = quat_multiply(upright_rotation, tabletop_quat)
        self.upright_quat = upright_quat
        approach = grasp_world + np.array([0.0, 0.0, APPROACH_ABOVE_GRASP])
        lifted = grasp_world + np.array([0.0, 0.0, self.LIFT_ABOVE_GRASP])

        grasp_to_ring_local = self.job.ring_center_local - self.job.grasp_center_local
        grasp_to_ring_upright = quat_to_mat(upright_rotation) @ quat_to_mat(object_pose[3:7]) @ grasp_to_ring_local
        inserted_ring = np.array(
            [
                HOOK_TIP_WORLD_X + WRENCH_PASS_HOOK_TIP,
                self.job.hook_center_world_y,
                HOOK_TIP_WORLD_Z + WRENCH_RING_CLEARANCE_ABOVE_TIP,
            ],
            dtype=np.float64,
        )
        pre_ring = inserted_ring + np.array([-PRE_HOOK_DISTANCE_IN_FRONT - WRENCH_PASS_HOOK_TIP, 0.0, 0.0])
        loaded_ring = inserted_ring + np.array([0.0, 0.0, -WRENCH_LOWER_ONTO_HOOK])
        self.live_ring_targets = {
            "pre": pre_ring,
            "insert": inserted_ring,
            "loaded": loaded_ring,
        }
        pre_grasp = pre_ring - grasp_to_ring_upright
        insert_grasp = inserted_ring - grasp_to_ring_upright
        loaded_grasp = loaded_ring - grasp_to_ring_upright
        retracted = loaded_grasp + np.array([-RETRACT_AWAY_FROM_BOARD, 0.0, RETRACT_UPWARD])

        q_approach = self.solve(seed, "approach", approach, tabletop_quat)
        q_grasp = self.solve(q_approach, "grasp", grasp_world, tabletop_quat)
        # A second pass is important for the thin shaft: it removes the
        # several-millimetre residual that can otherwise close above it.
        q_grasp = self.solve(q_grasp, "grasp", grasp_world, tabletop_quat)
        q_lift = self.solve(q_grasp, "lift", lifted, tabletop_quat)
        q_vertical = self.solve(q_lift, "orient_vertical", lifted, upright_quat)

        # Correct each hook waypoint at the ring, rather than trusting TCP
        # accuracy.  Even a small wrist-angle residual is magnified by the
        # 75--91 mm lever arm from the pinch point to these tiny openings.
        ring_from_tcp = quat_to_mat(upright_quat).T @ grasp_to_ring_upright
        self.planned_ring_from_tcp = ring_from_tcp

        def solve_ring_aware(seed_q, phase, tcp_target, ring_target):
            q = self.solve(seed_q, phase, tcp_target, upright_quat)
            tcp_actual, quat_actual = self.ik.fk_ee(q)
            ring_actual = tcp_actual + quat_to_mat(quat_actual) @ ring_from_tcp
            return self.solve(q, phase, tcp_target + ring_target - ring_actual, upright_quat)

        q_pre = solve_ring_aware(q_vertical, "move_to_pre_hook", pre_grasp, pre_ring)
        q_insert = solve_ring_aware(q_pre, "insert_ring_over_tip", insert_grasp, inserted_ring)
        q_loaded = solve_ring_aware(q_insert, "lower_onto_tip", loaded_grasp, loaded_ring)
        q_retracted = self.solve(q_loaded, "retract_from_hook", retracted, upright_quat)
        q_home = self.home_from(q_retracted)
        self.start_plan(
            {
                "approach": (seed, q_approach),
                "descend": (q_approach, q_grasp),
                "close": (q_grasp, q_grasp),
                "lift": (q_grasp, q_lift),
                "orient_vertical": (q_lift, q_vertical),
                "move_to_pre_hook": (q_vertical, q_pre),
                "correct_pre_hook": (q_pre, q_pre),
                "insert_ring_over_tip": (q_pre, q_insert),
                "lower_onto_tip": (q_insert, q_loaded),
                "open": (q_loaded, q_loaded),
                "retract_from_hook": (q_loaded, q_retracted),
                "return_home": (q_retracted, q_home),
                "settle": (q_home, q_home),
            },
            f"planned {self.job.object_name} for {self.job.arm_name} hook-tip alignment",
        )

    def phase_message(self, previous: str) -> None:
        super().phase_message(previous)
        if previous == "insert_ring_over_tip":
            self.message = f"{self.job.object_name} ring cleared hook tip; loading"
        elif previous == "open":
            self.message = f"released {self.job.object_name}; retracting {self.job.arm_name} arm"

    def job_done_message(self, completed: WrenchHangJob) -> str:
        return f"hung {completed.object_name}; starting {self.job.object_name}"


# One centred large wrench, using its measured 200 mm mesh landmarks.
WRENCH_HANG_JOBS_LEVEL1 = (
    WrenchHangJob(
        "wrench_1",
        "left",
        LEVEL1_Y,
        WRENCH_200_GRASP_CENTER_LOCAL * 1.30,
        WRENCH_200_RING_CENTER_LOCAL * 1.30,
    ),
)


class HangWrenchRoutineLevel1(HangWrenchRoutine):
    """:class:`HangWrenchRoutine` trimmed to wrenches_level1.task's one job."""

    JOBS = WRENCH_HANG_JOBS_LEVEL1
    LIFT_ABOVE_GRASP = 0.180
    DONE_MESSAGE = "wrench hung; arm returned home"


# ------------------------------------------------------------------------ the screwdrivers

SCREWDRIVER_1_HANDLE_CENTER_LOCAL = np.array([0.02835, 0.00025, 0.00733], dtype=np.float64)
SCREWDRIVER_1_TIP_LOCAL = np.array([0.175, 0.0, 0.0], dtype=np.float64)
SCREWDRIVER_2_HANDLE_CENTER_LOCAL = np.array([0.060, 0.0, 0.0], dtype=np.float64)
SCREWDRIVER_2_TIP_LOCAL = np.array([-0.100, -0.005697, 0.001303], dtype=np.float64)

# Match the yellow left and purple right hole markers in suite.py.
SCREWDRIVER_HOLE_CENTER_WORLD_X = 0.856
SCREWDRIVER_LEFT_HOLE_WORLD_Y = 0.0198
SCREWDRIVER_RIGHT_HOLE_WORLD_Y = -0.0198
SCREWDRIVER_HOLDER_TOP_WORLD_Z = 0.8836

SCREWDRIVER_LIFT_ABOVE_GRASP = 0.185
TIP_CLEARANCE_ABOVE_HOLE = 0.030
INSERTION_DEPTH = 0.060
RETRACT_ABOVE_HOLE = 0.080
SCREWDRIVER_IK_ITERATIONS = 80


@dataclass(frozen=True)
class ScrewdriverInsertJob:
    """Geometry and arm selection for one insertion."""

    object_name: str
    arm_name: str
    hole_center_world_y: float
    handle_center_local: np.ndarray
    tip_local: np.ndarray


SCREWDRIVER_INSERT_JOBS = (
    ScrewdriverInsertJob(
        object_name="screwdriver_1",
        arm_name="left",
        hole_center_world_y=SCREWDRIVER_LEFT_HOLE_WORLD_Y,
        handle_center_local=SCREWDRIVER_1_HANDLE_CENTER_LOCAL,
        tip_local=SCREWDRIVER_1_TIP_LOCAL,
    ),
    ScrewdriverInsertJob(
        object_name="screwdriver_2",
        arm_name="right",
        hole_center_world_y=SCREWDRIVER_RIGHT_HOLE_WORLD_Y,
        handle_center_local=SCREWDRIVER_2_HANDLE_CENTER_LOCAL,
        tip_local=SCREWDRIVER_2_TIP_LOCAL,
    ),
)


class InsertScrewdriverRoutine(TwoArmRoutine):
    """Insert the left screwdriver, then the right, and return both arms home."""

    JOBS = SCREWDRIVER_INSERT_JOBS
    PHASES = (
        "approach",
        "descend",
        "close",
        "lift",
        "orient_vertical",
        "raise_for_holder_clearance",
        "move_above_hole",
        "dwell_above_hole",
        "insert",
        "open",
        "retract",
        "return_home",
        "settle",
    )

    DURATIONS: ClassVar[dict[str, int]] = {
        "approach": 70,
        "descend": 55,
        "close": 50,
        "lift": 75,
        "orient_vertical": 150,
        "raise_for_holder_clearance": 90,
        "move_above_hole": 120,
        "dwell_above_hole": 60,
        "insert": 100,
        "open": 55,
        "retract": 90,
        "return_home": 90,
        "settle": 90,
    }
    GRIP_PHASES = frozenset(
        (
            "lift",
            "orient_vertical",
            "raise_for_holder_clearance",
            "move_above_hole",
            "dwell_above_hole",
            "insert",
        )
    )
    POSITION_LIMIT = POSITION_ERROR_LIMIT
    ROTATION_LIMIT = ROTATION_ERROR_LIMIT
    IK_STEPS = SCREWDRIVER_IK_ITERATIONS
    LIFT_ABOVE_GRASP: ClassVar[float] = SCREWDRIVER_LIFT_ABOVE_GRASP
    DONE_MESSAGE = "both screwdrivers inserted; both arms returned home"
    TELEMETRY_PREFIX = "insert"

    def plan(self) -> None:
        object_pose = self.tool.pose(self.env.state.body_q.numpy())
        grasp_world = self.world_point(object_pose, self.job.handle_center_local)
        seed = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float32)
        if self.home_q is None:
            self.home_q = seed.copy()

        hand = self.env.scene.embodiment.hands[self.job.arm_name]
        tabletop_orientation = np.asarray(top_down_grasp_site_quat(hand, 90.0), dtype=np.float64)
        upright_rotation = quat_from_axis_angle((0.0, 1.0, 0.0), -np.pi / 2.0)
        upright_orientation = quat_multiply(upright_rotation, tabletop_orientation)

        approach_world = grasp_world + np.array([0.0, 0.0, APPROACH_ABOVE_GRASP])
        lifted_world = grasp_world + np.array([0.0, 0.0, self.LIFT_ABOVE_GRASP])

        # Offset the gripper so the screwdriver tip reaches each target.
        grasp_to_tip_local = self.job.tip_local - self.job.handle_center_local
        grasp_to_tip_upright = quat_to_mat(upright_rotation) @ quat_to_mat(object_pose[3:7]) @ grasp_to_tip_local
        hole_center = np.array(
            [
                SCREWDRIVER_HOLE_CENTER_WORLD_X,
                self.job.hole_center_world_y,
                SCREWDRIVER_HOLDER_TOP_WORLD_Z,
            ],
            dtype=np.float64,
        )
        aligned_tip = hole_center + np.array([0.0, 0.0, TIP_CLEARANCE_ABOVE_HOLE])
        inserted_tip = hole_center + np.array([0.0, 0.0, -INSERTION_DEPTH])
        aligned_grasp = aligned_tip - grasp_to_tip_upright
        inserted_grasp = inserted_tip - grasp_to_tip_upright
        retracted_grasp = inserted_grasp + np.array([0.0, 0.0, RETRACT_ABOVE_HOLE])

        # Clear the holder vertically before moving over the hole.
        raised_grasp = lifted_world.copy()
        raised_grasp[2] = aligned_grasp[2]

        q_approach = self.solve(seed, "approach", approach_world, tabletop_orientation)
        q_grasp = self.solve(q_approach, "grasp", grasp_world, tabletop_orientation)
        q_lift = self.solve(q_grasp, "lift", lifted_world, tabletop_orientation)
        q_upright = self.solve(q_lift, "orient_vertical", lifted_world, upright_orientation)
        q_raised = self.solve(q_upright, "raise_for_holder_clearance", raised_grasp, upright_orientation)
        q_aligned = self.solve(q_raised, "move_above_hole", aligned_grasp, upright_orientation)

        # A long tool magnifies a small wrist-angle residual at its tip, so
        # compensate the insertion TCP target by the predicted tip error.
        tip_from_tcp = quat_to_mat(upright_orientation).T @ grasp_to_tip_upright

        def solve_tip_aware(seed_q, phase, tcp_target, tip_target):
            q = self.solve(seed_q, phase, tcp_target, upright_orientation)
            tcp_actual, quat_actual = self.ik.fk_ee(q)
            tip_actual = tcp_actual + quat_to_mat(quat_actual) @ tip_from_tcp
            corrected_tcp = tcp_target + (tip_target - tip_actual)
            return self.solve(q, phase, corrected_tcp, upright_orientation)

        q_inserted = solve_tip_aware(q_aligned, "insert", inserted_grasp, inserted_tip)
        q_retracted = self.solve(q_inserted, "retract", retracted_grasp, upright_orientation)
        q_home = self.home_from(q_retracted)

        waypoints = {
            "approach": (seed, q_approach),
            "descend": (q_approach, q_grasp),
            "close": (q_grasp, q_grasp),
            "lift": (q_grasp, q_lift),
            "orient_vertical": (q_lift, q_upright),
            "raise_for_holder_clearance": (q_upright, q_raised),
            "move_above_hole": (q_raised, q_aligned),
            "dwell_above_hole": (q_aligned, q_aligned),
            "insert": (q_aligned, q_inserted),
            "open": (q_inserted, q_inserted),
            "retract": (q_inserted, q_retracted),
            "return_home": (q_retracted, q_home),
            "settle": (q_home, q_home),
        }
        self.start_plan(
            waypoints,
            f"planned {self.job.object_name} insertion with {self.job.arm_name} arm",
        )

    def phase_message(self, previous: str) -> None:
        super().phase_message(previous)
        if previous == "raise_for_holder_clearance":
            self.message = f"{self.job.object_name} raised clear of holder; moving horizontally"
        elif previous == "move_above_hole":
            self.message = f"{self.job.object_name} tip aligned above hole; dwelling"
        elif previous == "dwell_above_hole":
            self.message = f"{self.job.object_name} tip aligned above hole; inserting"
        elif previous == "open":
            self.message = f"released {self.job.object_name}; retracting {self.job.arm_name} arm"

    def job_done_message(self, completed: ScrewdriverInsertJob) -> str:
        return f"inserted {completed.object_name}; starting {self.job.object_name}"


# screwdrivers_level1 uses the small screwdriver_slim with its original pair-1
# handle/tip landmarks and assigns the one centred job to the left arm.
SCREWDRIVER_INSERT_JOBS_LEVEL1 = (
    ScrewdriverInsertJob(
        object_name="screwdriver_1",
        arm_name="left",
        hole_center_world_y=LEVEL1_Y,
        handle_center_local=SCREWDRIVER_1_HANDLE_CENTER_LOCAL,
        tip_local=SCREWDRIVER_1_TIP_LOCAL,
    ),
)


class InsertScrewdriverRoutineLevel1(InsertScrewdriverRoutine):
    """:class:`InsertScrewdriverRoutine` trimmed to screwdrivers_level1.task's one job."""

    JOBS = SCREWDRIVER_INSERT_JOBS_LEVEL1
    LIFT_ABOVE_GRASP = 0.200
    DONE_MESSAGE = "screwdriver inserted; arm returned home"


# ------------------------------------------------------------------------ the scissors

SCISSOR_1_GRASP_CENTER_LOCAL = np.array([0.000, -0.018, 0.000], dtype=np.float64)
SCISSOR_1_HANDLE_CENTER_LOCAL = np.array([0.023, -0.044, 0.000], dtype=np.float64)
SCISSOR_2_GRASP_CENTER_LOCAL = np.array([0.000, 0.020, -0.00025], dtype=np.float64)
SCISSOR_2_HANDLE_CENTER_LOCAL = np.array([-0.011, 0.059, -0.00025], dtype=np.float64)

# Hook positions in world coordinates.
LEFT_HOOK_CENTER_WORLD_Y = 0.100
RIGHT_HOOK_CENTER_WORLD_Y = -0.100

# Motion clearances in meters.
SCISSOR_LIFT_ABOVE_GRASP = 0.190
HANDLE_HEIGHT_ABOVE_HOOK_TIP = 0.010
SCISSOR_LOWER_ONTO_HOOK_DISTANCE = 0.015
# Chained solutions need only 60 IK iterations.
SCISSOR_IK_ITERATIONS = 60


@dataclass(frozen=True)
class ScissorHangJob:
    """All object-specific choices for one otherwise shared hanging motion."""

    object_name: str
    arm_name: str
    hook_center_world_y: float
    grasp_center_local: np.ndarray
    handle_center_local: np.ndarray


SCISSOR_HANG_JOBS = (
    ScissorHangJob(
        object_name="scissors_1",
        arm_name="left",
        hook_center_world_y=LEFT_HOOK_CENTER_WORLD_Y,
        grasp_center_local=SCISSOR_1_GRASP_CENTER_LOCAL,
        handle_center_local=SCISSOR_1_HANDLE_CENTER_LOCAL,
    ),
    ScissorHangJob(
        object_name="scissors_2",
        arm_name="right",
        hook_center_world_y=RIGHT_HOOK_CENTER_WORLD_Y,
        grasp_center_local=SCISSOR_2_GRASP_CENTER_LOCAL,
        handle_center_local=SCISSOR_2_HANDLE_CENTER_LOCAL,
    ),
)


class HangScissorRoutine(TwoArmRoutine):
    """Hang two scissors sequentially, returning each working arm home."""

    JOBS = SCISSOR_HANG_JOBS
    PHASES = (
        "approach",
        "descend",
        "close",
        "lift",
        "orient_vertical",
        "move_to_pre_hook",
        "insert_over_hook",
        "lower_onto_hook",
        "open",
        "retract_from_hook",
        "return_home",
        "settle",
    )

    DURATIONS: ClassVar[dict[str, int]] = {
        "approach": 70,
        "descend": 55,
        "close": 50,
        "lift": 75,
        "orient_vertical": 150,
        "move_to_pre_hook": 100,
        "insert_over_hook": 70,
        "lower_onto_hook": 55,
        "open": 45,
        "retract_from_hook": 75,
        "return_home": 80,
        "settle": 90,
    }
    GRIP_PHASES = frozenset(PHASES[3:8])
    POSITION_LIMIT = POSITION_ERROR_LIMIT
    ROTATION_LIMIT = ROTATION_ERROR_LIMIT
    IK_STEPS = SCISSOR_IK_ITERATIONS
    LIFT_ABOVE_GRASP: ClassVar[float] = SCISSOR_LIFT_ABOVE_GRASP
    DONE_MESSAGE = "both scissors hung; both arms returned home"
    TELEMETRY_PREFIX = "hang"

    def plan(self) -> None:
        """Measure the active scissor and solve its complete pick-and-hang path."""
        object_pose = self.tool.pose(self.env.state.body_q.numpy())
        grasp_center_world = self.world_point(object_pose, self.job.grasp_center_local)
        seed = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float32)
        if self.home_q is None:
            self.home_q = seed.copy()

        hand = self.env.scene.embodiment.hands[self.job.arm_name]
        tabletop_grasp_orientation = np.asarray(top_down_grasp_site_quat(hand, 90.0), dtype=np.float64)
        rotate_scissor_vertical = quat_from_axis_angle((0.0, 1.0, 0.0), -np.pi / 2.0)
        vertical_grasp_orientation = quat_multiply(rotate_scissor_vertical, tabletop_grasp_orientation)

        approach_position = grasp_center_world + np.array([0.0, 0.0, APPROACH_ABOVE_GRASP])
        lifted_position = grasp_center_world + np.array([0.0, 0.0, self.LIFT_ABOVE_GRASP])

        # Offset the gripper so the handle center reaches the hook.
        grasp_to_handle_local = self.job.handle_center_local - self.job.grasp_center_local
        grasp_to_handle_vertical = (
            quat_to_mat(rotate_scissor_vertical) @ quat_to_mat(object_pose[3:7]) @ grasp_to_handle_local
        )
        handle_goal_past_hook_tip = np.array(
            [
                HOOK_TIP_WORLD_X + INSERT_DISTANCE_PAST_HOOK_TIP,
                self.job.hook_center_world_y,
                HOOK_TIP_WORLD_Z + HANDLE_HEIGHT_ABOVE_HOOK_TIP,
            ],
            dtype=np.float64,
        )
        insert_grasp_position = handle_goal_past_hook_tip - grasp_to_handle_vertical
        pre_hook_grasp_position = insert_grasp_position + np.array(
            [-PRE_HOOK_DISTANCE_IN_FRONT - INSERT_DISTANCE_PAST_HOOK_TIP, 0.0, 0.0]
        )
        loaded_grasp_position = insert_grasp_position + np.array([0.0, 0.0, -SCISSOR_LOWER_ONTO_HOOK_DISTANCE])
        retracted_grasp_position = loaded_grasp_position + np.array([-RETRACT_AWAY_FROM_BOARD, 0.0, RETRACT_UPWARD])

        q_approach = self.solve(seed, "approach", approach_position, tabletop_grasp_orientation)
        q_grasp = self.solve(q_approach, "grasp", grasp_center_world, tabletop_grasp_orientation)
        q_lift = self.solve(q_grasp, "lift", lifted_position, tabletop_grasp_orientation)
        q_vertical = self.solve(q_lift, "orient_vertical", lifted_position, vertical_grasp_orientation)
        q_pre_hook = self.solve(q_vertical, "move_to_pre_hook", pre_hook_grasp_position, vertical_grasp_orientation)
        q_insert = self.solve(q_pre_hook, "insert_over_hook", insert_grasp_position, vertical_grasp_orientation)
        q_loaded = self.solve(q_insert, "lower_onto_hook", loaded_grasp_position, vertical_grasp_orientation)
        q_retracted = self.solve(q_loaded, "retract_from_hook", retracted_grasp_position, vertical_grasp_orientation)
        q_home = self.home_from(q_retracted)

        waypoints = {
            "approach": (seed, q_approach),
            "descend": (q_approach, q_grasp),
            "close": (q_grasp, q_grasp),
            "lift": (q_grasp, q_lift),
            "orient_vertical": (q_lift, q_vertical),
            "move_to_pre_hook": (q_vertical, q_pre_hook),
            "insert_over_hook": (q_pre_hook, q_insert),
            "lower_onto_hook": (q_insert, q_loaded),
            "open": (q_loaded, q_loaded),
            "retract_from_hook": (q_loaded, q_retracted),
            "return_home": (q_retracted, q_home),
            "settle": (q_home, q_home),
        }
        message = (
            f"planned {self.job.object_name} with {self.job.arm_name} arm; "
            f"handle target y={self.job.hook_center_world_y:+.3f}"
        )
        self.start_plan(waypoints, message)

    def phase_message(self, previous: str) -> None:
        super().phase_message(previous)
        if previous == "insert_over_hook":
            self.message = f"{self.job.object_name} handle passed hook tip; lowering"
        elif previous == "open":
            self.message = f"released {self.job.object_name}; retracting {self.job.arm_name} arm"

    def job_done_message(self, completed: ScissorHangJob) -> str:
        return f"{completed.arm_name} arm returned home; starting {self.job.object_name}"


# One centred pair-1 scissors, retaining its measured landmarks.
SCISSOR_HANG_JOBS_LEVEL1 = (
    ScissorHangJob(
        object_name="scissors_1",
        arm_name="left",
        hook_center_world_y=LEVEL1_Y,
        grasp_center_local=SCISSOR_1_GRASP_CENTER_LOCAL,
        handle_center_local=SCISSOR_1_HANDLE_CENTER_LOCAL,
    ),
)


class HangScissorRoutineLevel1(HangScissorRoutine):
    """:class:`HangScissorRoutine` trimmed to scissors_level1.task's one job."""

    JOBS = SCISSOR_HANG_JOBS_LEVEL1
    LIFT_ABOVE_GRASP = 0.190
    DONE_MESSAGE = "scissors hung; arm returned home"


# ------------------------------------------------------------------------ the pliers

PLIER_1_GRASP_CENTER_LOCAL = np.array([0.036, 0.004, -0.0009], dtype=np.float64)
PLIER_1_HANG_AXIS_LOCAL = np.array([-0.010, 0.004, -0.0009], dtype=np.float64)
PLIER_2_GRASP_CENTER_LOCAL = np.array([0.000, -0.015, -0.0035], dtype=np.float64)
PLIER_2_HANG_AXIS_LOCAL = np.array([0.000, -0.045, -0.0035], dtype=np.float64)

LEFT_SUPPORT_WORLD_Y = 0.075
RIGHT_SUPPORT_WORLD_Y = -0.075
SUPPORT_TIP_WORLD_X = 0.85229
SUPPORT_TOP_WORLD_Z = 0.93658

PLIER_LIFT_ABOVE_GRASP = 0.210
AXIS_CLEARANCE_ABOVE_SUPPORT = 0.018
PRE_SUPPORT_DISTANCE = 0.045
PASS_SUPPORT_TIP = 0.003
LOWER_ONTO_SUPPORT = 0.015
RETRACT_FROM_BOARD = 0.100
PLIER_IK_ITERATIONS = 80


@dataclass(frozen=True)
class PlierHangJob:
    object_name: str
    arm_name: str
    support_world_y: float
    grasp_center_local: np.ndarray
    hang_axis_local: np.ndarray


PLIER_HANG_JOBS = (
    PlierHangJob(
        object_name="plier_1",
        arm_name="left",
        support_world_y=LEFT_SUPPORT_WORLD_Y,
        grasp_center_local=PLIER_1_GRASP_CENTER_LOCAL,
        hang_axis_local=PLIER_1_HANG_AXIS_LOCAL,
    ),
    PlierHangJob(
        object_name="plier_2",
        arm_name="right",
        support_world_y=RIGHT_SUPPORT_WORLD_Y,
        grasp_center_local=PLIER_2_GRASP_CENTER_LOCAL,
        hang_axis_local=PLIER_2_HANG_AXIS_LOCAL,
    ),
)


class HangPlierRoutine(TwoArmRoutine):
    """Hang plier 1 with the left arm, then plier 2 with the right arm."""

    JOBS = PLIER_HANG_JOBS
    PHASES = (
        "approach",
        "descend",
        "close",
        "lift",
        "orient_vertical",
        "raise_for_support_clearance",
        "align_above_support",
        "move_onto_support",
        "lower_onto_support",
        "open",
        "retract",
        "return_home",
        "settle",
    )

    DURATIONS: ClassVar[dict[str, int]] = {
        "approach": 70,
        "descend": 55,
        "close": 50,
        "lift": 110,
        "orient_vertical": 150,
        "raise_for_support_clearance": 90,
        "align_above_support": 120,
        "move_onto_support": 80,
        "lower_onto_support": 55,
        "open": 50,
        "retract": 80,
        "return_home": 90,
        "settle": 90,
    }
    GRIP_PHASES = frozenset(PHASES[3:9])
    POSITION_LIMIT = POSITION_ERROR_LIMIT
    ROTATION_LIMIT = ROTATION_ERROR_LIMIT
    IK_STEPS = PLIER_IK_ITERATIONS
    LIFT_ABOVE_GRASP: ClassVar[float] = PLIER_LIFT_ABOVE_GRASP
    DONE_MESSAGE = "both pliers hung; both arms returned home"
    TELEMETRY_PREFIX = "hang"

    def plan(self) -> None:
        object_pose = self.tool.pose(self.env.state.body_q.numpy())
        grasp_world = self.world_point(object_pose, self.job.grasp_center_local)
        seed = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float32)
        if self.home_q is None:
            self.home_q = seed.copy()

        hand = self.env.scene.embodiment.hands[self.job.arm_name]
        tabletop_orientation = np.asarray(top_down_grasp_site_quat(hand, 90.0), dtype=np.float64)
        upright_rotation = quat_from_axis_angle((0.0, 1.0, 0.0), -np.pi / 2.0)
        upright_orientation = quat_multiply(upright_rotation, tabletop_orientation)

        approach_world = grasp_world + np.array([0.0, 0.0, APPROACH_ABOVE_GRASP])
        lifted_world = grasp_world + np.array([0.0, 0.0, self.LIFT_ABOVE_GRASP])

        grasp_to_axis_local = self.job.hang_axis_local - self.job.grasp_center_local
        grasp_to_axis_upright = quat_to_mat(upright_rotation) @ quat_to_mat(object_pose[3:7]) @ grasp_to_axis_local
        loaded_axis = np.array(
            [
                SUPPORT_TIP_WORLD_X + PASS_SUPPORT_TIP,
                self.job.support_world_y,
                SUPPORT_TOP_WORLD_Z + AXIS_CLEARANCE_ABOVE_SUPPORT,
            ],
            dtype=np.float64,
        )
        pre_axis = loaded_axis + np.array([-PRE_SUPPORT_DISTANCE - PASS_SUPPORT_TIP, 0.0, 0.0])
        lowered_axis = loaded_axis + np.array([0.0, 0.0, -LOWER_ONTO_SUPPORT])
        pre_grasp = pre_axis - grasp_to_axis_upright
        loaded_grasp = loaded_axis - grasp_to_axis_upright
        lowered_grasp = lowered_axis - grasp_to_axis_upright
        raised_grasp = lifted_world.copy()
        raised_grasp[0] = 0.5 * (lifted_world[0] + pre_grasp[0])
        raised_grasp[2] = pre_grasp[2]
        retracted_grasp = lowered_grasp + np.array([-RETRACT_FROM_BOARD, 0.0, RETRACT_UPWARD])

        q_approach = self.solve(seed, "approach", approach_world, tabletop_orientation)
        q_grasp = self.solve(q_approach, "grasp", grasp_world, tabletop_orientation)
        # Seed IK at 120 mm, then command one straight 200 mm lift.
        lift_seed_world = grasp_world + np.array([0.0, 0.0, APPROACH_ABOVE_GRASP])
        q_lift_seed = self.ik.solve_q_full(
            q_grasp,
            lift_seed_world,
            tabletop_orientation,
            iterations=PLIER_IK_ITERATIONS,
        )
        q_lift = self.solve(q_lift_seed, "lift", lifted_world, tabletop_orientation)
        q_upright = self.solve(
            q_lift,
            "orient_vertical",
            lifted_world,
            upright_orientation,
        )
        q_raised = self.solve(q_upright, "raise_for_support_clearance", raised_grasp, upright_orientation)
        q_pre = self.solve(q_raised, "align_above_support", pre_grasp, upright_orientation)
        q_loaded = self.solve(q_pre, "move_onto_support", loaded_grasp, upright_orientation)
        q_lowered = self.solve(q_loaded, "lower_onto_support", lowered_grasp, upright_orientation)
        q_retracted = self.solve(q_lowered, "retract", retracted_grasp, upright_orientation)
        q_home = self.home_from(q_retracted)

        waypoints = {
            "approach": (seed, q_approach),
            "descend": (q_approach, q_grasp),
            "close": (q_grasp, q_grasp),
            "lift": (q_grasp, q_lift),
            "orient_vertical": (q_lift, q_upright),
            "raise_for_support_clearance": (q_upright, q_raised),
            "align_above_support": (q_raised, q_pre),
            "move_onto_support": (q_pre, q_loaded),
            "lower_onto_support": (q_loaded, q_lowered),
            "open": (q_lowered, q_lowered),
            "retract": (q_lowered, q_retracted),
            "return_home": (q_retracted, q_home),
            "settle": (q_home, q_home),
        }
        self.start_plan(waypoints, f"planned {self.job.object_name} hang with {self.job.arm_name} arm")

    def phase_message(self, previous: str) -> None:
        super().phase_message(previous)
        if previous == "align_above_support":
            self.message = f"{self.job.object_name} axis aligned; moving onto triangular support"
        elif previous == "open":
            self.message = f"released {self.job.object_name}; retracting {self.job.arm_name} arm"

    def job_done_message(self, completed: PlierHangJob) -> str:
        return f"hung {completed.object_name}; starting {self.job.object_name}"


# One centred combination plier, retaining its measured landmarks.
PLIER_HANG_JOBS_LEVEL1 = (
    PlierHangJob(
        object_name="plier_1",
        arm_name="left",
        support_world_y=LEVEL1_Y,
        grasp_center_local=PLIER_1_GRASP_CENTER_LOCAL,
        hang_axis_local=PLIER_1_HANG_AXIS_LOCAL,
    ),
)


class HangPlierRoutineLevel1(HangPlierRoutine):
    """:class:`HangPlierRoutine` trimmed to pliers_level1.task's one job."""

    JOBS = PLIER_HANG_JOBS_LEVEL1
    LIFT_ABOVE_GRASP = 0.180
    DONE_MESSAGE = "pliers hung; arm returned home"
