"""Lift + lateral move above the drop XY — the 2-step waypoint chain.

Two straight-line Cartesian legs: lift vertically at the current XY to
``safe_height``, then translate laterally to the drop XY at constant
height. The rectangular path guarantees the held object never dips below
``safe_height`` between pick and place.

A speed-motivated variant replaced the 2-step chain with a single
free-space ``curobo.plan_to_pose``. Without a collision world (transport
carries the grasped object, which would need an attachment model),
TrajOpt's shortest smooth path routinely dips LOW while translating —
recorded trial videos show the held object dragged through scene clutter,
shoving the container ~12 cm off its perceived pose, with deterministic
per-layout failures. The convergence cost that motivated the single-plan
variant does not apply here: ``robot.go_to_pose_cartesian`` executes a
pre-planned linear trajectory via fast waypoint playback, not per-step PD
convergence. A collision-aware planned variant remains available as the
SEPARATE node ``waypoint_move_carve`` (rebuilt world +
``curobo.plan_with_grasped_object``).
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.errors import PlanningFailed
from gap_core.types import Quaternion, Se3Pose


def _work_arm_id(ctx: NodeContext) -> int:
    """Which entry of ``get_observation()["arms"]`` is the arm doing the task.

    Index 0 happens to be right today, because the facade emits arms in *role*
    order -- work first -- but nothing in a graph says so, and a graph that reads
    index 0 on a robot whose roles are assigned the other way drives the arm that
    is holding something still. ``robot.describe_arm`` reports the id, so the
    convention is asked for rather than assumed.
    """
    return int(ctx.tool("robot.describe_arm")["arm_id"])


class Output(TypedDict):
    done: bool
    #: The height the lateral leg actually flew at, and how many rungs down the
    #: ladder it took. ``descents == 0`` is the cruise height it was asked for.
    flown_z: float
    descents: int


#: How far each retry drops the lateral leg [m], and how many rungs there are.
#:
#: **The failure this exists for, counted.** Seven lateral aborts across six
#: banked graph cells go to the *identical* world point ``(0.198, 0.403)`` at
#: ``z = 1.100`` with residuals of 41, 41, 44, 46, 52, 52 and 54 mm -- the servo
#: closing to within a few centimetres and stopping, which is a reach limit and
#: not a plan that failed to exist. Every one of them lost a goal with the tool
#: already in the hand, which the failure branch then opened over the bench.
#:
#: A reach limit is a function of height, so the recovery is to try the same
#: point lower rather than to abandon the goal. 30 mm clears the largest residual
#: above in one rung, and three rungs reach 1.010 -- which is where the arm's own
#: resting tool height is, and as low as this leg has any business going.
#:
#: **The cruise height itself is not this ladder's business, and trying to make
#: it so cost a cell that was at ceiling.** ``describe_workspace`` derives 1.011
#: here, from the arm's resting tool height floored by finger clearance over the
#: work *surface* -- and flying the whole lateral leg there scored ``hammer``
#: 0.00 against the 1.100 literal's 1.00, twice, because a bin rim is not the
#: work surface and a tool hanging below the fingertips clips it on the way in.
#: The derived number answers "where can this arm reach", the literal answers
#: "what clears the containers", and only the second one keeps the held object.
#: So the literal stays where the workflow states it, and the ladder is what
#: turns a reach failure from a lost goal into a lower pass.
DESCENT_STEP_M = 0.030
DESCENT_RUNGS = 3


# This file used to open with two constants: a top-down quaternion, and a lift
# altitude of 0.353 justified in Franka link semantics ("the configured TCP offset
# ≈0.097 m on Franka, so 0.353 puts the panda_hand link at z≈0.45"). Both are
# facts about one robot. ``robot.grasp_frame`` composes top-down from the live
# hand's measured approach and closing axes, and ``robot.describe_workspace``
# derives the lift from the arm's own resting tool height floored by what its
# fingers need to clear the surface -- which lands on 0.353 for this Franka and on
# whatever is right for anything else. The invariant downstream nodes assume is
# still "the hand is above everything on the bench", which is what the derivation
# states directly and the literal only implied.


def run(
    ctx: NodeContext,
    drop_x: float,
    drop_y: float,
    drop_rotation: Quaternion | None = None,
    safe_height: float = 0.0,
) -> Output:
    obs = ctx.tool("robot.get_observation")
    ee = obs["arms"][_work_arm_id(ctx)]["ee_pose"]["position"]

    workspace = ctx.tool("robot.describe_workspace")
    if safe_height <= 0.0:
        safe_height = float(workspace["transport_z"])
    # Never below the height a hover needs to hold the whole hand clear of the
    # work surface: the ladder below is allowed to trade reach for altitude, and
    # not to trade away the clearance the held object is being carried at.
    floor_z = float(workspace["surface_z"]) + float(workspace["align_clearance_m"])

    # Use the upstream-supplied drop rotation when available so the
    # lift+lateral phase doesn't unspool grasp-time yaw — that unspool
    # manifests as a redundant-joint reconfiguration ("circular elbow
    # motion") that can fling the held object.
    rotation = drop_rotation if drop_rotation is not None else ctx.tool("robot.grasp_frame")["rotation"]

    # Leg 1: straight vertical lift at the current XY.
    lift_pose: Se3Pose = {
        "position": {"x": float(ee["x"]), "y": float(ee["y"]), "z": float(safe_height)},
        "rotation": rotation,
    }
    ctx.tool("robot.go_to_pose_cartesian", pose=lift_pose)

    # Leg 2: lateral translate to the drop XY at constant height, dropping a
    # rung and trying again when the far corner is out of reach up here. The
    # lift already happened, so a retry is one Cartesian leg and not a re-plan.
    failures: list[str] = []
    for rung in range(DESCENT_RUNGS + 1):
        z = safe_height - rung * DESCENT_STEP_M
        if rung and z < floor_z:
            failures.append(f"rung {rung} at z={z:.3f} would break the {floor_z:.3f} clearance floor")
            break
        lateral_pose: Se3Pose = {
            "position": {"x": float(drop_x), "y": float(drop_y), "z": float(z)},
            "rotation": rotation,
        }
        try:
            ctx.tool("robot.go_to_pose_cartesian", pose=lateral_pose)
        except Exception as exc:  # linear plan can fail near joint limits
            failures.append(f"z={z:.3f}: {exc}")
            continue
        return {"done": True, "flown_z": float(z), "descents": rung}

    raise PlanningFailed(
        f"waypoint_move: lateral cartesian leg to ({drop_x:.3f}, {drop_y:.3f}) failed at every "
        f"height from {safe_height:.3f} down: " + "; ".join(failures)
    )
