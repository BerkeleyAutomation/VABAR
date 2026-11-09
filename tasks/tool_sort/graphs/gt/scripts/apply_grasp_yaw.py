"""Refine the top-down grasp: pick its yaw, and stop the descent above the floor.

``geometry.top_down_grasp_candidates`` is explicit that its primary pose does
not encode the object's yaw -- "the gripper always points straight down along
world -Z; yaw is NOT derived from the OBB" -- because its real consumer is a
goalset planner choosing among a 24-candidate fan. This graph takes ``poses.0``
alone, so both of that fan's jobs land here.

**Yaw, by attempt.** Under the DLS backend the commanded yaw was ignored
(wrist roll is left free) and the seeded chain happened to land
``home_yaw + handle_yaw`` -- verified to 0.0 degrees against a recorded
successful grasp. A full-orientation backend obeys the command, so the yaw has
to be chosen. Measured per tool, *neither* single choice wins: the hammer holds
at the fingers-across-the-handle yaw and not at yaw-zero, the syringe held at
yaw-zero and slipped at cross-handle. So the two passes the task allows
(``MAX_ATTEMPTS``) try one each -- a two-candidate fan sized to the retry
budget: attempt 1 commands the cross-handle composition (``sim.get_object_obb``'s
``grasp_quat``, built about the embodiment's *declared* home so a mid-episode
pass is not yawed by whatever the previous goal left on the wrist), attempt 2
falls back to the candidate's own unyawed rotation.

**Depth is not decided here any more.** It was: this node used to clamp the
candidate's z up to ``FLOOR_CLEARANCE`` above the object's measured underside,
which stopped a 13.6 mm syringe being grasped through the bench. That guard was
necessary and it was not sufficient, and a near-vertical hammer is what showed
the difference -- see :file:`clamp_grasp_depth.py`, which now owns the whole
depth rule and runs immediately after this node. It re-derives z from the box
alone, so whatever this node passes through is discarded; keeping a second rule
here would have been a dead one.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Se3Pose


class Output(TypedDict):
    grasp_pose: Se3Pose
    yaw_choice: str


def run(ctx: NodeContext, pose: Se3Pose, object_name: str = "") -> Output:
    goal = ctx.tool("sim.next_goal")
    if not object_name:
        object_name = str(goal["object_name"])
    obb = ctx.tool("sim.get_object_obb", object_name=object_name)

    # Where the heading comes from is **not** what the UR5's failure is about,
    # and that was measured rather than assumed. Deriving it here instead --
    # narrowest horizontal box axis, then ``robot.grasp_frame`` to put this
    # hand's measured axes on it, which is what ``astro``'s copy of this node
    # does -- leaves the Panda unchanged (706, 707 against 704, 704) and leaves
    # the UR5 exactly as broken (0/2 at 1507, 1504 against 0/5 at 1504-1515).
    # Two independent derivations of the same heading fail the same way, so the
    # heading is not the thing. Not kept: an unmeasured refactor of a node whose
    # output the native arm discards is a change nobody can falsify.
    cross = obb.get("grasp_quat")
    use_cross = cross is not None and int(goal.get("attempt", 1)) % 2 == 1
    rotation = dict(cross) if use_cross else dict(pose["rotation"])

    return {
        "grasp_pose": {"position": dict(pose["position"]), "rotation": rotation},
        "yaw_choice": "cross_handle" if use_cross else "candidate",
    }
