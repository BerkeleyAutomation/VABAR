"""Choose how deep the fingertips go, from the target box alone.

``geometry.top_down_grasp_candidates`` places its primary pose a fixed
``z_offset`` below the box's *top* face. That is the right shape of rule for a
short object and the wrong one for a tall one: at the default -0.04 a 0.084 m can
is gripped across its middle, and a 0.190 m milk carton is gripped by the 4 cm of
gable at its very top, where the faces slope away from the pads.

So the depth is stated as a fraction of the object's own height instead, measured
up from its underside, and then clamped at both ends.

**The fraction is 0.35 rather than the middle, and the mustard bottle is why.** This
suite's own object notes call it out: it is the narrowest item at 0.049 m and the one
whose shape is least like its bounding box, so a grasp at the centroid closes on the
tapering shoulder instead of on the body. Measured on a perception run, that is not a
weak grip but a violent one -- the pads met the taper during the descent, walked the
bottle 11 mm sideways before they closed, and then ejected it: from (0.274, 0.165) to
(1.497, 1.634) in twenty control steps, 1.6 m off the table, and the episode finished
at 0.80 with that item unreachable. Lower is on the parallel-sided body of every one
of the five (a can at 0.35 is still 0.025 m up its cylinder, well clear of the floor
rule below), and it makes the transport steadier too, since the item hangs less far
below the fingertips.

The clamps:

* not below ``FLOOR_CLEARANCE`` above the underside, because for a resting object
  the underside is the table -- and fingertips driven below it stop on the table
  with the pads still above the object, which closes the hand on air. That failure
  is silent: the gripper reports closed, the graph transports nothing, and the
  episode reads as a clean run scoring zero;
* not above ``TOP_INSET`` below the top face, so a pad's own height still has
  object beside it.

Every number comes from the box that was handed in, so this node is identical in
the ground-truth graph and the perception graph -- which is the property that makes
the two comparable. It reads no ``sim.*`` tool.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose

FLOOR_CLEARANCE = 0.006
"""How far above the object's underside the fingertips may stop [m]."""

HOUSING_RISE_FALLBACK = 0.035
"""How far above the fingertips the object's top may sit [m], when unmeasured.

Asked for first: ``robot.describe_gripper().housing`` carries ``rise_m`` and a
``stated`` flag, and how far the hand's body stands above its own fingertips is
a fact about the hand. This is what is used when the hand that arrived has not
been measured, because a graph that refused to grasp on an unmeasured hand would
refuse more than it should.

**This asked ``finger.reach_m`` when it arrived from ``grocery_packing``, and on
another hand that is a different number entirely.** ``tip_reach`` is the
distance from the *tool centre* to the fingertips, which on the DROID's hand is
about 35 mm -- the TCP sits between the jaws -- and stood in for jaw depth
convincingly. The Robotiq 2F-85 reports its TCP at the fingertip plane and
``tip_reach`` 4.4 mm, so the same expression pinned every grasp to
``TOP_INSET`` below the object's top face: a 149 mm ketchup bottle gripped 12 mm
below its cap, a pair of pliers by the taper at the end of its handle. Three of
four grasps failed that way on ``astro/wiring_sort`` before this line was the
housing's.

**The hand is not hollow above its pads**, which is what makes this a hard limit
rather than a preference. Everything above the fingertips up to the housing is
linkage; the housing itself is solid. An object whose top stands higher above the
fingertips than the housing's underside does is an object the hand is trying to
occupy the same space as.

The rule only binds on something tall. A 31 mm pair of pliers is gripped at the
fraction, 11 mm up, because its top is nowhere near the housing; a 149 mm bottle
is moved from the fraction's 52 mm up to 89 mm up, which on this hand is the
lowest band of it the housing clears.
"""

TOP_INSET = 0.012
"""How far below the object's top face the fingertips must stay [m].

Roughly a Robotiq pad's own height, so the grip has object either side of it.
"""


class Output(TypedDict):
    grasp_pose: Se3Pose
    fingertip_z: float
    grip_height: float


def run(
    ctx: NodeContext,
    pose: Se3Pose,
    target_obb: OrientedBoundingBox,
    grip_height_frac: float = 0.35,
) -> Output:
    # One tool call, and an unprivileged one: how far the hand's own body stands
    # above its fingertips is proprioception. Everything else here is box geometry.
    centre_z = float(target_obb["center"]["z"])
    half_z = float(target_obb["extent"]["z"])
    bottom, top = centre_z - half_z, centre_z + half_z

    # The fraction, then high enough that the object's top clears the housing,
    # then the two clamps.
    wanted = bottom + float(grip_height_frac) * (top - bottom)
    housing = ctx.tool("robot.describe_gripper")["housing"]
    rise = float(housing["rise_m"]) if bool(housing["stated"]) and float(housing["rise_m"]) else HOUSING_RISE_FALLBACK
    wanted = max(wanted, top - rise)
    low = bottom + FLOOR_CLEARANCE
    high = top - TOP_INSET
    # A box shorter than FLOOR_CLEARANCE + TOP_INSET has no valid band at all;
    # taking the low end keeps the fingers off the table, which is the failure
    # that matters.
    fingertip_z = max(low, min(wanted, high)) if high > low else low

    position = dict(pose["position"])
    position["z"] = fingertip_z
    return {
        "grasp_pose": {"position": position, "rotation": dict(pose["rotation"])},
        "fingertip_z": fingertip_z,
        # How far the object hangs below the fingertips once gripped here, which is
        # what the transport node needs to release it *into* the tote rather than
        # push it through the floor.
        "grip_height": fingertip_z - bottom,
    }
