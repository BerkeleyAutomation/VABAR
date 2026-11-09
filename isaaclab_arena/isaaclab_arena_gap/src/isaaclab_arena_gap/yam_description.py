"""Static description of Arena's YAM arm and the hands its scenes select."""

from __future__ import annotations

from dataclasses import dataclass

YAM_BIMANUAL_ARM_HOME = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0)
"""Home joints of each canonical-USD YAM arm."""


@dataclass(frozen=True)
class YamHand:
    """What the Arena-side observation adapter reads off a YAM hand: its finger joint and its open stroke."""

    open_joint_m: float
    finger_joint_name: str


YAM_CRANK_4310 = YamHand(open_joint_m=0.037524, finger_joint_name="left_finger")
"""The YAM USD's default ``crank_4310`` hand (AUTOLab names it ``yam_linear_arena``)."""

YAM_LINEAR_4310 = YamHand(open_joint_m=0.0475, finger_joint_name="joint7")
"""The YAM USD's ``linear_4310`` gripper variant: both fingers slide 47.5 mm from closed."""

YAM_CRANK_4310_WIDE = YamHand(open_joint_m=0.0475, finger_joint_name=YAM_CRANK_4310.finger_joint_name)
"""AUTOLab's tool-hanging hand: the crank hand with each finger opening to 47.5 mm."""
