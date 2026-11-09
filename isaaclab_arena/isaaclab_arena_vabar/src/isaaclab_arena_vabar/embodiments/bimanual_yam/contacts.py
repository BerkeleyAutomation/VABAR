"""Per-task fingertip contact settings for the bimanual YAM."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from isaaclab_arena_vabar.numeric import is_finite_real

from .config import CRANK_4310, YamHandSpec

ROBOT_PRIM_TOKENS = ("/LeftRobot/", "/RightRobot/")


@dataclass(frozen=True)
class FingertipContact:
    """Contact material for both robots' fingertips, replacing the asset's own."""

    friction: float
    stiffness: float
    damping: float
    gap: float
    margin: float

    def __post_init__(self) -> None:
        for name in ("friction", "stiffness", "damping", "gap", "margin"):
            value = getattr(self, name)
            if not is_finite_real(value) or value < 0.0:
                raise ValueError(f"fingertip {name} must be a finite nonnegative number")


def _apply_fingertip_contact(
    contact: FingertipContact, hand: YamHandSpec, _event_payload=None
) -> None:
    from isaaclab_newton.physics import NewtonManager
    from newton import ShapeFlags

    builder = NewtonManager._builder
    assert builder is not None, "MODEL_INIT fired before the Newton builder existed"
    labels = [str(label) for label in builder.shape_label]
    # Visual-only shapes share their collider's name prefix; only colliders touch.
    matched = [
        index
        for index, label in enumerate(labels)
        if builder.shape_flags[index] & ShapeFlags.COLLIDE_SHAPES
        and any(token in label for token in ROBOT_PRIM_TOKENS)
        and any(token in label for token in hand.fingertip_shape_tokens)
    ]
    robots = {
        label[: label.index(token) + len(token)]
        for label in labels
        for token in ROBOT_PRIM_TOKENS
        if token in label
    }
    expected = hand.fingertip_shapes_per_robot
    assert robots and len(matched) == expected * len(robots), (
        f"expected {expected} {hand.variant} fingertip shapes on each of "
        f"{len(robots)} YAM robots, matched {len(matched)}: "
        f"{[labels[index] for index in matched]}"
    )
    for index in matched:
        builder.shape_material_mu[index] = contact.friction
        builder.shape_material_ke[index] = contact.stiffness
        builder.shape_material_kd[index] = contact.damping
        builder.shape_gap[index] = contact.gap
        builder.shape_margin[index] = contact.margin


def register_fingertip_contact(
    contact: FingertipContact, name: str, hand: YamHandSpec = CRANK_4310
) -> None:
    """Apply ``contact`` to every ``hand`` fingertip when the Newton model is built."""
    from isaaclab.physics import PhysicsEvent, PhysicsManager
    from isaaclab_newton.physics import NewtonManager

    if any(entry[3] == name for entry in PhysicsManager._callbacks.values()):
        return
    NewtonManager.register_callback(
        partial(_apply_fingertip_contact, contact, hand),
        PhysicsEvent.MODEL_INIT,
        name=name,
        wrap_weak_ref=False,
    )
