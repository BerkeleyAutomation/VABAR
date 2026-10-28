# SPDX-License-Identifier: Apache-2.0
"""One spur gear: a free rigid body built from its own convex parts.

**An ``ObjectSpec`` subclass, not an extras entity, and that is load-bearing.**
Only the ``objects`` bin gets a full ``EntityHandle``, so only this way can a
goal say ``(gripped gear_a.body)`` or ``(encloses gear_a.bore board.peg_a)`` at
all -- ``box_decant``'s knife makes the same choice for the same reason. What a
gear cannot do from that bin is *publish*: ``_published`` resolves out of
``snapshot.extras`` alone, so every state a goal reads about a gear is published
by the board it turns on, the way the seam field publishes the carton's.

**``build`` does not call ``super()``.** ``EntityRigid._add_primitive`` wants a
``ShapeSpec``, and a gear has no primitive to be: it is a ring of authored
convex prisms, thirty-four to fifty-six of them, and a seed primitive would be a
phantom collider inside the hub with nothing to do but be hit. So the four lines
that make a free body -- ``add_body``, which creates the free joint *and* the
articulation SolverMuJoCo needs in order to integrate it at all, plus recording
the joint's coordinates so a reset does not return early and leave the gear where
the last episode left it -- are restated here, and the parts go on directly.
``EntityRigid.bind`` refills ``self.shapes`` from the body, so segmentation, the
grasp box and the contact probe all still see them.

**Mass is declared once and applied last.** ``set_body_mass`` rescales and
preserves the distribution, so stating it after every part exists is what makes
the declared mass the last word *and* the inertia the ring's rather than one
tooth's -- ``carton.py``'s move, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render.palette import METAL
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.build.bodies import set_body_mass
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs import ObjectSpec

from .gearing import BORE_RADIUS, FACE, gear_frame, gear_parts, gear_volume

__all__ = ["GEAR_DENSITY", "TIP_CHAMFER", "EntityGear", "SpecGear"]

GEAR_DENSITY = 1400.0
"""Density the gear's mass is derived from [kg/m^3], if none is declared.

Acetal, which is what a machine-shop spur gear this size is actually cut from,
and the reason the mass is *derived* rather than stated: the catalogue spans
tooth counts from 14 to 28, and one stated mass across that range would make the
small gear four times too heavy or the large one four times too light. The
vendored ``spur_gear_20t`` states 2.516 kg, which is the same gear at module 5 in
steel -- a useful check that the arithmetic here is the right shape, and a
reminder of why it is not the asset this suite ships."""

TIP_CHAMFER = 6.0e-4
"""Bevel on both end faces of every tooth [m].

**The lead-in, and it is mechanism rather than detail.** A gear is lowered onto
a peg beside a neighbour that is already seated, and its teeth have to find that
neighbour's spaces. Free on its peg, it finds them by turning -- but only if what
meets first is a slope. Two square tip corners meeting flat is a jam, and a jam
here reads as a policy that cannot aim. 0.6 mm on a 10 mm face costs nothing and
is what a real gear's tip relief is for."""


@dataclass(frozen=True, kw_only=True)
class SpecGear(ObjectSpec):
    """A spur gear, stated by its cutting parameters rather than its shape.

    Poses are in the work-surface frame. Every radius follows from ``module`` and
    ``teeth``; nothing here states one, because the moment a radius is written
    down twice a board stops meshing. See ``gearing.gear_frame``.

    Attributes:
        module: Tooth size [m]. **Two gears mesh only if this matches**, so it is
            a property of the catalogue rather than of one gear.
        teeth: Tooth count. Also the gear's identity in a chain: the ratio it
            turns at is the pinion's count over this one.
        face: Thickness along the axis [m].

            **16 mm, and it is a grasp measurement rather than a proportion.**
            Face width does not enter the meshing at all -- module and tooth
            count fix every radius -- so what sets it is the one other thing it
            decides: how much of the pads a gear lying flat presents. At 10 mm
            the pads engaged 7.5 mm of rim, touching a couple of tooth tips, and
            a carried gear was dropped on roughly one traverse in three. It is
            not a friction budget -- Coulomb friction does not care about contact
            area, and 40 N of grip against a 0.3 N gear has a factor of a hundred
            in hand -- it is *manifold stability*: more rim under the pad is more
            tooth tips in contact, and a contact set that does not drop out.
        bore_radius: Drawn bore [m]. The peg is sized against
            ``frame["bore_effective"]``, which is smaller.
        backlash: Slack at the pitch circle [m]; 0 takes the module's default.
        tip_chamfer: Lead-in bevel [m]; see :data:`TIP_CHAMFER`.
    """

    stage: ClassVar[str] = "objects"
    entity: str = "gear_wheel"
    category: str = "spur_gear"
    module: float = 0.0025
    teeth: int = 20
    face: float = FACE
    bore_radius: float = BORE_RADIUS
    backlash: float = 0.0
    tip_chamfer: float = TIP_CHAMFER
    density: float = GEAR_DENSITY
    color: Any = METAL
    # A gear is a tight fit on its peg and a tight mesh with its neighbour, so
    # its contact gap has to be at the scale of those fits rather than at the
    # arrangement's. It is bounded from above by the backlash -- a gap fattens
    # every collider on every face, and two of them eat the slack the mesh turns
    # in. See BACKLASH_FACTOR.
    contact_gap: float | None = 5.0e-5
    # tool_sort's measured ceiling, as polish and box_decant both use: Newton's
    # bare default is ~400x too soft for a part that has to carry a drive.
    ke: float | None = 6.0e4
    kd: float | None = 500.0
    friction: float | None = 1.4
    """Friction on the TEETH -- what the pads grip and what the mesh drives on.

    High for the grasp, and as high as the mesh will take. Two pads on a hard
    round rim can only touch it at the tooth tips, and at 1.0 a carried gear
    slipped 6 mm down the jaws between the pick and the peg. Going up costs the
    drive nothing until it suddenly does: with the bore split out, the assembled
    three-gear board runs at 0.972 of its ratio at mu 1.0, 0.970 at 1.2, 0.968 at
    1.4 and 0.925 at 1.7. 1.4 is the last value that is free. The vendored
    ``spur_gear_20t`` MJCF states 2.0, which is where this started and is a
    number about a gear being *sorted* rather than one being driven."""
    bore_friction: float = 0.15
    """Friction on the HUB -- the bore, which is a journal bearing.

    **A gear's rim and its bore are different surfaces, and giving them one
    number stops the train.** The mesh pushes a gear radially against its peg, so
    the bore carries the whole drive load; at the rim's own 2.0 that contact is a
    brake, and the measurement is not subtle -- the assembled three-gear board
    ran at 0.098 of its ratio with every gear still reading as seated, against
    0.968 at 1.0. It took a bisection to find because it looks exactly like teeth
    that will not transmit, and the teeth were transmitting perfectly: all three
    stations agreed to three decimals the whole time, which is a chain being held,
    not a chain slipping.

    0.15 is a dry steel journal, and it is what the peg already declares from its
    own side (``SpecGearBoard.peg_mu``). Sweeping the peg's alone found nothing,
    because MuJoCo combines the two and the gear's was the larger."""
    _frame: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        frame = gear_frame(
            module=self.module,
            teeth=self.teeth,
            face=self.face,
            bore_radius=self.bore_radius,
            backlash=self.backlash,
        )
        object.__setattr__(self, "_frame", frame)
        if self.mass is None:
            object.__setattr__(self, "mass", GEAR_DENSITY * gear_volume(frame))
        super().__post_init__()

    @property
    def frame(self) -> dict[str, Any]:
        """Every radius and profile this gear is built from. Derived once, in
        ``__post_init__``, so a spec and the body built from it cannot disagree."""
        return self._frame


@register_entity("gear_wheel")
class EntityGear(EntityRigid):
    """The gear's body: ``teeth`` tooth prisms and ``hub_sectors`` hub prisms."""

    spec: SpecGear

    def _placed(self, pose: np.ndarray) -> np.ndarray:
        """Always compose the work-surface frame. A gear is a *fifth* case.

        ``EntityRigid._placed`` reads "no asset and no shape" as "declared by the
        scene's own import, already where that import put it" -- and its own
        docstring records the fourth case that broke that test, a procedural cube
        left at the surface origin, inside the table, which it then fell through.
        A gear is one step further along: it has no asset *and* no ``ShapeSpec``,
        because it draws its own parts, and it draws them in the surface frame
        like any other task-supplied object. Measured before it was written: the
        gears spawned at the frame's origin and the board 0.6 m away, and the
        train read as three gears that were never on it.

        Stating it here rather than giving the spec a ``ShapeSpec`` it never
        draws: a shape on a spec that reaches no builder is a second answer to
        "what is this body" and the wrong one.
        """
        return pose if self.spec.world_xform else self.placed(pose)

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        spec = self.spec
        pose = self._placed(self._episode_pose(episode))
        xform = wp.transform(tuple(float(v) for v in pose[:3]), wp.quat(*[float(v) for v in pose[3:7]]))
        body = builder.add_body(xform=xform, label=spec.name)
        joint = builder.joint_count - 1
        self.coord_start = int(builder.joint_q_start[joint])
        self.dof_start = int(builder.joint_qd_start[joint])
        self.body = body

        cfg = self._contact_cfg(builder)
        # Every part carries the density, so the body's inertia distribution is
        # the RING's -- hollow, teeth at the rim -- rather than the solid disc a
        # single seed shape would have given it. set_body_mass then rescales that
        # distribution to the declared mass, which is carton.py's move and the
        # reason the mass is stated last rather than first.
        cfg.density = float(spec.density)
        # The hub runs on a shaft and the teeth are gripped and driven, so they
        # get different friction. Nothing but the peg ever touches the hub, so
        # this is exactly the bore's own contact and nothing else's.
        bore = cfg.copy()
        bore.mu = float(spec.bore_friction)
        first_shape = builder.shape_count
        for label, verts, indices in gear_parts(spec.frame, float(spec.tip_chamfer)):
            builder.add_shape_mesh(
                body,
                mesh=newton.Mesh(verts, indices),
                cfg=bore if label.startswith("hub") else cfg,
                color=spec.color,
                label=f"{spec.name}_{label}",
            )
        # What the parts weigh before the mass is restated. gear_volume sums the
        # same polygons this loop extruded, so the two are the same arithmetic by
        # two routes -- Newton's mesh inertia and a shoelace -- and a drift
        # between them means a part is being built that gear_volume does not
        # know about. Loose enough for the root seam and the tip chamfer, which
        # gear_volume deliberately does not model.
        built = float(builder.body_mass[body])
        if abs(built - spec.mass) > 0.05 * spec.mass:
            raise ValueError(
                f"gear {spec.name!r}: the parts weigh {built * 1000:.2f} g but gear_volume says "
                f"{spec.mass * 1000:.2f} g at density {spec.density:.0f}; the profile and the volume "
                "have come apart"
            )
        set_body_mass(builder, body, float(spec.mass))
        self._part_mu = {s: float(builder.shape_material_mu[s]) for s in range(first_shape, builder.shape_count)}

    def after_build(self, builder: newton.ModelBuilder, episode) -> None:
        """The rigid entity's bookkeeping, then the per-part friction put back.

        A gear has no asset and no ``shape`` -- it draws its own parts -- so
        RoboSimStudio's ``EntityRigid.after_build`` treats it as an object the
        scene declared and locates it, and locating stamps ``spec.friction`` on
        every shape of the body. That erases the bore's own 0.15: measured on
        ``rigs/gear_bench.py spin``, a hub at the teeth's 1.4 brakes on its peg
        and the whole train slips to 0.75 of its ratio, where the parts as built
        here run it at 0.968. The teeth already carry ``spec.friction``; only the
        hub differs, and only this entity knows which part is which.
        """
        super().after_build(builder, episode)
        for shape, mu in getattr(self, "_part_mu", {}).items():
            builder.shape_material_mu[shape] = mu
