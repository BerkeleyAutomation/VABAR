# SPDX-License-Identifier: Apache-2.0
"""The fittings this suite draws: a run's two ends, and the industrial cleat.

Three ``EntityStatic`` subclasses, and the reason they are here rather than in
:mod:`robosimstudio.domains.rod` is the seam that module's own docstring draws:
the *mechanism* -- a capsule chain, its latch, the imported F1 spool -- is the
library's because a second task needed it, and the *props* stay with the task
that stages them. A saddle bracket, a socket and a bolted cleat are props. No
other suite has asked for one, and the day one does the move is the same one the
rod made.

**These were one plain box each, and the box was honest about the physics and
dishonest about everything else.** A 32 x 32 x 10 mm slab is what a cable
terminates at only in the sense that a brick is what a door closes against: it
asserts a position and a face and says nothing about how a cable is *held*
there. So the welded end read as a rod lying next to a block -- which is what it
looked like because it is what the geometry said -- and the port read as a kerb
rather than as somewhere an end goes in.

What replaces it is the hardware the two ends are: a saddle bracket the run is
clamped under, and an open-topped socket the free end is dropped into. Both keep
the declared 32 x 32 mm footprint, so every number measured against the plate
stays the number it was -- :func:`~.layout.weave_terminals`' standoffs, the
port's scored region, the seat boxes beside it.

**Nothing in either assembly sits under the cable**, and that is a constraint
rather than a style. The rod rests on the bench at ``surface_z + radius``, so a
5 mm mounting plate laid across its path puts the first two capsules 5 mm inside
a static box -- at the anchor, where one of them is WELDED and cannot answer.
The fitting therefore straddles: two side rails either side of the channel, and
everything else bridged over the top of it. That is also what a real saddle
looks like, which is the usual way round -- the geometry that is right for the
contact set is the geometry the part actually has.

All three take their contact settings from
:func:`~robosimstudio.domains.rod.fixture_shape_cfg`, which is the spools' own.
A cable bearing against a fitting and a cable bearing against a spool have to
answer the same way; two fixtures on one board tuned differently is a difference
in the result that is not a difference in the task.
"""

from __future__ import annotations

import newton
import numpy as np
import warp as wp
from robosimstudio_materials.rod import fixture_shape_cfg

from robosimstudio.core.math import quat_to_mat
from robosimstudio.core.render.finish import finish_for
from robosimstudio.domains.static.entity import EntityStatic, episode_xform

from .fixtures import CLEAT_POST_RADIUS

__all__ = [
    "CLEAT_POST_H",
    "TERMINAL_HALF",
    "EntityCableCleat",
    "EntityCablePort",
    "EntityCableTerminal",
]

TERMINAL_HALF = 0.016
"""Half the terminal's footprint [m]: the declared 32 x 32 mm plate.

The asset record's own, and not free to move -- :func:`~.layout.weave_terminals`
places the anchor by it and the port's scored region is measured off it.
"""

TERMINAL_CHANNEL = 0.005
"""Half-width of the channel the cable lies in [m].

10 mm across a 7 mm cable: enough that a carry arriving a couple of millimetres
off still lands in it, tight enough that the cable is visibly *captured* rather
than resting near something.
"""

TERMINAL_RAIL_H = 0.0025
"""Half-height of the anchor's side rails [m] -- a 5 mm foot.

Thinner than the 10 mm slab this replaces, because the rails are a foot with a
fitting standing on them rather than the whole terminal.
"""

TERMINAL_RISE = 0.0125
"""How far a fitting stands above the bench [m].

A little under twice the cable's diameter, which is what a real gland or socket
stands: tall enough to read as a fitting from the bench camera at 450 mm, short
enough that it is not something the route must be lifted over. The F1 spools'
flanges reach 23.5 mm, so nothing here is the tallest thing on the board.
"""

TERMINAL_CLEAR = 0.0080
"""Underside of everything bridged over the channel [m].

The cable's crown sits at ``2 * radius`` = 7 mm on this suite's rod. 8 mm is the
first millimetre above it, and the margin is deliberately thin: a hood parked
high enough to be unmistakably clear is a hood that no longer reads as holding
anything.
"""

TERMINAL_SCREW_R = 0.0025
TERMINAL_SCREW_H = 0.0009
"""The boss on each rail [m]: visual only. See :meth:`EntityCableTerminal._rails`."""

TERMINAL_METAL = (0.38, 0.40, 0.43)
TERMINAL_BRIGHT = (0.72, 0.74, 0.77)
"""The rails and the screws [RGB in 0..1].

The FITTING keeps the spec's own colour and the foot does not, which is a
perception decision as much as a visual one: ``plug_port`` grounds the terminals
with ``"orange block"`` against a 4000 px area floor, and a terminal painted
entirely metal would stop being findable by the prompt that finds it. Colouring
the bridged parts -- the tallest and most face-on of the assembly -- keeps the
mask where the detector already reads it, and puts grey where a mounting plate's
grey belongs.
"""


class EntityCableTerminal(EntityStatic):
    """The anchor bracket: two rails, a hood over the end, and a saddle clamp.

    The rod is welded to the world here (``RodSpec.anchors``), and this is what
    makes that legible. The cable runs in between the rails, passes under a
    saddle, and disappears under the hood -- which is how a fixed end is fixed on
    real hardware and, not coincidentally, what the weld already does in physics.

    **Local +x is the direction the run leaves by.** The assembly is drawn about
    that axis and the spec's yaw aims it, so one class serves an anchor at either
    end of a bench and a board an episode has turned. Nothing here reads a world
    coordinate.
    """

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        # The base class resolves the episode's pose at the top of its own
        # ``build``; this one overrides ``build`` and must do it too, or a
        # sampler that places the terminals moves their scored regions and
        # leaves the fittings behind. The same trap ``EntityPeg`` records.
        self.xform_now = episode_xform(self.spec, episode)
        del episode
        self._parts: dict[int, tuple[float, float, float]] = {}
        placed = self.placed(self.xform_now)
        self._draw(
            builder,
            np.asarray(placed[:3], dtype=np.float64),
            np.asarray(placed[3:7], dtype=np.float64),
        )

    # ------------------------------------------------------------------ parts
    def _box(self, builder, origin, quat, *, centre, half, collides=True, shade=None, tag=""):
        """One box of the assembly, stated in the terminal's own frame."""
        world = origin + quat_to_mat(quat) @ np.asarray(centre, dtype=np.float64)
        shape = builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*[float(v) for v in world]), wp.quat(*[float(v) for v in quat])),
            hx=float(half[0]),
            hy=float(half[1]),
            hz=float(half[2]),
            cfg=fixture_shape_cfg(builder, collides=collides),
            label=f"{self.spec.name}_{tag}" if tag else self.spec.name,
        )
        if shade is not None:
            self._parts[int(shape)] = tuple(float(v) for v in shade)
        return shape

    def _rails(self, builder, origin, quat, *, half_x: float, half_z: float, shade=TERMINAL_METAL) -> None:
        """The two side rails: the foot, and the only parts that touch the bench.

        Either side of the channel and never across it -- see the module note.
        Drawn as one pair for both terminals because the constraint that shapes
        them is the same at both ends.

        ``shade`` is what differs, and it is a perception constraint rather than
        a taste. On the anchor the rails are a FOOT with an orange fitting
        standing on it, so grey is both right and cheap -- the hood carries the
        mask. On the port the rails ARE the fitting: there is nothing above them,
        so painting them grey leaves ``plug_port``'s ``"orange block"`` prompt a
        3 mm back wall to find against a 4000 px floor. Pass ``None`` there and
        they keep the spec's own colour.
        """
        half_y = 0.5 * (TERMINAL_HALF - TERMINAL_CHANNEL)
        for sign in (-1.0, +1.0):
            self._box(
                builder,
                origin,
                quat,
                centre=(0.0, sign * (TERMINAL_CHANNEL + half_y), half_z),
                half=(half_x, half_y, half_z),
                shade=shade,
                tag="rail",
            )
        rot = quat_to_mat(quat)
        for sign in (-1.0, +1.0):
            centre = (-0.5 * half_x, sign * (TERMINAL_CHANNEL + half_y), 2.0 * half_z + TERMINAL_SCREW_H)
            world = origin + rot @ np.asarray(centre, dtype=np.float64)
            shape = builder.add_shape_cylinder(
                body=-1,
                xform=wp.transform(wp.vec3(*[float(v) for v in world]), wp.quat(*[float(v) for v in quat])),
                radius=TERMINAL_SCREW_R,
                half_height=TERMINAL_SCREW_H,
                # Visual. A 2.5 mm boss proud of a rail is a thing to look at,
                # not a thing to plan around, and admitting it to the contact
                # set only gives a fingertip somewhere new to catch.
                cfg=fixture_shape_cfg(builder, collides=False),
                label=f"{self.spec.name}_screw",
            )
            self._parts[int(shape)] = TERMINAL_BRIGHT

    def _draw(self, builder, origin, quat) -> None:
        self._rails(builder, origin, quat, half_x=TERMINAL_HALF, half_z=TERMINAL_RAIL_H)
        # THE HOOD, bridging the rails at the back. The run passes under it and
        # stops there, so the end is inside a fitting rather than beside a box --
        # which is the whole complaint the slab earned.
        hood_h = 0.5 * (TERMINAL_RISE - TERMINAL_CLEAR)
        self._box(
            builder,
            origin,
            quat,
            centre=(-0.5 * TERMINAL_HALF, 0.0, TERMINAL_CLEAR + hood_h),
            half=(0.5 * TERMINAL_HALF, TERMINAL_HALF, hood_h),
            tag="hood",
        )
        # THE SADDLE, a strap across the channel at the mouth. Same underside as
        # the hood, so the cable runs level under both.
        self._box(
            builder,
            origin,
            quat,
            centre=(0.5 * TERMINAL_HALF, 0.0, TERMINAL_CLEAR + hood_h),
            half=(0.20 * TERMINAL_HALF, TERMINAL_HALF, hood_h),
            tag="saddle",
        )

    # ---------------------------------------------------------------- display
    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """The spec's colour to the fitting, and metal to the rails and screws.

        The base implementation paints every shape in the entity's range one
        colour, which is right for a static that IS one shape and wrong for an
        assembly: a mounting plate and the connector standing on it are not the
        same colour on any bench, and painting them alike is most of what made a
        slab read as a slab. Overridden rather than generalised, because "some of
        my shapes are furniture" is a fact about this fixture and not about
        scenery.

        **The finish moves with the colour**, which the override it was ported
        from did not do. :meth:`~robosimstudio.scene.entity.Entity.apply_display`
        fills ``self.finishes`` beside ``shape_color``, and the path-traced
        backend binds a material per instance out of that table rather than
        re-reading the colour -- so writing ``shape_color`` alone left the rails
        grey in the GL viewer and orange plastic under RTX. Re-deriving the
        finish from the colour the shape ends up with is what keeps the two
        renderers showing one bench.
        """
        super().apply_display(builder)
        if not getattr(self.spec, "visible", True):
            return
        role = self.display_role
        for shape, rgb in self._parts.items():
            builder.shape_color[shape] = rgb
            self.finishes[shape] = finish_for(rgb, role)


class EntityCablePort(EntityCableTerminal):
    """The socket the run terminates in: a channel with a stop, open on top.

    The difference from the anchor is the difference between a place a cable is
    *held* and a place a cable is *put*. A saddle would have to be threaded; this
    is open above, because that is how ``plug_port`` actually arrives -- it
    stages over the mouth at carry height and descends, full-pose IK over this
    port having refused a vertical wrist 120 mm up and accepted one at rest
    height.

    So the rails become walls: the same two pieces, tall enough to retain a rod
    dropped between them. The back wall is the part that does work -- it is what
    an end arriving under tension stops against, and without it the carry's
    overshoot is the difference between a plugged run and one pushed out the far
    side.
    """

    def _draw(self, builder, origin, quat) -> None:
        wall_h = 0.5 * TERMINAL_RISE
        # No shade: on a port the walls are the fitting. See :meth:`_rails`.
        self._rails(builder, origin, quat, half_x=TERMINAL_HALF, half_z=wall_h, shade=None)
        # The stop, at the far end of the channel from the run.
        back = 0.20 * TERMINAL_HALF
        self._box(
            builder,
            origin,
            quat,
            centre=(-(TERMINAL_HALF - back), 0.0, wall_h),
            half=(back, TERMINAL_CHANNEL, wall_h),
            tag="back",
        )


CLEAT_POST_H = 0.030
"""How tall the standoff is [m]. Above the cable, so a run cannot ride off the
top under tension -- the failure a plain post has and the reason the spool
variant uses a grooved fixture.

The RADIUS is :data:`~.fixtures.CLEAT_POST_RADIUS`, declared beside the record
that states the ``shaft`` feature about it, because that number is what the
difficulty axis reads: a crossing's turn is set by how far off a station's axis
a seated cable rests, and 9.5 mm against the F1's 4.75 is why five cleats
accumulate 143 degrees where five spools would turn 90."""

CLEAT_FOOT = (0.014, 0.020, 0.0035)
"""Half-extents of the bolted foot [m]: 28 x 40 x 7 mm."""

CLEAT_CAP = (0.016, 0.016, 0.0030)
"""Half-extents of the retaining cap on top [m]. Wider than the post, so the
part reads as a cleat rather than a pin, and so a cable lifted by tension meets
an overhang instead of open air."""

CLEAT_BOLT_R = 0.0028
CLEAT_BOLT_H = 0.0012
CLEAT_BOLT_Y = 0.0135
"""The two bolt heads on the foot [m] -- visual only, like the terminals'."""

CLEAT_STEEL = (0.46, 0.48, 0.52)
CLEAT_DARK = (0.24, 0.25, 0.28)
"""The standoff and bolts, and the foot [RGB in 0..1]. The CAP keeps the spec's
own colour for the reason :data:`TERMINAL_METAL` records: it is the tallest and
most face-on part, so it is where a grounding prompt reads the mask."""


class EntityCableCleat(EntityCableTerminal):
    """A bolted cable cleat: foot, standoff, retaining cap.

    **The same question as a spool, asked by a different part.** A weave scores
    on which side of a station the cable finishes, so what a fixture must do is
    stand up, be found by a camera, and hold a route the hand has let go of.
    The F1 spool does that with a groove between two flanges. A cleat does it
    with a post and an overhanging cap, which is how a harness is dressed on a
    machine: bolted to a backplane, the run passing between standoffs.

    Subclasses the terminal because it needs exactly what the terminal already
    has -- box and cylinder helpers in the fixture's own frame, the spools' own
    contact settings so two fixtures on one bench answer a cable the same way,
    and the ``build`` that resolves the episode's pose before drawing (the trap
    :class:`~robosimstudio.domains.rod.EntityPeg` records, where a sampler moves
    the scored region and leaves the fixture behind).

    Nothing here reads a world coordinate: the assembly is drawn about local
    ``+z`` and the spec's xform places it, so one class serves a board an
    episode has yawed and translated.
    """

    def _draw(self, builder, origin, quat) -> None:
        # The foot. It is what stands on the bench, and it is deliberately wider
        # across the run than along it -- a cleat resists the cable's pull
        # sideways, and the footprint says so.
        self._box(
            builder,
            origin,
            quat,
            centre=(0.0, 0.0, CLEAT_FOOT[2]),
            half=CLEAT_FOOT,
            shade=CLEAT_DARK,
            tag="foot",
        )

        rot = quat_to_mat(quat)

        # The standoff: the surface a cable actually bears against, and the one
        # the asset's ``shaft`` feature is declared about.
        post_c = np.asarray((0.0, 0.0, 2.0 * CLEAT_FOOT[2] + 0.5 * CLEAT_POST_H), dtype=np.float64)
        shape = builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(
                wp.vec3(*[float(v) for v in origin + rot @ post_c]),
                wp.quat(*[float(v) for v in quat]),
            ),
            radius=CLEAT_POST_RADIUS,
            half_height=0.5 * CLEAT_POST_H,
            cfg=fixture_shape_cfg(builder, collides=True),
            label=f"{self.spec.name}_post",
        )
        self._parts[int(shape)] = CLEAT_STEEL

        # The cap. Visual and geometric both: it is the overhang a cable rides
        # into instead of off, and it is most of what makes the part legible
        # from the cable camera, which looks down the posts at 40 degrees.
        cap_z = 2.0 * CLEAT_FOOT[2] + CLEAT_POST_H + CLEAT_CAP[2]
        self._box(builder, origin, quat, centre=(0.0, 0.0, cap_z), half=CLEAT_CAP, shade=None, tag="cap")

        # Two bolt heads on the foot. Visual only -- a 2.8 mm boss proud of a
        # plate is a thing to look at, not a thing to plan around, and admitting
        # it to the contact set only gives a fingertip somewhere to catch. The
        # terminals' screws are declared the same way for the same reason.
        for sign in (-1.0, +1.0):
            centre = np.asarray((0.0, sign * CLEAT_BOLT_Y, 2.0 * CLEAT_FOOT[2] + CLEAT_BOLT_H), dtype=np.float64)
            bolt = builder.add_shape_cylinder(
                body=-1,
                xform=wp.transform(
                    wp.vec3(*[float(v) for v in origin + rot @ centre]),
                    wp.quat(*[float(v) for v in quat]),
                ),
                radius=CLEAT_BOLT_R,
                half_height=CLEAT_BOLT_H,
                cfg=fixture_shape_cfg(builder, collides=False),
                label=f"{self.spec.name}_bolt",
            )
            self._parts[int(bolt)] = CLEAT_STEEL
