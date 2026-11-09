# SPDX-License-Identifier: Apache-2.0
"""The assembly board: a plate, a row of pegs, a driven pinion and a start button.

**The pegs are where the gears mesh, and nobody chose where that is.** Their
spacing comes out of ``gearing.chain_centres`` -- ``module * (z_a + z_b) / 2``
for each neighbouring pair -- so a board is assemblable by construction and a
board that is not cannot be authored. Isaac-CaP's ``gear_insertion`` states its
three post offsets as measured constants; this states the rule they are an
instance of.

**Topology, the ``polish`` machine's.** Plate, pegs and button housing are
shapes on body ``-1``: world-fixed scenery, nothing to integrate. Only two
things move -- the pinion, on a velocity-driven revolute to the world, and the
button's plunger, on a prismatic. Every gear the *robot* handles is a separate
free body from ``gear.py``; the board never owns one.

**Why there is a pinion at all.** "The gears turn" needs something to turn them,
and it may not be a gear the policy places, or the task would score its own
setup. So the board carries a motor pinion the robot never touches, meshing with
the first peg. The chain that follows is the assembly: a gear missing from the
middle stops everything downstream of it, which is exactly the property that
makes "all of them turn" worth measuring.

**The drive is gated inside the captured graph.** `polish` measured that a
revolute with ``target_ke=0, target_kd=kv`` infers ``JointTargetMode.VELOCITY``
and SolverMuJoCo installs a velocity servo clamped by ``effort_limit``, reading
its target from ``control.joint_target_qd`` every substep. So the button does
not need a host round trip: one kernel on the ``rigid_force`` phase reads the
plunger's own coordinate, latches, and writes the pinion's target. Before the
press the target is zero and the train is dead; after it, and for good, it is
``drive_speed``. The latch is a device array because the kernel that sets it and
the kernel that reads it are both inside the graph -- ``box_decant``'s seal tape
holds its flaps the same way, and for the same reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render.palette import ACCENT_PEG, ACCENT_PLUG, ACCENT_PROP, MACHINE
from robosimstudio.scene.entity import Entity, label_index
from robosimstudio.scene.registry import register_entity

from .gearing import BORE_RADIUS, FACE, chain_centres, gear_frame, gear_parts

__all__ = ["EntityGearBoard", "SpecGearBoard", "board_frame", "forward_of", "peg_mesh", "plate_centre", "plate_corners", "yaw_of"]

SPIN_GATE_FRACTION = 0.3
"""Share of its ideal ratio at which a gear counts as turning.

Stated as a fraction rather than a rate because the ideal differs per station --
the chain divides the pinion's speed by each gear's tooth count -- and a single
number would be generous at one end of the board and impossible at the other.
**There is no middle population, which is what lets the gate sit low.** A gear in
a complete chain holds 0.89-0.95 of its ideal; a gear whose chain is broken
upstream is driven by nothing and reads exactly 0. Nothing lands in between --
a gear being turned by the gripper is not seated, and fails the other half of
the clause. So the gate is placed for margin against the *dips*, not against a
rival population: on the four-gear chain, which is the worst case, the 5th
percentile of a healthy windowed reading is 0.51, and 0.3 clears it with room
while still being six times anything a stopped gear produces. Half was tried
first and is too close -- a `hold-for` behind it broke on the ripple."""

SPIN_WINDOW = 0.5
"""How long a rate is averaged over [s].

**A driven gear's instantaneous rate is not its rate.** It turns through its own
backlash: free between the flanks either side of it until a tooth catches, then
hit. Measured on the shipped three-gear chain at 4 rad/s, with every gear holding
its ideal ratio to within 6% on the mean, the 5th percentile of the instantaneous
reading is 0.35, 0.14 and 0.05 of ideal going down the chain -- so a gate placed
anywhere useful is crossed downward several times a second and every ``hold-for``
sequenced behind it restarts forever.

So a rate here is **net angle swept over a window**, not a sampled angular
velocity. That is what "this gear is turning at 3 rad/s" means about a rattling
gear, and it is the same answer the ``contact_force`` atom reached for the same
problem -- a signal whose median is fine and whose individual samples drop out.
Half a second is several teeth at the slowest station, and short enough that a
gear knocked loose reads stopped well inside the goal's own hold. A quarter was
tried: enough for the three-gear chain, not enough for the four-gear one, whose
extra compliance shows up as a slow wind-and-release the shorter window follows
instead of averaging out."""

PEG_SIDES = 16
"""Facets on a peg's collider prism.

Never a CYLINDER primitive: mujoco_warp's multi-point manifold covers box and
mesh pairs under ``enable_multiccd`` and never cylinders, and ``polish`` measured
what that costs -- one contact point, and a part that visibly rocks. A gear rides
its peg on the inside of a bore, which is the one contact that decides whether it
stays upright, so it gets facets."""


def board_frame(spec: SpecGearBoard) -> dict[str, Any]:
    """Every length the board is built from, derived once.

    The board's own frame: origin at the centre of the plate's **top face**, +x
    along the chain from the pinion outward, +z up. So a gear seated on a peg
    has its centre at ``z = face / 2`` and nothing has to add a plate thickness
    twice.

    Returns a dict carrying the gear frames as well, because the peg radius, the
    plate's extent and the seated heights are all consequences of them.
    """
    counts = (int(spec.pinion_teeth), *(int(z) for z in spec.stations))
    frames = [
        gear_frame(
            module=spec.module,
            teeth=z,
            face=spec.face,
            bore_radius=spec.bore_radius,
            backlash=spec.backlash,
        )
        for z in counts
    ]
    centres = chain_centres(spec.module, counts)
    # The chain as a polyline in the board's own plane: the link lengths are the
    # centre distances, and the bends say which way each one points.
    links = [far - near for near, far in pairwise(centres)]
    bends = tuple(float(b) for b in spec.bends)
    if len(bends) > max(0, len(links) - 1):
        raise ValueError(f"{len(bends)} bends for a chain of {len(links)} links; one per link after the first")
    heading = 0.0
    points = [(0.0, 0.0)]
    for index, length in enumerate(links):
        if index:
            heading += bends[index - 1] if index - 1 < len(bends) else 0.0
        points.append((points[-1][0] + length * math.cos(heading), points[-1][1] + length * math.sin(heading)))
    points = tuple(points)
    # The peg is sized against what the hub's collider actually leaves, not
    # against the drawn bore: the hull of an annular sector chords its inner arc.
    # Getting this wrong is the "fingertip hull is a phantom" failure -- a peg
    # that fits the drawing and jams in the model.
    bore_effective = min(f["bore_effective"] for f in frames[1:])
    peg_radius = bore_effective - spec.peg_clearance
    if peg_radius <= 0.0:
        raise ValueError(f"peg_clearance={spec.peg_clearance} leaves no peg inside a {bore_effective:.4f} m bore")

    # The plate carries whatever the polyline spans, which for a straight chain
    # is the old span exactly.
    reach = max(p[0] + f["r_tip"] for p, f in zip(points, frames, strict=True))
    back = min(p[0] - f["r_tip"] for p, f in zip(points, frames, strict=True))
    left = min(p[1] - f["r_tip"] for p, f in zip(points, frames, strict=True))
    right = max(p[1] + f["r_tip"] for p, f in zip(points, frames, strict=True))
    button_x = back - spec.button_gap - spec.button_radius
    half_x = 0.5 * (reach - button_x) + spec.plate_margin
    half_y = 0.5 * (right - left) + spec.plate_margin
    return {
        "counts": counts,
        "frames": frames,
        "centres": centres,
        "points": points,
        "links": tuple(links),
        "peg_radius": float(peg_radius),
        "peg_height": float(spec.face + spec.peg_lead),
        "seat_z": 0.5 * float(spec.face),
        "plate_half": (float(half_x), float(half_y), 0.5 * float(spec.plate_thickness)),
        # Centred on the span it has to carry, which runs from behind the button
        # to past the last gear's tip -- not on the pinion, which is at one end.
        "plate_centre_x": float(0.5 * (reach + button_x)),
        "plate_centre": (float(0.5 * (reach + button_x)), float(0.5 * (right + left))),
        "button_at": (float(button_x), 0.0),
    }


def peg_mesh(
    radius: float, height: float, taper: float = 0.0, tip_scale: float = 0.55, sides: int = PEG_SIDES
) -> tuple[np.ndarray, np.ndarray]:
    """A shaft standing on the plate, optionally coned at the top.

    **The taper is the lead-in, and without it the task cannot be done at all.**
    The fit is 0.3 mm of radial clearance; a solved arm pose lands within about
    1.5 mm of where it was asked to be, and servo tracking under a carried load
    adds more. A gear lowered onto a straight shaft at that error does not go on
    -- it lands on the shaft's top face and stops. Coning the top few
    millimetres turns a miss into a slide: the bore meets a slope, the gear is
    free to move in the jaws' compliance, and it drops on.

    Isaac-CaP's own gear base tapers its posts for the same reason -- its
    ``_author_convex_peg`` builds each one with a top radius strictly under the
    bottom. This is that, generated rather than recovered from a mesh.

    Convex by construction: a truncated cone on a cylinder is convex as long as
    the tip is the narrow end.
    """
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    rings = (
        [(radius, 0.0), (radius, height)]
        if taper <= 0.0
        else [
            (radius, 0.0),
            (radius, height - taper),
            (radius * tip_scale, height),
        ]
    )
    verts = np.vstack([np.hstack([unit * r, np.full((sides, 1), z)]) for r, z in rings])
    verts = np.vstack([verts, [[0.0, 0.0, rings[-1][1]]], [[0.0, 0.0, 0.0]]]).astype(np.float32)
    top, bottom = len(verts) - 2, len(verts) - 1
    last = sides * (len(rings) - 1)
    idx: list[int] = []
    for i in range(sides):
        j = (i + 1) % sides
        idx += [last + i, last + j, top]
        idx += [j, i, bottom]
        for r in range(len(rings) - 1):
            lo, hi = sides * r, sides * (r + 1)
            idx += [lo + i, hi + i, lo + j, lo + j, hi + i, hi + j]
    return verts, np.asarray(idx, dtype=np.int32)


@dataclass(frozen=True, kw_only=True)
class SpecGearBoard:
    """The board's knobs. Poses are in the work-surface frame.

    Attributes:
        stations: Tooth counts of the gears the pegs are cut for, in chain order
            from the pinion outward. Its length is the peg count, and changing it
            is what makes one variant harder than another.
        gears: The object names those stations expect, same order. The board
            resolves them at bind so it can publish their states -- an
            ``ObjectSpec`` may not publish its own.
        drive_speed: Pinion speed once the button latches [rad/s]. Every gear's
            ideal rate follows: ``drive_speed * pinion_teeth / teeth``.
        drive_tau_max: Torque ceiling on the servo [N*m]. What makes a jammed
            train stall honestly instead of forcing its way through.
        press_latch: Plunger travel that starts the motor [m]. Under the full
            ``button_travel`` so the press does not have to bottom out.
    """

    stage: ClassVar[str] = "extras"
    entity: str = "gear_board"
    name: str = "board"
    at: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    """The board's own frame: where the PINION END of the chain stands and which way
    the chain runs. What the entity builds from (``EntityGearBoard._world``)."""

    pose: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None
    """Where the PLATE CENTRE stands, and how the board is turned -- the shape a
    document's ``(:start-poses ...)`` file records for its ``board`` fixture.

    **Optional, and when given they are the placement.** ``at`` is the pinion end
    of the chain, which is a fine frame to build from and a poor one to record: it
    moves with the stations, because the plate is centred on the span it carries
    and the pinion sits at one end of it. A drawn variant states where the plate
    is and which way it faces (``pose`` in metres, surface frame, z ignored;
    ``quat`` xyzw about +z), and ``__post_init__`` derives ``at`` and ``yaw`` from
    them, so a document moves the board the way it moves a gear -- through the
    poses file the loader overlays (``taskbuild.place.build_entities`` copies
    ``pose`` and ``quat`` onto whatever the record built). Left ``None``, ``at``
    and ``yaw`` stand as written and nothing changes."""

    module: float = 0.0025
    pinion_teeth: int = 14
    stations: tuple[int, ...] = (16, 20, 24)
    bends: tuple[float, ...] = ()
    """How far the chain turns at each station after the first [rad], in order.

    **A train does not have to be a straight line, and the arithmetic never
    said it did.** Two external gears mesh when their centres are
    ``module * (z_a + z_b) / 2`` apart, which fixes the LENGTH of each link and
    says nothing about its direction; a chain is a polyline whose segment
    lengths are those distances. Empty leaves every link pointing the same way,
    which is the straight board the hand-written tiers ship and what every
    number measured on them refers to. One angle per link after the first turns
    the chain at that station: what changes is where the pegs are and how big a
    plate has to be to carry them, and what does not change is that every
    neighbouring pair still meshes."""
    gears: tuple[str, ...] = ()
    face: float = FACE
    bore_radius: float = BORE_RADIUS
    backlash: float = 0.0

    peg_clearance: float = 6.0e-4
    """Radial slack between peg and bore [m].

    **Set by how far a released gear may cock, not by how well it is located.**
    A bore binds on its shaft at about ``atan(2 * clearance / face)``, so the
    0.3 mm this first shipped with allowed 3.4 degrees -- and a gear let go of by
    two pads that grip its rim is free to rock about the line joining them.
    Measured: every gear went onto its peg within a millimetre laterally and
    jammed nine millimetres up. 0.6 mm doubles the angle a gear may be released
    at and still fall.

    The cost is a centre distance that can be up to two clearances short of
    nominal, which matters because teeth bind when brought together -- but only
    in principle: a driven mesh pushes its gears APART along the line of centres,
    so a running train sits at the far end of its own slack rather than the near
    one. Verified by ``tests/test_gear_mesh_drive.py``, which drives the
    assembled board at this fit."""
    peg_taper: float = 0.010
    peg_tip_scale: float = 0.22
    """The coned lead-in: how tall it is [m], and how sharp.

    **Sized off the miss, not off the fit.** What has to fit inside the bore is
    the cone's TIP, because that is what arrives first: land the tip on the bore's
    rim instead of inside it and the gear does not slide, it tips -- measured at
    11 to 21 degrees, jammed near the top of the shaft, with every gear placed
    within 1 to 3 mm of its peg. A 0.55 tip is 2.9 mm on this peg against a
    5.9 mm bore, which is only 3 mm of margin and exactly the miss. At 0.22 the
    tip is 1.2 mm and a 3 mm miss still drops it in with room to spare."""
    peg_lead: float = 0.022
    """How far a peg stands proud of a seated gear [m].

    **Mechanism, not trim.** A gear is lowered beside a neighbour that is already
    meshed, and its teeth have to find that neighbour's spaces. It finds them by
    turning freely on the peg -- so the peg must capture the bore *before* the
    teeth are in the same plane as anything. It also has to be long enough to
    carry the coned lead-in above the shaft a seated gear needs -- AND a
    staging zone above that.

    **The staging zone is the part that is mechanism.** A gear cannot be pushed
    all the way down onto its peg while gripped, because the last thing it has to
    do is turn until its teeth find its neighbour's spaces, and jaws stop it
    turning. So it is let go once the bore is on parallel shaft but the teeth are
    still above the neighbour's -- which needs a peg with a face-width of
    parallel shaft ABOVE the seated gear, before the cone starts. 22 mm is one
    face of that staging plus the 10 mm cone."""
    plate_thickness: float = 0.006
    plate_margin: float = 0.012

    peg_mu: float = 0.15
    """Friction between a peg and the bore riding on it.

    **A peg is a bearing, not a grip.** It inherited the plate's 0.8 in the first
    build, and a 0.8 journal is a brake: the pinion stalled at 0.6 of a commanded
    6 rad/s and every gear turned at a tenth of its ratio, which reads exactly
    like a mesh that cannot transmit and is nothing of the kind -- the mesh was
    carrying the whole train faithfully, into a bore that would not let it turn.
    0.15 is a dry steel-on-steel journal."""

    drive_speed: float = 4.0
    """Pinion speed once the button latches [rad/s].

    Measured down from 6. At 6 the train runs and every gear holds its ratio to
    within 6%; at 4 the same chain holds it to within 6% with less than half the
    jitter, and a 45 mm gear at 4 rad/s has a rim speed of 9 cm/s, which is a
    train a camera can see turning rather than a blur."""
    drive_kv: float = 4.0
    """Velocity-servo gain on the pinion [N*m*s/rad].

    **The number the bench was written to find, and the one that made the
    difference between a working train and an apparently dead one.** A gear train
    is a far heavier load than ``polish``'s free-spinning drum, whose servo runs
    at 0.01: at that gain the pinion held 0.6 of a commanded 6 rad/s and every
    gear turned at a tenth of its ratio -- which reads exactly like a mesh that
    cannot transmit, and was nothing of the kind. The mesh was carrying the whole
    train faithfully; the servo simply had no authority. Measured on the
    three-gear chain: kv 0.02 -> 0.10 of ratio, 0.5 -> 0.83, 1.0 -> 0.89,
    2.0 -> 0.94-1.02, 5.0 -> 0.97-1.00 -- and 2.0 is the knee for THREE gears.
    Four need more: at 2.0 the quad chain runs at 0.79 of ideal, at 4.0 at 0.89,
    at 8.0 at 0.87. 4.0 is the knee for the longest board this suite ships, and
    the suite ships one number rather than one per variant, because a drive that
    changed with the task would make the variants incomparable."""
    drive_tau_max: float = 1.5
    """Torque ceiling on the servo [N*m].

    Running torque on the three-gear chain is about 0.6 N*m, all of it contact
    dissipation in the teeth and bores -- a lot for parts weighing tens of
    grams, and a fact about the contact model rather than about gears. The
    ceiling is a bit over twice that, so the motor has margin on a clean train
    and **stalls honestly on a jammed one** instead of forcing its way through.
    A ceiling never reached is not a ceiling."""
    drive_armature: float = 3.0e-4
    """Rotor inertia on the pinion [kg*m^2].

    ``polish``'s ``spindle_armature`` move and ``box_decant``'s ``flap_armature``
    move: dominate the part's own inertia so the servo's gains are nowhere near
    the ``kp < I/dt^2`` bound an explicit drive is held to. A 14-tooth gear at
    this module is ~7 g and ~1e-6 kg*m^2, so this is two orders over it."""

    button_radius: float = 0.011
    button_height: float = 0.008
    button_clearance: float = 0.0015
    """Radial gap between the plunger and its well [m].

    Measured, not chosen. The first build set the walls' inner faces exactly on
    the cap's radius: a zero-clearance fit, four walls of it, and at mu ~1 the
    friction stalled the press at 4.01 mm of an 8 mm stroke -- under the latch,
    so the motor could never start and the whole train read as dead. A plunger
    needs a slide fit, not a locating one."""
    button_travel: float = 0.008
    button_gap: float = 0.018
    button_ke: float = 400.0
    button_kd: float = 4.0
    button_armature: float = 1.0e-3
    press_latch: float = 0.005
    button_mass: float = 0.02

    spin_gate: float = 0.0
    """Absolute rate above which every gear counts as turning [rad/s].

    0, the default, is the right answer and this field is the escape hatch. Each
    station's gate is then :data:`SPIN_GATE_FRACTION` of **its own** ideal rate,
    which differs per station because the chain divides the pinion's speed by
    each tooth count -- so one absolute number is generous at one end of the
    board and impossible at the other. Set it only to probe what a flat
    threshold would have done."""

    def __post_init__(self) -> None:
        if len(self.stations) < 1:
            raise ValueError("a board needs at least one peg")
        if self.gears and len(self.gears) != len(self.stations):
            raise ValueError(f"{len(self.gears)} gear names for {len(self.stations)} stations")
        frame = board_frame(self)
        # A recorded placement wins over the frame fields: the poses file says
        # where the plate centre is, and the pinion end follows from the span.
        if self.quat is not None:
            object.__setattr__(self, "yaw", yaw_of(self.quat))
        if self.pose is not None:
            offset = float(frame["plate_centre_x"])
            fx, fy = math.cos(self.yaw), math.sin(self.yaw)
            object.__setattr__(
                self, "at", (float(self.pose[0]) - fx * offset, float(self.pose[1]) - fy * offset)
            )


def yaw_of(quat: tuple[float, float, float, float]) -> float:
    """The heading a unit quaternion (xyzw) turns +x to, about +z [rad]."""
    x, y, z, w = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def forward_of(spec: SpecGearBoard) -> tuple[float, float]:
    """The unit vector the chain runs along, pinion outward, in the surface frame."""
    return (math.cos(float(spec.yaw)), math.sin(float(spec.yaw)))


def plate_centre(spec: SpecGearBoard) -> tuple[float, float]:
    """Where the plate's centre stands in the surface frame -- ``pose``'s xy when one was given."""
    fx, fy = forward_of(spec)
    offset = float(board_frame(spec)["plate_centre_x"])
    return (float(spec.at[0]) + fx * offset, float(spec.at[1]) + fy * offset)


def plate_corners(spec: SpecGearBoard) -> tuple[tuple[float, float], ...]:
    """The plate's four corners in the surface frame, going round.

    One statement of the footprint for the layout (a loose gear may not start on
    the plate), the generator (its rejection rule) and the tests (the assertion
    that they agree), whatever the board's yaw.
    """
    cx, cy = plate_centre(spec)
    fx, fy = forward_of(spec)
    ax, ay = -fy, fx
    hx, hy, _ = board_frame(spec)["plate_half"]
    return tuple(
        (cx + fx * sx * hx + ax * sy * hy, cy + fy * sx * hx + ay * sy * hy)
        for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    )


@register_entity("gear_board")
class EntityGearBoard(Entity):
    """Plate, pegs, pinion and button -- and everything a goal reads about them."""

    spec: SpecGearBoard

    def __init__(self, spec: SpecGearBoard, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.geometry = board_frame(spec)
        self.pinion_body = -1
        self.pinion_joint = -1
        self.pinion_dof = -1
        self.button_body = -1
        self.button_coord = -1
        self.button_dof = -1
        self.gear_bodies: list[int] = []
        self._latched: wp.array | None = None
        self._target_qd: wp.array | None = None
        self._gains: wp.array | None = None
        self._swept: np.ndarray | None = None
        self._angle: np.ndarray | None = None
        self._trace: list[tuple[float, np.ndarray]] = []
        self._clock = 0.0
        self._seat_world: np.ndarray | None = None
        self._up_world: np.ndarray | None = None
        self._along: np.ndarray | None = None
        self._across: np.ndarray | None = None
        self._button_rest: np.ndarray | None = None
        self._published_at: tuple[Any, dict[str, float]] | None = None

    # ------------------------------------------------------------------ build
    def _world(self, local: wp.transform) -> wp.transform:
        """A board-local transform in the scene's work-surface frame."""
        spec = self.spec
        root = wp.transform(
            wp.vec3(float(spec.at[0]), float(spec.at[1]), float(spec.plate_thickness)),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(spec.yaw)),
        )
        seven = self.placed(np.array([*wp.transform_multiply(root, local)], dtype=np.float64))
        return wp.transform(
            wp.vec3(float(seven[0]), float(seven[1]), float(seven[2])),
            wp.quat(float(seven[3]), float(seven[4]), float(seven[5]), float(seven[6])),
        )

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec, frame = self.spec, self.geometry
        ident = wp.quat_identity()
        scenery = builder.default_shape_cfg.copy()
        scenery.mu = 0.8

        half = frame["plate_half"]
        builder.add_shape_box(
            -1,
            xform=self._world(
                wp.transform(wp.vec3(frame["plate_centre"][0], frame["plate_centre"][1], -half[2]), ident)
            ),
            hx=half[0],
            hy=half[1],
            hz=half[2],
            cfg=scenery,
            color=MACHINE,
            label=f"{spec.name}_plate",
        )
        bearing = builder.default_shape_cfg.copy()
        bearing.mu = float(spec.peg_mu)
        bearing.ke, bearing.kd, bearing.gap = 6.0e4, 500.0, 5.0e-5
        peg_verts, peg_idx = peg_mesh(
            frame["peg_radius"], frame["peg_height"], float(spec.peg_taper), float(spec.peg_tip_scale)
        )
        for index, point in enumerate(frame["points"][1:]):
            builder.add_shape_mesh(
                -1,
                xform=self._world(wp.transform(wp.vec3(float(point[0]), float(point[1]), 0.0), ident)),
                mesh=newton.Mesh(peg_verts, peg_idx),
                cfg=bearing,
                # The brightest thing on the plate: a peg is the target, and in
                # the mat's own grey it rendered as a dark wedge nobody could
                # find in the overhead frame (measured, outputs/snapshots/gear_before);
                # the palette entry says why not METAL either.
                color=ACCENT_PEG,
                label=f"{spec.name}_peg_{index}",
            )

        self._build_pinion(builder, scenery)
        self._build_button(builder, scenery)

    def _build_pinion(self, builder: newton.ModelBuilder, scenery) -> None:
        """The motor gear: one dynamic link on a velocity-driven revolute."""
        spec, frame = self.spec, self.geometry
        at = self._world(
            wp.transform(
                wp.vec3(float(frame["points"][0][0]), float(frame["points"][0][1]), float(frame["seat_z"])),
                wp.quat_identity(),
            )
        )
        link = builder.add_link(xform=at, label=f"{spec.name}_pinion")
        cfg = builder.default_shape_cfg.copy()
        cfg.mu = 1.0
        cfg.ke, cfg.kd, cfg.gap = 6.0e4, 500.0, 5.0e-5
        for label, verts, indices in gear_parts(frame["frames"][0], tip_chamfer=6.0e-4):
            builder.add_shape_mesh(
                link,
                mesh=newton.Mesh(verts, indices),
                cfg=cfg,
                color=ACCENT_PLUG,
                label=f"{spec.name}_pinion_{label}",
            )
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            parent_xform=at,
            child_xform=wp.transform(),
            axis=(0.0, 0.0, 1.0),
            # Zero, not drive_speed: the motor is OFF until the button latches,
            # and the target is rewritten every substep from inside the graph.
            target_vel=0.0,
            target_ke=0.0,
            target_kd=float(spec.drive_kv),
            actuator_mode=newton.JointTargetMode.VELOCITY,
            effort_limit=float(spec.drive_tau_max),
            armature=float(spec.drive_armature),
            label=f"{spec.name}_pinion",
        )
        builder.add_articulation([joint], label=f"{spec.name}_pinion")
        self.pinion_body, self.pinion_joint = int(link), int(joint)
        self.pinion_dof = int(builder.joint_qd_start[joint])
        del scenery

    def _build_button(self, builder: newton.ModelBuilder, scenery) -> None:
        """A sprung plunger in a well: the repo's first prismatic joint.

        The spring is the joint's own position servo held at zero, which is what
        a return spring is. Travel is read off the plunger's body pose rather
        than its joint coordinate, because the evaluator's context carries body
        poses and not joint ones.

        **Every height here is measured from the plate's top face, and nothing
        may reach beneath it.** The first build hung the cap half below its link
        origin, which put its underside 4 mm under the plate at full stroke: the
        press bottomed out on the plate at 4.013 mm against a 5 mm latch, so the
        button could not be pressed at all, the motor never started, and a
        perfectly good train read as dead. There were no contacts to find, which
        is what made it worth writing down -- the plunger simply had nowhere left
        to go.
        """
        spec, frame = self.spec, self.geometry
        bx, by = frame["button_at"]
        ident = wp.quat_identity()
        # Cap fully up: [travel, travel + height]. Cap fully down: [0, height].
        well = float(spec.button_travel) + float(spec.button_height)
        # **No well around the plunger, and that is a fix rather than a
        # simplification.** It had one -- four low walls, guiding a stroke the
        # prismatic joint already constrains completely -- and they guided the
        # gripper straight into themselves: commanded 7 mm further down, the
        # tool stopped dead with its closed fingers resting on the wall tops and
        # the button 0.16 mm depressed against a 5 mm latch. The motor could
        # never start, which reads in a trace as a train that will not turn and
        # is nothing of the kind. A button has to be reachable by the thing that
        # presses it; the joint does the guiding.
        at = self._world(wp.transform(wp.vec3(bx, by, well), ident))
        link = builder.add_link(xform=at, label=f"{spec.name}_button")
        cap = builder.default_shape_cfg.copy()
        cap.mu = 1.2
        cap.ke, cap.kd = 6.0e4, 500.0
        cap.density = float(spec.button_mass) / (math.pi * spec.button_radius**2 * spec.button_height)
        verts, indices = peg_mesh(float(spec.button_radius), float(spec.button_height))
        verts[:, 2] -= float(spec.button_height)
        builder.add_shape_mesh(
            link,
            mesh=newton.Mesh(verts, indices),
            cfg=cap,
            color=ACCENT_PROP,
            label=f"{spec.name}_button_cap",
        )
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=link,
            parent_xform=at,
            child_xform=wp.transform(),
            axis=(0.0, 0.0, 1.0),
            target_pos=0.0,
            target_ke=float(spec.button_ke),
            target_kd=float(spec.button_kd),
            actuator_mode=newton.JointTargetMode.POSITION,
            limit_lower=-float(spec.button_travel),
            limit_upper=0.0,
            armature=float(spec.button_armature),
            label=f"{spec.name}_button",
        )
        builder.add_articulation([joint], label=f"{spec.name}_button")
        self.button_body, self.button_joint = int(link), int(joint)
        self.button_coord = int(builder.joint_q_start[joint])
        self.button_dof = int(builder.joint_qd_start[joint])

    # ------------------------------------------------------------------- bind
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        spec = self.spec
        labels = list(model.body_label)
        self.gear_bodies = [label_index(labels, name) for name in spec.gears]
        device = model.device
        self._latched = wp.zeros(1, dtype=float, device=device)
        self._gains = wp.array(
            [float(spec.press_latch), float(spec.drive_speed), float(self.pinion_dof), float(self.button_coord)],
            dtype=float,
            device=device,
        )

    def contribute_hooks(self, hooks) -> None:
        hooks.add("rigid_force", self._drive, name="gear-board-drive", cadence=1)
        hooks.add("after", self._track, name="gear-board-spin", cadence=1)

    def _track(self, state, dt) -> None:
        """Once per control step, eagerly: unwrap each gear's turn about the axis.

        Accumulating the *swept* angle rather than sampling a velocity is the
        whole point -- see :data:`SPIN_WINDOW`. The unwrap is what makes it work
        past half a turn: at 4 rad/s and 60 Hz a gear moves 0.07 rad a step, so
        a wrap can only ever mean it went the short way.
        """
        if self._swept is None or not self.gear_bodies:
            return
        body_q = state.body_q.numpy()
        for index, body in enumerate((*self.gear_bodies, self.pinion_body)):
            angle = self._turn_of(body_q[body])
            step = (angle - self._angle[index] + math.pi) % (2.0 * math.pi) - math.pi
            self._swept[index] += step
            self._angle[index] = angle
        self._clock += float(dt)
        self._trace.append((self._clock, self._swept.copy()))
        while len(self._trace) > 1 and self._clock - self._trace[0][0] > SPIN_WINDOW:
            self._trace.pop(0)

    def _turn_of(self, pose: np.ndarray) -> float:
        """A body's rotation about the board's up axis [rad], wrapped.

        Read off ``body_q`` because the evaluator has no joint coordinates and a
        free gear has no joint to read anyway: the angle of the body's own +x,
        projected into the plane the board's up axis is normal to.
        """
        forward = _forward_of(pose[3:7])
        flat = forward - float(np.dot(forward, self._up_world)) * self._up_world
        return float(math.atan2(float(np.dot(flat, self._across)), float(np.dot(flat, self._along))))

    def _rates(self) -> np.ndarray:
        """Windowed rate of every tracked body [rad/s], gears then the pinion."""
        if self._swept is None:
            return np.zeros(0)
        if len(self._trace) < 2:
            return np.zeros(len(self._swept))
        (t0, a0), (t1, a1) = self._trace[0], self._trace[-1]
        span = t1 - t0
        return (a1 - a0) / span if span > 0.0 else np.zeros(len(self._swept))

    def _drive(self, state, dt) -> None:
        del dt
        if self._target_qd is None:
            return
        wp.launch(
            _drive_kernel,
            dim=1,
            inputs=[state.joint_q, self._gains, self._latched, self._target_qd],
            device=self.model.device,
        )

    def on_scene_ready(self, pipeline) -> None:
        """Capture the drive's target array, and the board's axes in world.

        The board's own frame is read off the **pinion's rest pose** rather than
        recomposed from ``at``/``yaw`` and the cell's surface transform. Two
        reasons, and the second is the load-bearing one: the entity would
        otherwise restate a frame composition the compiler already did, which is
        the duplication ``board_frame`` exists to stop; and the poses a goal
        reads arrive in world coordinates, so a surface-frame answer would be
        silently one origin out. Read at rest, before anything spins, because a
        turning pinion's own orientation is no longer the board's.
        """
        self._target_qd = pipeline.control.joint_target_qd
        body_q = pipeline.state_0.body_q.numpy()
        at = np.asarray(body_q[self.pinion_body][:3], dtype=np.float64)
        quat = np.asarray(body_q[self.pinion_body][3:7], dtype=np.float64)
        self._up_world = _axis_of(quat)
        forward = _forward_of(quat)
        # A fixed pair of axes in the plane the gears turn in, so an angle read
        # off a spinning body is measured against the BOARD rather than against
        # itself. Taken once, at rest, with the pinion.
        self._along = forward
        self._across = np.cross(self._up_world, forward)
        points = self.geometry["points"]
        # Every peg's seated gear centre, in world. The pinion already sits at
        # seat height, so this is an offset in the board's own plane and no z
        # arithmetic -- along the chain's first link and across it, which is the
        # single offset it used to be whenever the chain does not bend.
        across = np.cross(self._up_world, forward)
        self._seat_world = np.stack(
            [at + forward * (p[0] - points[0][0]) + across * (p[1] - points[0][1]) for p in points[1:]]
        )
        self._button_rest = np.asarray(body_q[self.button_body][:3], dtype=np.float64)
        self._swept = np.zeros(len(self.gear_bodies) + 1)
        self._angle = np.array(
            [self._turn_of(body_q[b]) for b in (*self.gear_bodies, self.pinion_body)], dtype=np.float64
        )
        self._trace = [(0.0, self._swept.copy())]
        self._clock = 0.0

    def reset(self, pipeline, episode) -> None:
        """Un-press the button and un-start the motor.

        The latch is irreversible *within* an episode and must not survive one:
        a board that came up already running would score every episode's spin
        clause before the policy touched anything, which is the shape of failure
        the gauntlet's do-nothing floor exists to catch.
        """
        super().reset(pipeline, episode)
        if self._latched is not None:
            self._latched.zero_()
        if self._swept is not None:
            self._swept[:] = 0.0
            # Re-seed the LAST angle too, not just the accumulator. The episode
            # sampler turns the seeded gears, so a stale reference would make the
            # first step of a new episode measure a rotation that happened
            # between episodes -- wrapped to under half a turn by the unwrap, and
            # therefore plausible rather than obviously wrong.
            body_q = pipeline.state_0.body_q.numpy()
            self._angle = np.array(
                [self._turn_of(body_q[b]) for b in (*self.gear_bodies, self.pinion_body)], dtype=np.float64
            )
            self._trace = [(0.0, self._swept.copy())]
            self._clock = 0.0
        self._published_at = None

    # -------------------------------------------------------------- published
    def published_states(self, snapshot) -> dict[str, float]:
        """What a goal may say about this board, off ``body_q`` and ``body_qd``.

        Spin is the world angular velocity projected on the board's own up axis,
        which for a gear seated on a vertical peg is its rate about that peg.
        ``polish``'s spindle reads its speed exactly this way; the alternative,
        reaching for ``context.joint_q``, returns nothing at all -- gears are free
        bodies and have no hinge coordinate to read even if it did.
        """
        context = getattr(snapshot, "ctx", snapshot)
        cached = self._published_at
        if cached is not None and cached[0] is context:
            return cached[1]
        body_q = getattr(context, "body_q", None)
        body_qd = getattr(context, "body_qd", None)
        out: dict[str, float] = {}
        if body_q is not None and body_qd is not None and self.pinion_body >= 0:
            out = self._read(np.asarray(body_q), np.asarray(body_qd))
        self._published_at = (context, out)
        return out

    def _read(self, body_q: np.ndarray, body_qd: np.ndarray) -> dict[str, float]:
        del body_qd  # rates come off the swept-angle window, not off a sample
        spec, frame = self.spec, self.geometry
        rates = self._rates()
        travel = float(np.dot(self._button_rest - body_q[self.button_body][:3], self._up_world))
        out: dict[str, float] = {
            "pinion_speed": abs(float(rates[-1])) if len(rates) else 0.0,
            "press_travel": travel,
            # The LATCH, not the travel. A button released after a press reads
            # zero travel and the motor is still running -- reporting the live
            # depth here would make `started` flicker off under the policy's own
            # hand, and every clause sequenced after it restart.
            "started": float(self._latched.numpy()[0]) if self._latched is not None else 0.0,
        }
        seated = turning = 0
        # A STATION is scored, and any gear of its class may fill it. Two gears
        # of one tooth count are the same part -- the same profile, the same
        # mass, the same finish -- and a policy that cannot tell them apart
        # (none can; the camera sees two identical discs) assembles exactly the
        # same mechanism whichever it puts where. Tying station k to the body
        # the document happened to list k-th scored a working three-gear train
        # 1/3 (gearmesh-medium-train-001, two 16-tooth gears swapped, every
        # gear turning at its ratio). So each station takes the gear of its
        # class that is seated on it, and falls back to its own listing only for
        # the spin readouts of an empty station.
        chosen: list[int] = []
        for index in range(len(self.gear_bodies)):
            teeth = frame["counts"][index + 1]
            same = [
                b
                for k, b in enumerate(self.gear_bodies)
                if frame["counts"][k + 1] == teeth and b not in chosen and self._seated(body_q[b], index)
            ]
            chosen.append(same[0] if same else -1)
        for index, listed in enumerate(self.gear_bodies):
            body = chosen[index] if chosen[index] >= 0 else listed
            body_index = self.gear_bodies.index(body)
            teeth = frame["counts"][index + 1]
            ideal = spec.drive_speed * frame["counts"][0] / teeth
            gate = spec.spin_gate if spec.spin_gate > 0.0 else SPIN_GATE_FRACTION * ideal
            rate = float(rates[body_index]) if body_index < len(rates) else 0.0
            at_peg = chosen[index] >= 0
            out[f"gear{index}_spin"] = abs(rate)
            out[f"gear{index}_signed"] = rate
            # Alternating sign is not decoration: external gears counter-rotate,
            # so a chain whose signs do not alternate is not a chain.
            out[f"gear{index}_ideal"] = ideal if index % 2 else -ideal
            # Normalised, and that is what makes it sayable in a document. The
            # ideal rate differs per station -- the chain divides the pinion's
            # speed by each tooth count -- so a goal written against rad/s would
            # need a different number per gear AND per variant, and every one of
            # them would be a constant nothing checked. A ratio is one threshold
            # for every board this suite ships.
            out[f"gear{index}_ratio"] = abs(rate) / ideal if ideal else 0.0
            out[f"gear{index}_seated"] = 1.0 if at_peg else 0.0
            seated += int(at_peg)
            turning += int(at_peg and abs(rate) >= gate)
        out["gears_seated"] = float(seated)
        out["turning_fraction"] = float(turning) / float(len(self.gear_bodies) or 1)
        return out

    def _seated(self, pose: np.ndarray, index: int) -> bool:
        """Is this gear on station ``index``'s peg, flat and low enough to mesh?

        Deliberately not a bore-over-shaft test: ``encloses(gear.bore, peg)`` is
        the goal language's own way to ask that, it is calibrated, and it is what
        the documents use. This is the board's cheaper internal read, feeding the
        turning fraction the reward is shaped on, and it asks the two things that
        make a spin clause mean something -- the gear is over its own peg, and it
        is lying flat rather than cocked on the rim. A gear spinning in the
        gripper satisfies neither.
        """
        offset = np.asarray(pose[:3], dtype=np.float64) - self._seat_world[index]
        rise = float(np.dot(offset, self._up_world))
        if float(np.linalg.norm(offset - rise * self._up_world)) > self.geometry["peg_radius"]:
            return False
        if abs(rise) > 0.5 * self.spec.face:
            return False
        return float(np.dot(_axis_of(pose[3:7]), self._up_world)) > math.cos(math.radians(12.0))


@wp.kernel(enable_backward=False)
def _drive_kernel(
    joint_q: wp.array(dtype=float),
    gains: wp.array(dtype=float),  # [press_latch, drive_speed, pinion_dof, button_coord]
    latched: wp.array(dtype=float),
    joint_target_qd: wp.array(dtype=float),
):
    """Latch on the plunger's travel, then hold the pinion's velocity target.

    One thread. The plunger's coordinate is negative going down, so its travel is
    the negated coordinate; the latch only ever rises. Writing the target every
    substep rather than once is what lets a captured graph carry a state change
    at all -- there is no host in the loop to write it later.
    """
    travel = -joint_q[int(gains[3])]
    if travel >= gains[0]:
        latched[0] = 1.0
    joint_target_qd[int(gains[2])] = latched[0] * gains[1]


def _axis_of(quat: np.ndarray) -> np.ndarray:
    """A body's own +z in world coordinates, from an xyzw quaternion."""
    x, y, z, w = (float(v) for v in quat)
    return np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)])


def _forward_of(quat: np.ndarray) -> np.ndarray:
    """A body's own +x in world coordinates -- here, the chain's direction."""
    x, y, z, w = (float(v) for v in quat)
    return np.array([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)])
