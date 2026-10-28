# SPDX-License-Identifier: Apache-2.0
"""Involute spur-gear geometry, and the one rule that makes a board assemblable.

**Peg spacing is derived, never measured.** Two external involute gears of the
same module mesh when, and only when, their centres are ``module * (z_a + z_b) /
2`` apart -- so :func:`chain_centres` computes the board's peg positions *from*
the tooth counts, and there is no way to author a board whose gears do not mesh.
Isaac-CaP's ``gear_insertion`` states its three post offsets as the constants
``0.0823685 / 0.0366185 / -0.0391315``; those numbers are correct for its three
Factory gears and carry no argument for why. This module is the argument.

**Why the profile is generated rather than vendored.** Two involute spur gears
already ship here -- ``assets/vab_tool_sort/assets/industrial/spur_gear_{20t,30t}``,
a matched module-5 pair -- and they are the wrong tool three times over: two tooth
counts cannot supply a four-gear chain or a size axis, at 110 and 160 mm outside
diameter they are wider than any gripper in this repo, and
``assets/PROVENANCE.md`` records that their licence "is not established here".
Generated teeth cost eighty lines and buy the whole variation surface.

**The decomposition is authored, not searched.** MuJoCo collides every mesh geom
as its convex hull, so a gear imported as one mesh is a disc: the teeth and the
bore both vanish (measured on the vendored pair, the hull is 1.37-1.48x the mesh
volume). The way out here is ``knife.py``'s rather than CoACD's -- a gear is a
ring of parts each of which is convex *by construction*:

* **one prism per tooth**, from just under the root circle to the tip. An
  external involute flank is convex, so a single tooth needs no decomposition
  at all; :func:`gear_parts` asserts that against the profile it just built,
  rather than trusting the textbook.
* **one prism per hub sector**, bore to root. An annular sector is *not* convex
  -- its inner arc is concave -- so the hull of a sector's points cuts the bore
  chord-wise. That loss is exact and knowable, ``r_bore * cos(pi / sectors)``,
  and :func:`gear_frame` reports it as ``bore_effective`` so the peg clearance is
  computed against what the collider actually is instead of against the drawing.
  This is the ``fingertip hull is a phantom`` failure, forestalled by arithmetic.

Nothing here imports newton or warp: it is arrays in and arrays out, so the
geometry is testable in the fast tier with no GPU and no scene.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np

__all__ = [
    "BACKLASH_FACTOR",
    "BORE_RADIUS",
    "DEDENDUM_FACTOR",
    "FACE",
    "HUB_SECTORS",
    "PRESSURE_ANGLE",
    "ROOT_OVERLAP",
    "chain_centres",
    "gear_frame",
    "gear_parts",
    "gear_volume",
    "hub_sector",
    "involute",
    "mesh_phase",
    "mesh_separation",
    "prism",
    "tooth_profile",
]

PRESSURE_ANGLE = math.radians(20.0)
"""Standard involute pressure angle [rad].

20 degrees, which is what every catalogue spur gear and both vendored meshes
here are cut at. It is not a free parameter: two gears only mesh if they share
a module *and* a pressure angle, so a variant that changed it would need its own
whole catalogue."""

DEDENDUM_FACTOR = 1.25
"""Dedendum in modules. Root radius is ``r_pitch - 1.25 * module``.

The quarter-module over the addendum is the standard bottom clearance -- the gap
a mating tooth's tip runs in. Dropping it to 1.0 would have tip meeting root
exactly, which in a solver with any contact gap at all is a permanent
interference at the one place the mesh is meant to be free."""

BACKLASH_FACTOR = 0.06
"""Circumferential backlash, in modules. 0.15 mm at the 2.5 mm module.

**A textbook tooth has no backlash and therefore cannot be simulated.** A
standard tooth is exactly half a circular pitch thick and a standard space
exactly half wide, so at the correct centre distance the best clocking any
search can find has the flanks touching at precisely zero -- which is what the
first build of this module did, and :func:`mesh_phase` refused every pair it was
handed. Real gears are cut thin to leave a gap; so is this one.

The factor is bounded from below by the solver, not by machining practice.
A contact gap makes every collider effectively that much fatter on every face,
so a mesh needs ``backlash > 2 * contact_gap`` or the teeth are permanently in
contact and the train jams -- the same shape of error as ``box_decant``'s 2 mm
seam against a 1.5 mm blade, where the geometry admitted something the design
said it did not. 0.06 clears the 0.05 mm per-gear contact gap this suite ships
with three times over; typical machining practice is 0.02-0.04 and would not.
``tasks/gear_mesh/rigs/gear_bench.py`` sweeps it against transmitted torque and rattle."""

FACE = 0.016
"""Thickness of every gear in a catalogue [m], and of the pegs cut for them.

**One constant because it is stated in two places otherwise**, and the two are
not free to disagree: a gear's own thickness sets where its centre rides on a
peg, and the board computes that seat height for itself when it lays the pegs
out. They were separate defaults for exactly one change -- thickening the gears
for a better grasp -- and the board went on seating them for the old thickness,
which put the pinion half out of plane with the train and dropped the whole
chain to 0.09 of its ratio while every gear still read as seated.

Face width does not enter the meshing: module and tooth count fix every radius.
What it decides is the grasp -- see ``gear.SpecGear.face``."""

BORE_RADIUS = 0.006
"""Drawn bore radius of every gear in a catalogue [m], for the same reason as
:data:`FACE`: the board sizes its pegs against it."""

HUB_SECTORS = 24
"""Convex sectors in a gear's hub ring, for every size in a catalogue.

**One number, not one per gear**, and that is what makes a catalogue a catalogue.
The hull of an annular sector chords its inner arc, so the bore a peg actually
has to fit is ``r_bore * cos(pi / sectors)`` -- which means a sector count that
tracked the tooth count would give a 24-tooth gear a wider effective bore than a
16-tooth one, and a peg sized for the tightest would sit loose in the others by
tens of microns. Fixed, every gear in a catalogue has the *same* bore and every
gear fits every peg, which is exactly the difficulty the four-gear tier wants:
putting a gear on the wrong peg has to be possible, and wrong.

24 keeps the chorded bore within 0.05 mm of the drawn one at this scale, well
inside the fit clearance."""

ROOT_OVERLAP = 3.0e-4
"""How far a tooth prism reaches below the root circle [m].

0.3 mm. The hub sectors' outer boundary is chorded, so it sits a sagitta *inside*
the root circle; without an overlap the tooth would start above where the hub
ends and leave a notch at the root -- exactly where the mating tooth's tip goes.
Parts on one rigid body may overlap freely, so the fix costs nothing."""


def involute(phi: float) -> float:
    """``tan(phi) - phi``, the involute function of a pressure angle [rad]."""
    return math.tan(phi) - phi


def gear_frame(
    *,
    module: float,
    teeth: int,
    face: float,
    bore_radius: float,
    backlash: float = 0.0,
    hub_sectors: int = 0,
    flank_samples: int = 6,
) -> dict[str, Any]:
    """Every radius and angle a gear is built from, derived once.

    The ``carton_frame`` pattern: the one place this geometry exists, so a
    tooth, a peg, a grasp box and a goal threshold cannot drift apart. Restating
    any of it elsewhere is how a board ends up not meshing.

    Args:
        module: Tooth size [m]. Pitch diameter is ``module * teeth``.
        teeth: Tooth count. Below 17 an ideal rack would undercut the flank; the
            profile here simply runs the involute down to the base circle and a
            radial line below it, which is what a real generated gear does.
        face: Gear thickness along its axis [m].
        bore_radius: Drawn bore radius [m]. What a peg has to clear is
            ``bore_effective``, which is smaller -- see the module docstring.
        backlash: Circumferential slack at the pitch circle [m], taken off the
            tooth thickness. 0 picks ``BACKLASH_FACTOR * module``; a literal 0 is
            not available, because a zero-backlash pair has no clocking that
            clears and :func:`mesh_phase` would refuse it.
        hub_sectors: Convex parts in the hub ring. 0 picks :data:`HUB_SECTORS`,
            which every size in a catalogue shares so that every gear fits every
            peg identically.
        flank_samples: Points per involute flank. Six holds the profile to
            ~10 um on the sizes this suite ships.

    Returns:
        A dict of scalars plus ``tooth`` and ``sector`` profiles as ``[n, 2]``
        arrays in the gear's own frame, both centred on +x.
    """
    if teeth < 8:
        raise ValueError(f"teeth={teeth} is below 8; the involute flank has no room between base and tip")
    if module <= 0.0 or face <= 0.0 or bore_radius <= 0.0:
        raise ValueError(f"module={module}, face={face} and bore_radius={bore_radius} must all be positive")
    sectors = int(hub_sectors) if hub_sectors else HUB_SECTORS
    slack = float(backlash) if backlash > 0.0 else BACKLASH_FACTOR * module

    r_pitch = 0.5 * module * teeth
    r_base = r_pitch * math.cos(PRESSURE_ANGLE)
    r_tip = r_pitch + module
    r_root = r_pitch - DEDENDUM_FACTOR * module
    if bore_radius >= r_root - module:
        raise ValueError(
            f"bore_radius={bore_radius:.4f} leaves under one module of hub inside a root radius of "
            f"{r_root:.4f}; the ring would be thinner than a tooth is tall"
        )

    frame: dict[str, Any] = {
        "module": float(module),
        "teeth": int(teeth),
        "face": float(face),
        "half_face": 0.5 * float(face),
        "r_pitch": float(r_pitch),
        "r_base": float(r_base),
        "r_tip": float(r_tip),
        "r_root": float(r_root),
        "r_bore": float(bore_radius),
        # What the COLLIDER's bore is, against what the drawing says. The hull of
        # an annular sector chords its inner arc, and this is that chord's radius.
        "bore_effective": float(bore_radius * math.cos(math.pi / sectors)),
        "hub_sectors": sectors,
        "flank_samples": int(flank_samples),
        "backlash": float(slack),
        "pitch_angle": float(2.0 * math.pi / teeth),
        "sector_angle": float(2.0 * math.pi / sectors),
        # Half the tooth's angular thickness at the pitch circle [rad]. Half a
        # pitch is the standard tooth; the second term is the backlash, cut off
        # the thickness the way a hobber does it rather than by opening the
        # centre distance, which would stop the pitch circles being the pitch
        # circles and put chain_centres out of agreement with itself.
        "half_thickness_angle": float(math.pi / (2.0 * teeth) - 0.5 * slack / r_pitch),
    }
    frame["tooth"] = tooth_profile(frame)
    frame["sector"] = hub_sector(frame)
    return frame


def _half_angle(frame: dict[str, Any], radius: float) -> float:
    """Half the tooth's angular thickness at ``radius`` [rad].

    ``psi(r) = pi/(2z) + inv(alpha) - inv(phi(r))`` with ``phi(r) =
    acos(r_base / r)``. At the pitch circle ``phi`` is the pressure angle and the
    two involutes cancel, leaving half a tooth; the tooth narrows above it and
    widens below.
    """
    r_base = frame["r_base"]
    if radius <= r_base:
        radius = r_base
    phi = math.acos(min(1.0, r_base / radius))
    return frame["half_thickness_angle"] + involute(PRESSURE_ANGLE) - involute(phi)


def tooth_profile(frame: dict[str, Any]) -> np.ndarray:
    """One tooth as a closed ``[n, 2]`` polygon, centred on +x, wound CCW.

    Root outward along the -psi flank, across the tip, back down the +psi flank.
    The bottom two points sit :data:`ROOT_OVERLAP` below the root circle so the
    tooth meets the hub with material to spare rather than exactly.
    """
    r_root, r_tip, r_base = frame["r_root"], frame["r_tip"], frame["r_base"]
    # Below the base circle there is no involute; a real generator leaves a
    # near-radial trochoid there, and a radial line is its honest simplification.
    radii = np.linspace(max(r_root, r_base), r_tip, int(frame["flank_samples"]))
    angles = np.array([_half_angle(frame, float(r)) for r in radii])
    if r_root < r_base:
        radii = np.concatenate([[r_root], radii])
        angles = np.concatenate([[angles[0]], angles])
    foot = r_root - ROOT_OVERLAP
    radii = np.concatenate([[foot], radii])
    angles = np.concatenate([[angles[0]], angles])

    rising = np.stack([radii * np.cos(-angles), radii * np.sin(-angles)], axis=1)
    falling = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)[::-1]
    return np.vstack([rising, falling]).astype(np.float64)


def hub_sector(frame: dict[str, Any]) -> np.ndarray:
    """One hub sector as a closed ``[n, 2]`` polygon, centred on +x, wound CCW.

    Bore to root over one sector angle. The inner edge is stated as the two
    corner points and nothing between them, because a convex hull would discard
    anything between them anyway -- see ``bore_effective``. The outer edge is
    sampled along the root arc, where the hull's chord loses material rather than
    gaining it and finer sampling therefore helps.
    """
    half = 0.5 * frame["sector_angle"]
    r_in, r_out = frame["r_bore"], frame["r_root"]
    outer_angles = np.linspace(-half, half, 5)
    inner = np.array([[r_in * math.cos(half), r_in * math.sin(half)], [r_in * math.cos(-half), r_in * math.sin(-half)]])
    outer = np.stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)], axis=1)
    return np.vstack([outer, inner]).astype(np.float64)


def _is_convex(points: np.ndarray, tol: float = 1.0e-9) -> bool:
    """Whether a closed 2D polygon turns the same way at every vertex."""
    edges = np.roll(points, -1, axis=0) - points
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    return bool(np.all(cross >= -tol) or np.all(cross <= tol))


def prism(points: np.ndarray, half_height: float, chamfer: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a convex 2D polygon into a closed triangle mesh along +-z.

    ``(vertices, indices)`` in the form ``newton.Mesh`` wants. The caps are fans
    about the polygon's centroid, which is inside it because the polygon is
    convex -- the same construction ``polish``'s ``_drum_mesh`` uses for its
    sixteen-gon drum, generalised off the regular case.

    ``chamfer`` bevels both end faces at 45 degrees, leaving the part widest at
    mid-face. On a tooth that is the **lead-in**: a gear lowered onto a peg beside
    an already-seated neighbour has to find the tooth space, and it finds it by
    turning on the peg -- but only if the first thing that meets is a slope
    rather than two square tip corners. The inset is a scaling about the
    centroid, not a true offset, which keeps the result provably convex and is
    indistinguishable from one at a chamfer this small.
    """
    if not _is_convex(points):
        raise ValueError("prism() needs a convex polygon; MuJoCo would hull a concave one and fill it in")
    n = len(points)
    centre = points.mean(axis=0)
    rings = [(points, half_height), (points, -half_height)]
    if chamfer > 0.0:
        reach = float(np.max(np.linalg.norm(points - centre, axis=1)))
        if chamfer >= reach or chamfer >= half_height:
            raise ValueError(f"chamfer={chamfer} does not fit a part of reach {reach} and half-height {half_height}")
        capped = centre + (points - centre) * (1.0 - chamfer / reach)
        rings = [
            (capped, half_height),
            (points, half_height - chamfer),
            (points, chamfer - half_height),
            (capped, -half_height),
        ]
    verts = np.vstack([np.hstack([ring, np.full((n, 1), z)]) for ring, z in rings])
    verts = np.vstack([verts, [[*rings[0][0].mean(axis=0), rings[0][1]]], [[*rings[-1][0].mean(axis=0), rings[-1][1]]]])
    ct, cb = len(verts) - 2, len(verts) - 1
    idx: list[int] = []
    for i in range(n):
        j = (i + 1) % n
        idx += [i, j, ct]
        base = n * (len(rings) - 1)
        idx += [base + j, base + i, cb]
        for r in range(len(rings) - 1):
            lo, hi = n * r, n * (r + 1)
            idx += [lo + i, hi + i, lo + j, lo + j, hi + i, hi + j]
    return verts.astype(np.float32), np.asarray(idx, dtype=np.int32)


def _separation(a: np.ndarray, b: np.ndarray) -> float:
    """Signed gap between two convex CCW polygons [m]; negative means overlap.

    Separating-axis over both edge-normal sets. The sign is exact, which is all
    the clocking search below needs, and the magnitude is the true distance
    wherever the closest features are a vertex and an edge -- which, on two tooth
    flanks, they are.
    """
    best = -math.inf
    for near, far in ((a, b), (b, a)):
        edges = np.roll(near, -1, axis=0) - near
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        gaps = (far @ normals.T).min(axis=0) - (near @ normals.T).max(axis=0)
        best = max(best, float(gaps.max()))
    return best


def _teeth_facing(frame: dict[str, Any], yaw: float, centre: np.ndarray, toward: np.ndarray, count: int = 3):
    """The ``count`` tooth polygons of a placed gear nearest the given bearing."""
    bearing = math.atan2(toward[1] - centre[1], toward[0] - centre[0])
    pitch = frame["pitch_angle"]
    nearest = round((bearing - yaw) / pitch)
    return [_rotate(frame["tooth"], yaw + (nearest + k) * pitch) + centre for k in range(-(count // 2), count // 2 + 1)]


def mesh_separation(
    near: dict[str, Any],
    far: dict[str, Any],
    centre_distance: float,
    yaw_near: float,
    yaw_far: float,
) -> float:
    """Closest approach between two meshing gears' teeth [m]; negative is a clash.

    The two gears sit on the x axis, ``near`` at the origin. Only the three teeth
    each presents to the other can reach, so this is nine polygon pairs rather
    than ``z_a * z_b``.
    """
    a_at, b_at = np.zeros(2), np.array([centre_distance, 0.0])
    a_teeth = _teeth_facing(near, yaw_near, a_at, b_at)
    b_teeth = _teeth_facing(far, yaw_far, b_at, a_at)
    return min(_separation(a, b) for a in a_teeth for b in b_teeth)


def mesh_phase(
    near: dict[str, Any],
    far: dict[str, Any],
    centre_distance: float,
    yaw_near: float,
    *,
    samples: int = 721,
) -> tuple[float, float]:
    """``(yaw_far, backlash)`` -- how to clock the far gear, and by how much it may.

    **Assembly is a computed state, not a placed one.** Two meshing gears are not
    free to be at any relative angle: the far gear's teeth have to sit in the
    near gear's spaces, and dropping an assembled board in at arbitrary yaws
    starts every episode with two teeth inside each other. Rather than derive the
    phase in closed form -- which is a page of algebra whose parity cases differ
    for odd and even tooth counts -- sweep one pitch and take the angle of
    greatest clearance. The width of the window where the teeth are apart at all
    is the **backlash**, returned because it is the number the coupling-law
    fallback's deadband is set from and the number a tooth-contact drive is
    judged against.

    Args:
        near: The already-placed gear's frame.
        far: The frame of the gear being clocked, at ``+centre_distance`` in x.
        centre_distance: From :func:`chain_centres`; anything else does not mesh.
        yaw_near: The placed gear's yaw [rad].
        samples: Sweep resolution over one far-gear pitch.

    Returns:
        The far gear's yaw at maximum clearance [rad], and the backlash window
        [rad] -- the arc over which it may turn before its teeth touch.
    """
    engagement = near["r_tip"] + far["r_tip"] - centre_distance
    if engagement < near["module"]:
        raise ValueError(
            f"a {near['teeth']}- and a {far['teeth']}-tooth gear {centre_distance * 1000:.2f} mm apart "
            f"overlap by only {engagement * 1000:.2f} mm of tip; two modules is the meshing depth and "
            "under one is a pair that would disengage rather than drive"
        )
    pitch = far["pitch_angle"]
    sweep = np.linspace(0.0, pitch, samples, endpoint=False)
    gaps = np.array([mesh_separation(near, far, centre_distance, yaw_near, float(y)) for y in sweep])
    if not np.any(gaps > 0.0):
        raise ValueError(
            f"no clocking of a {far['teeth']}-tooth gear clears a {near['teeth']}-tooth one at "
            f"{centre_distance * 1000:.2f} mm; the centre distance is not this pair's"
        )
    best = int(np.argmax(gaps))
    return float(sweep[best]), float(np.count_nonzero(gaps > 0.0)) * pitch / samples


def gear_parts(frame: dict[str, Any], tip_chamfer: float = 0.0) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    """Every convex part of one gear: ``(label, vertices, indices)``.

    ``teeth`` tooth prisms and ``hub_sectors`` hub prisms, each rotated into
    place about z. :func:`prism` refuses a concave profile, so this is where the
    "an external involute tooth is convex" claim is checked against the profile
    actually built rather than taken from a textbook.

    ``tip_chamfer`` bevels the teeth only. The hub never meets anything it has to
    find its way into -- it slides down a peg -- so a chamfer there would open
    the bore for nothing.
    """
    parts: list[tuple[str, np.ndarray, np.ndarray]] = []
    half_face = frame["half_face"]
    for index in range(int(frame["teeth"])):
        yaw = index * frame["pitch_angle"]
        parts.append((f"tooth_{index:02d}", *prism(_rotate(frame["tooth"], yaw), half_face, tip_chamfer)))
    for index in range(int(frame["hub_sectors"])):
        yaw = index * frame["sector_angle"]
        parts.append((f"hub_{index:02d}", *prism(_rotate(frame["sector"], yaw), half_face)))
    return tuple(parts)


def _polygon_area(points: np.ndarray) -> float:
    """Unsigned area of a closed 2D polygon [m^2], by the shoelace formula."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def gear_volume(frame: dict[str, Any]) -> float:
    """Solid volume of one gear [m^3], summed over the parts it is built from.

    The parts, not a disc: a toothed ring is meaningfully lighter than the
    cylinder that bounds it, and a gear's mass is what its inertia -- and so its
    spin-up, and so whether the drive's ``effort_limit`` is an honest number --
    is computed from. Sound as long as the parts do not overlap, which they do
    only by :data:`ROOT_OVERLAP` at the root; that seam is counted twice and is
    worth about a tenth of a percent.
    """
    tooth = _polygon_area(frame["tooth"]) * frame["teeth"]
    hub = _polygon_area(frame["sector"]) * frame["hub_sectors"]
    return float((tooth + hub) * frame["face"])


def _rotate(points: np.ndarray, yaw: float) -> np.ndarray:
    """A 2D point set turned about the origin by ``yaw`` [rad]."""
    c, s = math.cos(yaw), math.sin(yaw)
    return points @ np.array([[c, s], [-s, c]])


def chain_centres(module: float, counts: tuple[int, ...]) -> tuple[float, ...]:
    """Where each gear of a meshing chain sits along its line [m], first at 0.

    ``a_i = module * (z_i + z_{i+1}) / 2`` -- the standard centre distance for
    two external gears of one module, and the whole reason a board built from
    this function meshes. The gears may be any sizes in any order; what they may
    not be is a different module from one another.
    """
    if len(counts) < 2:
        raise ValueError(f"a chain needs at least two gears, got {len(counts)}")
    centres = [0.0]
    for near, far in pairwise(counts):
        centres.append(centres[-1] + 0.5 * module * (near + far))
    return tuple(centres)
