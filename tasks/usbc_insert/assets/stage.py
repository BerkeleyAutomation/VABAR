# SPDX-License-Identifier: Apache-2.0
"""USB-C plug / receptacle geometry: measured, asserted, and placed.

A port of ``newton_tasks/sim/usbc_asset.py``, reading the vendored STLs under
this task's own ``assets/``. The two meshes are authored in metres in a shared frame:

* **Plug** -- a rounded-rectangular body 11.0 x 5.0 mm spanning
  ``z in [-17.0, 0] mm`` with the mating **tab** 8.25 x 2.40 mm on top,
  ``z in [0, +6.65] mm``. ``+z`` is the insertion direction.
* **Port** -- a 75 x 20 x 31.5 mm block whose top face (``z = +6.5 mm``)
  carries a cavity 8.4 x 2.6 mm running 6.5 mm down to a floor at ``z = 0``.

Nothing is trusted from these comments: every dimension is *measured* from the
STL at load time and asserted against the expected topology, so a re-exported
mesh fails loudly instead of silently shifting the mate point.

Two things differ from a naive re-read of the reference, both deliberate:

* The module-level ``REF_*`` constants restate the measured values at the
  task's x2 scale so ``suite.py`` can place statics and cameras without
  loading a mesh (registration must survive a box with no trimesh). The
  loaders assert the measurement matches the restatement to a micrometre.
* ``CHAMFER`` reproduces the reference MJC sim's **effective** lead-in. That
  sim passed ``chamfer = args.chamfer * scale`` into a loader that treats
  chamfer as pre-scale and multiplies the vertices by ``scale`` afterwards, so
  its "1 mm" argument was a **4 mm world chamfer**. The kamino sim's was 2 mm.
  This port states the pre-scale value that lands at the MJC sim's actual
  geometry, because that is the geometry its published behaviour (grasp,
  partial seat, jam) was measured on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robosimstudio.core.math import mat_to_quat
from robosimstudio.paths import resolve_asset
from robosimstudio.scene.builtin.benches import BENCH_075
from robosimstudio.specs import AssetSpec

__all__ = [
    "BULKHEAD_QUAT",
    "CELL",
    "CHAMFER",
    "INSERT_AXIS",
    "PLUG_QUAT",
    "PORT_QUAT",
    "REF",
    "SCALE",
    "Assembly",
    "PlugGeom",
    "PortGeom",
    "bulkhead_barrel_grasp",
    "bulkhead_spawn",
    "clearance_mm",
    "load_plug",
    "load_port",
    "make_assembly",
]

SCALE = 2.0
"""Uniform scale both sims run at. At x1 the 0.1 mm-a-side fit is below what
the contact pipeline can resolve at any sane gap."""

CHAMFER = 0.002
"""Pre-scale chamfer [m] -> 4 mm world at x2: the reference MJC sim's
effective lead-in (see the module docstring for the double-scale it inherits)."""

_CAVITY_SPLIT_R = 0.020
"""A vertex closer to the part axis than this belongs to the cavity."""

# ---------------------------------------------------------------- reference
class _Ref:
    """The measured geometry at :data:`SCALE`, restated for spec construction.

    Asserted against the live measurement by :func:`load_plug` /
    :func:`load_port`; a mesh edit that moves any of these fails the build.
    """

    plug_half_x = 0.011
    plug_half_y = 0.005
    plug_tab_half_x = 0.00825
    plug_tab_half_y = 0.0024
    plug_body_z0 = -0.034
    plug_tab_tip_z = 0.0133
    port_half_x = 0.075
    port_half_y = 0.020
    port_cav_half_x = 0.0084
    port_cav_half_y = 0.0026
    port_block_z0 = -0.050
    port_mouth_z = 0.013
    bore_depth = 0.013


REF = _Ref()


@dataclass(frozen=True)
class PlugGeom:
    """Measured plug geometry, in metres, in the STL's own frame."""

    half_x: float
    half_y: float
    tab_half_x: float
    tab_half_y: float
    body_z0: float
    shoulder_z: float
    tab_tip_z: float
    com_z: float

    @property
    def length(self) -> float:
        return self.tab_tip_z - self.body_z0


@dataclass(frozen=True)
class PortGeom:
    """Measured port geometry, in metres, in the STL's own frame."""

    half_x: float
    half_y: float
    cav_half_x: float
    cav_half_y: float
    block_z0: float
    floor_z: float
    mouth_z: float

    @property
    def depth(self) -> float:
        return self.mouth_z - self.floor_z


_HERE = Path(__file__).resolve().parent


def _load_stl(name: str):
    """One of this task's own meshes, from beside this file.

    Beside rather than through an asset root: the geometry ships with the task
    that measured it, so the directory this module is in *is* where it lives.
    """
    import trimesh

    path = resolve_asset(AssetSpec(path=str(_HERE / name)))
    tm = trimesh.load(str(path))
    if not hasattr(tm, "vertices"):
        tm = tm.dump(concatenate=True)
    return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64)


def _z_levels(V, tol=1.0e-6):
    return np.unique(np.round(V[:, 2] / tol).astype(np.int64)) * tol


def load_plug(scale: float = SCALE):
    """``(newton.Mesh, PlugGeom)`` for the plug, both scaled by ``scale``."""
    import newton  # noqa: PLC0415
    import trimesh

    V, F = _load_stl("USB_C_Plug.stl")
    zs = _z_levels(V)
    if len(zs) != 3:
        raise RuntimeError(f"USB_C_Plug.stl: expected 3 z-levels (rear/shoulder/tip), got {zs}")
    z_rear, z_shoulder, z_tip = float(zs[0]), float(zs[1]), float(zs[2])
    rear = V[np.abs(V[:, 2] - z_rear) < 1e-6]
    tip = V[np.abs(V[:, 2] - z_tip) < 1e-6]
    # The grasp station wants the CoM, not the body midpoint: gripping off the
    # CoM puts a droop moment on the friction patch. The tab pulls the CoM
    # ~1.5 mm forward of the body centre.
    com_z = float(trimesh.Trimesh(V, F, process=False).center_mass[2])
    geom = PlugGeom(
        half_x=float(np.abs(rear[:, 0]).max()) * scale,
        half_y=float(np.abs(rear[:, 1]).max()) * scale,
        tab_half_x=float(np.abs(tip[:, 0]).max()) * scale,
        tab_half_y=float(np.abs(tip[:, 1]).max()) * scale,
        body_z0=z_rear * scale,
        shoulder_z=z_shoulder * scale,
        tab_tip_z=z_tip * scale,
        com_z=com_z * scale,
    )
    if scale == SCALE:
        for field_name, ref in (
            ("half_x", REF.plug_half_x),
            ("half_y", REF.plug_half_y),
            ("tab_half_x", REF.plug_tab_half_x),
            ("tab_half_y", REF.plug_tab_half_y),
            ("body_z0", REF.plug_body_z0),
            ("tab_tip_z", REF.plug_tab_tip_z),
        ):
            measured = getattr(geom, field_name)
            if abs(measured - ref) > 1.0e-6:
                raise RuntimeError(f"USB_C_Plug.stl: {field_name} measured {measured}, reference {ref}")
    mesh = newton.Mesh(
        np.ascontiguousarray(V * scale, dtype=np.float32), np.ascontiguousarray(F.reshape(-1), dtype=np.int32)
    )
    return mesh, geom


def load_port(scale: float = SCALE, cavity_scale: float = 1.0, chamfer: float = CHAMFER):
    """``(newton.Mesh, PortGeom)`` for the receptacle.

    ``cavity_scale`` widens only the cavity cross-section about the part axis;
    ``chamfer`` [m, **pre-scale**] adds a 45-degree lead-in at the mouth: the
    mouth loop drops to ``mouth_z - chamfer`` and an outward-offset loop takes
    its place in the top face. The nominal fit below the chamfer is unchanged,
    and the returned :class:`PortGeom` reports the nominal (un-chamfered)
    values.
    """
    import newton  # noqa: PLC0415
    import trimesh

    V, F = _load_stl("USB_C_Port.stl")
    zs = _z_levels(V)
    if len(zs) != 3:
        raise RuntimeError(f"USB_C_Port.stl: expected 3 z-levels (bottom/floor/mouth), got {zs}")
    z_bot, z_floor, z_mouth = float(zs[0]), float(zs[1]), float(zs[2])

    r = np.hypot(V[:, 0], V[:, 1])
    is_cav = r < _CAVITY_SPLIT_R
    on_floor = np.abs(V[:, 2] - z_floor) < 1e-6
    on_mouth = np.abs(V[:, 2] - z_mouth) < 1e-6
    cav_floor = np.where(is_cav & on_floor)[0]
    cav_mouth = np.where(is_cav & on_mouth)[0]
    if len(cav_floor) != len(cav_mouth) or len(cav_floor) == 0:
        raise RuntimeError(
            f"USB_C_Port.stl: cavity loops mismatched ({len(cav_floor)} floor verts, {len(cav_mouth)} mouth verts)"
        )

    bottom = V[np.abs(V[:, 2] - z_bot) < 1e-6]
    geom = PortGeom(
        half_x=float(np.abs(bottom[:, 0]).max()) * scale,
        half_y=float(np.abs(bottom[:, 1]).max()) * scale,
        cav_half_x=float(np.abs(V[cav_floor, 0]).max()) * cavity_scale * scale,
        cav_half_y=float(np.abs(V[cav_floor, 1]).max()) * cavity_scale * scale,
        block_z0=z_bot * scale,
        floor_z=z_floor * scale,
        mouth_z=z_mouth * scale,
    )
    if scale == SCALE and cavity_scale == 1.0:
        for field_name, ref in (
            ("half_x", REF.port_half_x),
            ("half_y", REF.port_half_y),
            ("cav_half_x", REF.port_cav_half_x),
            ("cav_half_y", REF.port_cav_half_y),
            ("block_z0", REF.port_block_z0),
            ("mouth_z", REF.port_mouth_z),
        ):
            measured = getattr(geom, field_name)
            if abs(measured - ref) > 1.0e-6:
                raise RuntimeError(f"USB_C_Port.stl: {field_name} measured {measured}, reference {ref}")

    V = V.copy()
    V[np.where(is_cav)[0], :2] *= cavity_scale

    if chamfer > 0.0:
        # Push the mouth loop down and add an outward-offset loop in the top
        # face; the band between them is the lead-in. Top-face triangles are
        # re-pointed at the new loop so nothing is re-triangulated.
        new_idx = {}
        extra = []
        for vi in cav_mouth:
            p = V[vi].copy()
            d = np.array([p[0], p[1]], dtype=np.float64)
            n = d / max(np.linalg.norm(d), 1e-12)
            extra.append([p[0] + n[0] * chamfer, p[1] + n[1] * chamfer, z_mouth])
            new_idx[int(vi)] = len(V) + len(extra) - 1
            V[vi, 2] = z_mouth - chamfer
        V = np.vstack([V, np.asarray(extra, dtype=np.float64)])
        mouth_set = {int(i) for i in cav_mouth}
        F = F.copy()
        top_face = np.array(
            [
                all((int(i) in mouth_set) or (abs(V[i, 2] - z_mouth) < 1e-6 and int(i) not in mouth_set) for i in f)
                for f in F
            ]
        )
        for fi in np.where(top_face)[0]:
            for k in range(3):
                vi = int(F[fi, k])
                if vi in new_idx:
                    F[fi, k] = new_idx[vi]
        loop = _order_loop(V[cav_mouth, :2])
        band = []
        m = len(cav_mouth)
        for a in range(m):
            i0 = int(cav_mouth[loop[a]])
            i1 = int(cav_mouth[loop[(a + 1) % m]])
            band.append([i0, i1, new_idx[i1]])
            band.append([i0, new_idx[i1], new_idx[i0]])
        F = np.vstack([F, np.asarray(band, dtype=np.int64)])
        # The band's winding follows an angular sort, consistent only up to the
        # loop's handedness -- normalise on the watertight mesh.
        tm = trimesh.Trimesh(V, F, process=False)
        trimesh.repair.fix_winding(tm)
        trimesh.repair.fix_inversion(tm)
        if not tm.is_winding_consistent:
            raise RuntimeError("chamfered port mesh has inconsistent winding")
        V = np.asarray(tm.vertices, dtype=np.float64)
        F = np.asarray(tm.faces, dtype=np.int64)

    mesh = newton.Mesh(
        np.ascontiguousarray(V * scale, dtype=np.float32), np.ascontiguousarray(F.reshape(-1), dtype=np.int32)
    )
    return mesh, geom


def _order_loop(P2):
    """Angular ordering of a convex-ish planar loop about its centroid."""
    c = P2.mean(axis=0)
    return np.argsort(np.arctan2(P2[:, 1] - c[1], P2[:, 0] - c[0]))


def clearance_mm(plug: PlugGeom, port: PortGeom) -> tuple[float, float]:
    """Per-side tab/cavity clearance in mm (wide axis, thin axis)."""
    return (port.cav_half_x - plug.tab_half_x) * 1e3, (port.cav_half_y - plug.tab_half_y) * 1e3


# ------------------------------------------------------------- world placement
# A horizontal +y insertion: the plug rests on a bench at bore height, is
# grasped top-down, and is pushed straight along +y into a port facing -y.
# Both parts are rolled so the connector's WIDE axis is VERTICAL, which
# presents the plug's 5 mm dimension to a top-down jaw pair and puts the two
# big flat faces against the blades -- the contact patch's length along the
# insertion axis is what resists yaw-on-close.
_R_PLUG = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
_R_PORT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)

PLUG_QUAT = tuple(float(v) for v in mat_to_quat(_R_PLUG))
"""xyzw: plug local +x -> world +z, +y -> +x, +z (insertion) -> +y."""
PORT_QUAT = tuple(float(v) for v in mat_to_quat(_R_PORT))

_R_BULKHEAD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
"""Bulkhead mesh frame -> world. Barrel axis +x -> +y (interior behind the
mouth, the block port's own arrangement), slot width y -> +z (wide axis
vertical, the roll the plug is presented at), slot height z -> +x. A cyclic
permutation, det +1."""

BULKHEAD_QUAT = tuple(float(v) for v in mat_to_quat(_R_BULKHEAD))
"""The same rotation an ``ObjectSpec`` states, so a scene that imports the raw
mesh lands where :func:`load_bulkhead` puts the measured one."""


def bulkhead_spawn(scale: float = SCALE) -> tuple[float, float, float]:
    """Where the bulkhead's mesh origin sits in the world [m].

    The translation half of the placement, derived from the reference assembly
    rather than stated: it puts the slot axis exactly on the block port's bore
    line with the two mouth faces coplanar, which is what lets this variant use
    the block variant's thresholds unchanged.

    Paired with :data:`BULKHEAD_QUAT`, this is the whole transform -- so a task
    can spawn the raw mesh as an ordinary free body and get the receptacle the
    measured loader would have built, without either copy carrying a number the
    other could contradict.
    """
    asm = reference_assembly()
    return (
        float(asm.plug_mated[0]),
        float(asm.mouth_y - BULK.mouth_x * scale),
        float(asm.plug_mated[2]),
    )


"""xyzw: port local +x -> world +z, +y -> -x, +z (mouth) -> -y."""
INSERT_AXIS = np.array([0.0, 1.0, 0.0])
"""World insertion direction."""


@dataclass(frozen=True)
class Assembly:
    """Where every part sits in the world, derived from measured geometry."""

    plug: PlugGeom
    port: PortGeom
    pad_top: float
    plug_spawn: tuple[float, float, float]
    plug_mated: tuple[float, float, float]
    port_origin: tuple[float, float, float]
    port_base_z: float

    @property
    def insert_dist(self) -> float:
        return float(self.plug_mated[1] - self.plug_spawn[1])

    @property
    def mouth_y(self) -> float:
        """World y of the port's mouth plane (where the tab first enters)."""
        return float(self.port_origin[1] - self.port.mouth_z)

    @property
    def free_approach(self) -> float:
        """Tab-tip-to-mouth clearance at spawn [m]."""
        return self.mouth_y - float(self.plug_spawn[1] + self.plug.tab_tip_z)


def make_assembly(
    plug: PlugGeom, port: PortGeom, anchor_xy, pad_top: float, insert_dist: float, lift: float = 0.0
) -> Assembly:
    """Place the parts for a +y insertion with the plug starting at ``anchor_xy``.

    The plug rests on the bench and the bore axis sits ``lift`` above that, so
    the robot picks the part, raises it clear, and only then translates.
    ``lift`` is not cosmetic: carried along the bench (lift=0) the part's
    underside keeps dragging while the grip acts above its CoM, and the couple
    pitches it out of the jaws -- the reference measured 10 mm of climb and 25
    degrees during the stroke.
    """
    ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
    plug_z = pad_top + plug.half_x  # local +x -> world +z: lower face is half_x below origin
    bore_z = plug_z + lift
    spawn = (ax, ay, plug_z)
    mated = (ax, ay + float(insert_dist), bore_z)
    # Mated = tab tip on the cavity floor, which sits at the port origin's y.
    port_origin = (ax, mated[1] + plug.tab_tip_z, bore_z)
    return Assembly(
        plug=plug,
        port=port,
        pad_top=float(pad_top),
        plug_spawn=spawn,
        plug_mated=mated,
        port_origin=port_origin,
        port_base_z=bore_z - port.half_x,
    )


def reference_assembly() -> Assembly:
    """The task's placement at the reference numbers, mesh-free.

    What ``suite.py`` positions statics and cameras from: the same
    :func:`make_assembly` arithmetic over the asserted ``REF`` constants, so
    the spec is buildable without trimesh and cannot drift from the build.
    """
    plug = PlugGeom(
        half_x=REF.plug_half_x,
        half_y=REF.plug_half_y,
        tab_half_x=REF.plug_tab_half_x,
        tab_half_y=REF.plug_tab_half_y,
        body_z0=REF.plug_body_z0,
        shoulder_z=0.0,
        tab_tip_z=REF.plug_tab_tip_z,
        com_z=-0.01408,
    )
    port = PortGeom(
        half_x=REF.port_half_x,
        half_y=REF.port_half_y,
        cav_half_x=REF.port_cav_half_x,
        cav_half_y=REF.port_cav_half_y,
        block_z0=REF.port_block_z0,
        floor_z=0.0,
        mouth_z=REF.port_mouth_z,
    )
    return make_assembly(plug, port, ANCHOR_XY, PAD_TOP, INSERT_DIST, lift=LIFT)


# The reference MJC sim's own defaults, restated.
CELL = BENCH_075
"""The cell every world number in this module was authored in.

Named, not assumed: :data:`ANCHOR_XY`, :data:`PAD_TOP` and everything the
assembly derives from them are world coordinates in a sim whose table top was
0.750, and this is what the placement factories put them through
(:func:`~robosimstudio.scene.frames.point_from_surface`) to get offsets that
survive a different table."""
ANCHOR_XY = (0.44, -0.07)
"""Where the plug starts on the bench [m]."""
BENCH_H = 0.05
"""Bench height above the table [m]."""
PAD_TOP = 0.75 + BENCH_H
"""World z of the bench top, in the reference cell (0.750 surface + BENCH_H)."""
INSERT_DIST = 0.080
"""Grasp-to-mate travel along +y [m]."""
LIFT = 0.0124
"""Bore-axis height above the resting plug [m]."""
GRASP_DY = -0.0199
"""Grasp station offset along -y from the spawn: the jaws close behind the
CoM, clear of the tab."""


# --------------------------------------------------------------- the bulkhead
#
# A second receptacle, and the reason it is here rather than in a file of its
# own: it is measured off a mesh with the same loader, at the same SCALE, and
# placed against the same reference_assembly(). Splitting it would mean two
# copies of that arithmetic, which is the failure this module exists to avoid.


class _BulkheadRef:
    """Measured dimensions of the original Bulkhead.stl at scale 1.

    The active asset uses the delivered Bulkhead_Fitted.stl.
    Its inner sleeve is narrower; these raw dimensions still describe the
    barrel, cradle interface, mouth plane, and boot. load_bulkhead measures
    the fitted sleeve separately, against bulkhead_geometry.py's dimensions.
    """

    mouth_x = -0.00225
    boot_x = 0.03075
    slot_half_w = 0.0049
    slot_inner_half_w = 0.00455
    """The wide wall the tab actually touches. The slot wall is double-walled
    -- two nested profile loops at every station -- and ``slot_half_w`` is the
    outer skin, which is all a section's extent can read (so it is what
    :func:`load_bulkhead` asserts). A ray from the bore axis hits this one:
    the true wide clearance is 0.85 mm a side at x2, not the 1.55 the outer
    skin suggests, and every narrowing experiment in this class's notes was
    calibrated against the wrong wall."""
    slot_half_h = 0.0020
    slot_deep_x = 0.01155
    barrel_r = 0.0105
    band_x0 = 0.013
    band_x1 = 0.021
    # The bimanual variant's fixture and mass facts, measured off the placed
    # mesh (tools run once, restated here so specs build mesh-free):
    flange_r = 0.015
    """Mounting-flange radius: the part's lowest point hangs this far under
    the axis."""
    flange_x0 = 0.0025
    flange_x1 = 0.0055
    """The flange disc's rim band along the axis: where the front step
    supports it."""
    step_x1 = 0.017
    """Rear step and walls end here: past it the barrel steps down to r10 and
    a support stated at r10.5 would float over it."""
    mass = 0.090
    """The reference bulkhead sim's ground-truth mass for the metal-shelled
    part: the STL is a non-watertight shell, so a density-derived mass is
    meaningless and the body carries this explicitly."""
    com_local = (0.0, 0.0, -0.0065)
    """Shell-area centroid in the port-local frame: on the axis, 6.5 mm behind
    the origin (the nose -- ring, flange, hex -- is the heavy end)."""


BULK = _BulkheadRef()


@dataclass(frozen=True)
class BulkheadGeom:
    """Measured bulkhead slot geometry, in metres, at the loaded scale."""

    cav_half_wide: float
    cav_half_thin: float
    depth: float
    barrel_radius: float


def load_bulkhead(scale: float = SCALE):
    """``(newton.Mesh, BulkheadGeom)`` for the panel-mount bulkhead receptacle.

    The mesh comes back **in the block port's local frame**: rotated so the
    barrel axis takes the plug's ``+y`` insertion (mouth toward ``-y``, slot
    wide-axis vertical -- the roll the block port presents), translated so the
    slot axis lies exactly on the block port's bore line with the mouth face on
    the block's mouth plane, then expressed against ``(port_origin,
    PORT_QUAT)``. Placing it with the block port's own static transform
    therefore reproduces the block's mate line, and the evaluator, routine and
    graph -- which read the bore from ``reference_assembly()``, never from the
    mesh -- carry over without modification.

    Slot dimensions are measured from rays hitting the fitted inner walls:
    the slot walls run from mouth to pocket with no intermediate vertices, and
    a band-based measurement reads the mouth ring's counterbore instead (the
    ``newton_tasks`` bulkhead sim's own relearned lesson).
    """
    import newton  # noqa: PLC0415
    import trimesh

    from .bulkhead_geometry import FIT_HALF_THIN_M, FIT_HALF_WIDE_M, FIT_SCALE  # noqa: PLC0415

    V, F = _load_stl("Bulkhead_Fitted.stl")
    fitted_w, fitted_h = FIT_HALF_WIDE_M / FIT_SCALE, FIT_HALF_THIN_M / FIT_SCALE
    tm = trimesh.Trimesh(V, F, process=False)

    if abs(float(V[:, 0].min()) - BULK.mouth_x) > 1e-4 or abs(float(V[:, 0].max()) - BULK.boot_x) > 1e-4:
        raise RuntimeError(
            f"Bulkhead.stl: axis extent [{V[:, 0].min():.5f}, {V[:, 0].max():.5f}] does not match "
            f"the reference [{BULK.mouth_x}, {BULK.boot_x}]"
        )

    # The delivered mesh tapers along its existing sleeve triangles. Its
    # seated section reaches the nominal fit immediately before x=0.01145;
    # the upstream generator does not split those triangles at LEAD_LEN.
    # Check the actual taper without changing the rendered/contact geometry.
    previous = None
    for x0 in (0.002, 0.008, 0.011449):
        origins = np.array([[x0, 0., 0.]] * 4)
        directions = np.array([[0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])
        hits, rays, _ = tm.ray.intersects_location(origins, directions, multiple_hits=True)
        measured = []
        for index, expected in enumerate((fitted_w, fitted_w, fitted_h, fitted_h)):
            distances = np.linalg.norm(hits[rays == index] - origins[index], axis=1)
            if not len(distances):
                raise RuntimeError(f"Bulkhead_Fitted.stl: missing inner wall at x={x0}, ray={index}")
            distance = float(distances.min())
            if distance < expected - 1.e-6 or (x0 == 0.011449 and abs(distance - expected) > 1.e-6):
                raise RuntimeError(f"Bulkhead_Fitted.stl: inner wall differs from fit at x={x0}, ray={index}")
            if previous is not None and distance > previous[index] + 1.e-6:
                raise RuntimeError(f"Bulkhead_Fitted.stl: sleeve widens inward at x={x0}, ray={index}")
            measured.append(distance)
        previous = measured
    if fitted_w * scale <= REF.plug_tab_half_x or fitted_h * scale <= REF.plug_tab_half_y:
        raise RuntimeError("Bulkhead_Fitted.stl: slot does not clear the plug tab")

    sec = tm.section(plane_origin=[BULK.band_x0 + 0.002, 0.0, 0.0], plane_normal=[1.0, 0.0, 0.0])
    r = float(np.hypot(np.asarray(sec.vertices)[:, 1], np.asarray(sec.vertices)[:, 2]).max())
    if abs(r - BULK.barrel_r) > 2e-4:
        raise RuntimeError(f"Bulkhead.stl: barrel radius {r:.5f} in the mount band, reference {BULK.barrel_r}")

    # Local -> world, from the one matrix that states it. Applied here and
    # published as BULKHEAD_QUAT for whoever places the raw mesh instead of this
    # one; two spellings of a rotation are two things that can disagree, and a
    # receptacle rotated a quarter turn from its own bore reads as a mate that
    # never seats rather than as a transform that is wrong.
    Vw = (V @ _R_BULKHEAD.T) * scale
    Vw += np.asarray(bulkhead_spawn(scale), dtype=np.float64)
    # Express in the port static's local frame, so the block port's own
    # (port_origin, PORT_QUAT) transform places it.
    from robosimstudio.core.math import quat_to_mat as _q2m  # noqa: PLC0415

    asm = reference_assembly()

    Vl = (Vw - np.asarray(asm.port_origin, dtype=np.float64)) @ _q2m(np.asarray(PORT_QUAT))

    # The flange rim is the part's lowest point and the front step is stated
    # at its underside; port-local x runs up, so the rim is the frame's floor.
    if abs(float(Vl[:, 0].min()) + BULK.flange_r * scale) > 2e-4:
        raise RuntimeError(
            f"Bulkhead.stl: lowest point {Vl[:, 0].min():.5f} in port-local x, "
            f"reference flange radius {BULK.flange_r} x {scale}"
        )

    geom = BulkheadGeom(
        cav_half_wide=fitted_w * scale,
        cav_half_thin=fitted_h * scale,
        depth=(BULK.slot_deep_x - BULK.mouth_x) * scale,
        barrel_radius=BULK.barrel_r * scale,
    )
    mesh = newton.Mesh(np.ascontiguousarray(Vl, dtype=np.float32), np.ascontiguousarray(F.reshape(-1), dtype=np.int32))
    return mesh, geom


def bulkhead_boot_tip() -> tuple[float, float, float]:
    """The boot tip's centre in reference-world coordinates, mesh-free.

    Where the rear cable roots: on the barrel axis (the slot is centred on it,
    measured and asserted by :func:`load_bulkhead`), at the boot's rear face.
    """
    asm = reference_assembly()
    return (
        float(asm.plug_mated[0]),
        asm.mouth_y + (BULK.boot_x - BULK.mouth_x) * SCALE,
        float(asm.plug_mated[2]),
    )


def bulkhead_barrel_grasp() -> tuple[float, float, float]:
    """Where a side grasp takes the bulkhead, in reference-world coordinates.

    On the barrel axis, in the middle of the smooth band the mount is stated
    under -- so the pads land on the cylinder rather than across the flange rim
    at one end or the boot's step at the other. Derived from the same asserted
    band :func:`load_bulkhead` measures the barrel radius in, so a re-export
    that moves the band moves the grasp with it.
    """
    asm = reference_assembly()
    band_mid_x = 0.5 * (BULK.band_x0 + BULK.step_x1)
    return (
        float(asm.plug_mated[0]),
        float(asm.mouth_y + (band_mid_x - BULK.mouth_x) * SCALE),
        float(asm.plug_mated[2]),
    )


def bulkhead_fixture_boxes() -> tuple[tuple[str, tuple, tuple], ...]:
    """The parts-tray channel the free bulkhead rests in: ``(name, center, half)``.

    Four plain boxes, all from the asserted ``BULK`` constants so the spec is
    buildable without trimesh and cannot drift from the mesh the loader
    verifies:

    * ``cradle_front`` -- top at the flange rim's underside (``bore -
      flange_r``), beneath the disc's rim band. The nose (ring, flange, hex)
      carries the CoM, so a support forward of it is what stops the resting
      part pitching nose-down onto the table.
    * ``cradle_rear`` -- top at the smooth r10.5 barrel's underside, beneath
      ``band_x0..step_x1``. With the front step it holds the axis level: the
      CoM sits between the two bearing lines.
    There are deliberately **no side cheeks and no axial stop**. Cheeks were
    here while the part was braced in place, and they are exactly what a side
    grasp cannot live with: the support jaws close across the barrel at
    +-21 mm, which is where a cheek beside a 42 mm barrel has to stand. They
    are not missed -- a body of revolution has no gravity roll torque about its
    own axis, the rear cable roots on that axis too, and the part measures 0.05
    degrees of tilt over a settle without them. Roll during the lift is held by
    the grip, which is the honest place for it. Nothing stops the part sliding
    along its own axis either, because reacting the insertion push is the
    support arm's job -- which is what makes this bimanual rather than scenery.
    """
    asm = reference_assembly()
    x0 = float(asm.plug_mated[0])
    bore = float(asm.plug_mated[2])
    table = asm.pad_top - BENCH_H

    def band(a: float, b: float) -> tuple[float, float]:
        return (asm.mouth_y + (a - BULK.mouth_x) * SCALE, asm.mouth_y + (b - BULK.mouth_x) * SCALE)

    def box(name: str, y0: float, y1: float, top: float, half_x: float, x_off: float = 0.0):
        return (
            name,
            (x0 + x_off, 0.5 * (y0 + y1), 0.5 * (top + table)),
            (half_x, 0.5 * (y1 - y0), 0.5 * (top - table)),
        )

    fy0, fy1 = band(BULK.flange_x0, BULK.flange_x1)
    ry0, ry1 = band(BULK.band_x0, BULK.step_x1)
    # Half-widths stay inside the support pads' +-21 mm closing line, so the
    # jaws reach the barrel without meeting the fixture they lift it off.
    return (
        box("cradle_front", fy0, fy1, bore - BULK.flange_r * SCALE, 0.018),
        box("cradle_rear", ry0, ry1, bore - BULK.barrel_r * SCALE, 0.014),
    )
