# SPDX-License-Identifier: Apache-2.0
"""The plug and the receptacle, as declarations a goal can measure against.

The geometry itself lives in :mod:`.stage`, which builds the colliders; this is
the part a *task* is allowed to name. Two features carry the whole insertion
goal between them:

``port.bore`` — a ``cavity``
    Where the mouth plane is, which way the axis runs, how deep it goes, and how
    far along the plug its tab reaches. Stated in the **port's own frame**, so a
    receptacle mounted anywhere needs no new numbers; the reader rotates and
    translates it by the kind's own rules.

``plug.body`` — a ``grasp_box``
    Half-extents, from which the jaw measure takes the part's width across the
    closing direction. 10 mm of plug, which the fingers are commanded past.

**Every number is the reference assembly's own**, read from it at import rather
than retyped. Retyping would let the declaration and the collider it describes
disagree, which is the one thing a goal measured against a mesh must not do:
a bore stated 2 mm shallower than the one that was built scores a seat that
never happened, silently.

**Every part also says where it stands.** Each record names a factory
(``build=``) returning the spec a document's ``(:objects plug - usb_c_plug)``
or ``(:fixtures port - usb_c_port)`` lowers to: the mesh the record declares,
posed where the reference assembly spawns it, with the colliders a 0.15 mm fit
needs. That is the split the document keeps -- what exists and what must become
true is the document's business, where it starts is the assembly's -- and
stating the placement beside the declaration is what lets the document alone
put the three parts in a scene.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.assetlib import AssetRecord, register_asset
from robosimstudio.assetlib.lower import declared_asset
from robosimstudio.assetlib.semantics import FeatureSpec
from robosimstudio.core.math import quat_to_mat
from robosimstudio.core.render.palette import ACCENT_PLUG, ACCENT_PORT
from robosimstudio.scene.frames import point_from_surface
from robosimstudio.specs import ObjectSpec, SimRequirements, StaticSpec, contacts_with

from .stage import (
    BULK,
    BULKHEAD_QUAT,
    CELL,
    INSERT_AXIS,
    PLUG_QUAT,
    PORT_QUAT,
    REF,
    SCALE,
    bulkhead_spawn,
    reference_assembly,
)

__all__ = [
    "MATE_HYDROELASTIC",
    "MATE_SDF_VOXEL",
    "MATE_MU",
    "USB_C_BULKHEAD",
    "USB_C_PLUG",
    "USB_C_PORT",
    "benched_plug",
    "bolted_port",
    "free_bulkhead",
]

MATE_HYDROELASTIC = True
"""Collide the mating pair as pressure fields rather than as point sets.

**The default, because it is the contact model this fit actually works under.**
A 0.15 mm-class clearance carried by whichever handful of points the narrow
phase picked that step is a fit that chatters: the tab catches a corner, the
plug cocks, and a stroke that should seat reports a jam. An iso-pressure patch
carries it as a distributed surface instead.

It works here where swapping the *solver* did not, and the reason is structural:
hydroelastic changes what the narrow phase **emits**, not who integrates it. The
arm stays MuJoCo's, so the MJCF's own ``<position>`` actuators and
``mujoco:gravcomp`` -- the things every non-MuJoCo solver ignores, and the reason
they all failed on this task -- are untouched.

**Stated once and read by both parts**, because both ends of a mate have to
declare it or the pair falls back to point contacts silently, behaving as though
neither had asked. One constant is what makes that impossible to half-apply.

Costs a cooked SDF per mesh: slow once, cached on disk, and only on the two
parts that mate. Turn it off here to compare against the point-contact reading;
nothing else has to change.
"""

MATE_SDF_VOXEL = 2.5e-4
"""Voxel size the mating pair's hydroelastic SDFs are cooked at [m].

**The fit is 0.15 mm a side and the default cook cannot see it.**
:data:`~robosimstudio.scene.build.colliders.SDF_MAX_RESOLUTION` is 128 voxels
across the longest axis, and the bulkhead is 66 mm long once scaled, so the
default field is sampled every 0.5 mm -- more than three times coarser than the
clearance it has to carry. What the solver then collides is not this part: it is
a smeared slot whose walls are located to half a millimetre, and a tab that
should be guided by 0.15 mm of steel is free to cock inside it.

Measured against that default: 0/13 seated, median depth 1.0 mm, median tilt
10 degrees against a 5 degree gate. The authored geometry is not what failed --
`Bulkhead_Fitted.stl` really does carry the sleeve (measured inner half-width
4.198 mm against the tab's 4.125 mm, pre-scale). The sampling is what failed.

0.25 mm is chosen against what the sleeve actually IS: a DOUBLE-WALLED shell
whose fitted liner stands 0.552 mm (thin axis) and 0.646 mm (wide) inside its
outer skin. What the field has to do is SEPARATE those two skins; the 0.15 mm
clearance behind them is carried by the liner once the liner exists at all. At
the 128-cap the bulkhead's 66 mm box samples every 0.516 mm -- about one cell
between the skins -- so they merge, `fit_bulkhead.py`'s narrowing is discarded
wholesale, and the tab rides the OUTER skin at 3.152 mm half-height instead of
the fitted 2.600 mm. That surplus 0.55 mm is the slack the plug cocks in.

0.25 mm puts roughly 2.2 cells between the skins, which is what it takes for the
liner to exist to collision. Finer was measured and is not affordable: at
0.15 mm a single environment resident is about 29 GB and the runner is held to
one worker; at 0.1 mm Warp refuses the 7.5 GB allocation outright (CUDA error 2).
The field is DENSE over the mesh's bounding box, so cost goes as the cube.

The cook is content-addressed on disk, so the cost is paid once per machine
rather than once per episode.

**Stated once and read by both parts**, for :data:`MATE_HYDROELASTIC`'s reason:
a pressure pair sampled at two different scales is a fit whose two halves
disagree about where they are, and one constant is what makes that impossible to
half-apply.

This is a sampling refinement, not a change to the part. Nothing here moves a
wall, widens a slot, or adds a force -- it asks the simulator to represent the
mouth the CAD already describes.
"""


MATE_MU = 2.5
"""Receptacle friction on the free bulkhead, and the reason the plug stays in.

The block port keeps the reference sim's 0.35, and it is right for a bore whose
0.15 mm fit holds a plug **geometrically**. The bulkhead's slot clears the tab
by 1.55 mm a side, so once both hands are off, nothing holds the plug but
friction -- and a real USB-C receptacle does not rely on friction alone: it has
sprung detent tangs that latch the plug's shell. Standing those up as a raised
coefficient is a stated approximation, not a measured material, and 2.5 is past
any dry pair.

**On the receptacle only**, which is the load-bearing detail. Friction pairs by
the larger of the two, so the mate gets it either way -- but raising the *plug*
would also raise it against the gripper blades, the one contact the reference
calibrated its whole grasp on. Measured: both parts at 1.4 mates to 13.50 mm and
retains 11.94; the receptacle alone at 1.4 mates to 13.72 and retains 12.27 --
better on every axis with the shipped grasp untouched.

**Why 2.5 and not less**: the plug creeps out under its own cable, so the number
you measure depends on how long you watch. At 1.4 an episode ending as the jaws
opened read 12.3 mm; the same episode through a full retract-and-hold read
8.6-10.1, under the 10.4 gate, 1/4. The retract exposed it rather than causing
it. **And why not more**, which is the more useful half: friction saturates.

===========  =========  ==================
``MATE_MU``  successes  retained depth
===========  =========  ==================
1.4          1/4        8.6-10.1 mm
2.5          4/4        10.8-11.5 mm
4.0          3/4        10.9-11.3 mm
8.0          2/4        7.1-11.1 mm
===========  =========  ==================

Retained depth plateaus near 11 mm however hard the surfaces grip, because the
plug is not *sliding* out -- it is being **levered** out. The cable on its rear
face applies a moment, the plug cocks in a slot that clears its tab by 1.55 mm a
side, and past some angle the tab is not in the slot at all. Friction cannot
answer a moment, which is why more of it buys nothing and costs consistency --
and why :func:`~tasks/usbc_insert/assets/stage.py::load_bulkhead`'s own note
records 8.0 moving the post-release droop not at all.
"""


def _bore_in_port_frame() -> dict[str, object]:
    """The bore, converted out of world coordinates into the port's own.

    The reference assembly states its geometry in the world, because that is
    where the reference measured it. A feature has to be frame-relative or it
    cannot travel with the body it belongs to -- so the conversion happens here,
    once, against the port's own resting orientation.

    On this assembly it comes out to a mouth 13 mm along the port's local +z with
    the axis running back along -z, which is what a receptacle looks like from
    inside: the bore runs *into* the block from its face.
    """
    assembly = reference_assembly()
    rotation = quat_to_mat(np.asarray(PORT_QUAT, dtype=np.float64))
    origin = np.asarray(assembly.port_origin, dtype=np.float64)
    mouth_world = np.array([origin[0], assembly.mouth_y, origin[2]])
    return {
        "mouth": [round(float(v), 9) for v in rotation.T @ (mouth_world - origin)],
        "axis": [round(float(v), 9) for v in rotation.T @ np.asarray(INSERT_AXIS, dtype=np.float64)],
        "depth": float(REF.bore_depth),
        "half_extents": [float(REF.port_cav_half_x), float(REF.port_cav_half_y)],
        # How far along the plug's own +z its tab tip reaches. Declared beside
        # the bore rather than on the plug because the bore is the only thing
        # that cares: a receptacle accepting parts with different tip offsets
        # would be measuring different things under one name.
        "tip_offset": float(assembly.plug.tab_tip_z),
    }


def bolted_port() -> StaticSpec:
    """The receptacle bolted to the bench, where the reference assembly puts it.

    Names the suite's own ``usbc_port`` entity rather than letting the generic
    mesh path load ``USB_C_Port.stl`` as authored: the reference's published
    behaviour was measured on a **chamfered** mouth, and a receptacle without the
    lead-in demands the 0.15 mm approach accuracy raw. See
    :class:`~tasks/usbc_insert/assets/rig.py.PortUsbc`, which builds it, and
    ``stage.CHAMFER`` for the double-scale it inherits.

    ``reference_assembly().port_origin`` is the same arithmetic
    :func:`~tasks/usbc_insert/assets/stage.py::bulkhead_spawn` runs through to land
    the free mesh on the same mate line, which is why the two receptacles' mouths
    are coplanar by construction and the two documents can state one set of
    thresholds.
    """
    return StaticSpec(
        name="port",
        entity="usbc_port",
        xform=(*point_from_surface(CELL, reference_assembly().port_origin), *PORT_QUAT),
        color=ACCENT_PORT,
    )


USB_C_PORT = register_asset(
    AssetRecord(
        name="usb_c_port",
        summary="A USB-C receptacle: a 13 mm bore behind a 16.8 x 5.2 mm mouth.",
        tags=("connector", "receptacle"),
        features=(FeatureSpec(name="bore", kind="cavity", payload=_bore_in_port_frame()),),
        sim=SimRequirements(
            domain="static",
            contacts=contacts_with("objects", "robot"),
            notes="Welded: the plug mates into it and the work hand's blades reach into the mouth.",
        ),
        provenance=(
            "USB Type-C connector dimensions by way of the insertion reference this suite "
            "reproduces; built by tasks/usbc_insert/assets/stage.py, which owns the colliders."
        ),
        build=bolted_port,
    )
)


def _bulkhead_bore() -> dict[str, object]:
    """The bulkhead's slot, in its own frame -- which is the **STL's frame**.

    Not the block port's, which is what this payload copied for as long as the
    two mouths being coplanar *in the world* was mistaken for the two bodies
    sharing a local frame. The free bulkhead is spawned as the raw STL --
    barrel axis along local ``+x``, exactly the frame :data:`~.stage.BULK`
    states its constants in -- and :data:`~.stage.BULKHEAD_QUAT` is the
    permutation that carries local ``+x`` onto the world mate line. Composing
    the port-frame mouth/axis (``+z``/``-z``) with that body pose sent the
    measured bore axis to world ``-x``: the insertion atom read **+13 mm of
    depth at spawn** with the plug still 46 mm out, and every seat and verify
    this variant ever scored was read in that permuted frame.

    So: mouth at ``BULK.mouth_x`` along local ``+x`` on the barrel axis, the
    insertion direction ``+x`` into the slot, both post-scale like the port's.

    The current fitted sleeve uses the block port's 16.8 x 5.2 mm internal
    section. The original barrel shell and 27.6 mm pocket depth are retained.
    Its fixed dimensions are declared in assets/bulkhead_geometry.py and checked
    by ray sections through the delivered collision/render mesh.
    """
    payload = dict(_bore_in_port_frame())
    payload["mouth"] = [round(float(BULK.mouth_x * SCALE), 9), 0.0, 0.0]
    payload["axis"] = [1.0, 0.0, 0.0]
    payload["depth"] = float((BULK.slot_deep_x - BULK.mouth_x) * SCALE)
    from .bulkhead_geometry import FIT_HALF_WIDE_M, FIT_HALF_THIN_M
    payload["half_extents"] = [FIT_HALF_WIDE_M, FIT_HALF_THIN_M]
    return payload


def free_bulkhead() -> ObjectSpec:
    """The receptacle as a free body, resting in its cradle -- what the document says.

    **It was an extras entity and could not be measured.** ``(:objects bulkhead
    - usb_c_bulkhead)`` is a claim that the scene publishes a handle under that
    name, and only objects, statics and deformables do; a suite-owned extra is built
    and stepped and is invisible to ``handles.entity``. So the goal's
    ``bulkhead.bore`` raised at the first reset, naming a scene whose objects
    were ``['plug']``. A chain gets away with it -- the rod's readers go through
    ``snapshot.extras`` on purpose, because its segments are bodies no document
    names -- and a rigid body a document *does* name has no such excuse.

    Placed by :data:`~tasks/usbc_insert/assets/stage.py.BULKHEAD_QUAT` and
    :func:`~tasks/usbc_insert/assets/stage.py::bulkhead_spawn`, which are the same
    rotation and translation ``load_bulkhead`` applies to the mesh it measures.
    Stated once there and read here, so the imported receptacle and the measured
    one cannot end up a quarter turn apart.

    ``mass`` rather than a density: the shell is not watertight, so a
    density-derived mass off it is noise. 90 g is the reference sim's own
    ground truth for the metal-shelled part.

    The authored mesh, colliding as authored -- retention is NOT this
    collider's job. Its corridor clears the tab 0.85 mm a side (the inner
    skin of a double-walled shell; every extent read reports the outer one at
    1.55 -- see :data:`~tasks/usbc_insert/assets/stage.py._BulkheadRef.slot_inner_half_w`),
    and no slot this stack can express holds a released plug against its own
    cable in a braced-free arrangement: liners from 0.45 down to 0.2 mm a
    side, catch lips, dovetails and a Morse taper were all cut, cooked at up
    to 75 um voxels, run, and refuted. Retention is the detent entity's job
    -- :class:`~tasks/usbc_insert/assets/entity.py.SpecUsbcDetent`, which says so out
    loud -- and entry, the part that is real manipulation, needs the authored
    mouth exactly as it is.
    """
    return ObjectSpec(
        name="bulkhead",
        category="usb_c_bulkhead",
        # The scale stays here: it is a fact about this import, not the part.
        asset=declared_asset(USB_C_BULKHEAD, scale=SCALE),
        pose=point_from_surface(CELL, bulkhead_spawn()),
        quat=BULKHEAD_QUAT,
        mass=BULK.mass,
        # Not the plug's 0.35: this slot is 1.55 mm a side wider than the tab
        # and friction is what keeps a released mate in it. See MATE_MU.
        friction=MATE_MU,
        contact_gap=2.0e-4,
        # The receptacle half of the pressure pair. See MATE_HYDROELASTIC.
        hydroelastic=MATE_HYDROELASTIC,
        # Sampled fine enough to carry its own slot. See MATE_SDF_VOXEL.
        sdf_voxel=MATE_SDF_VOXEL,
        color=ACCENT_PORT,
    )


USB_C_BULKHEAD = register_asset(
    AssetRecord(
        name="usb_c_bulkhead",
        summary="A panel-mount USB-C bulkhead: a 27.6 mm slot down a 42 mm barrel, carried free.",
        tags=("connector", "receptacle", "panel-mount"),
        files={"mesh": "Bulkhead_Fitted.stl"},
        features=(FeatureSpec(name="bore", kind="cavity", payload=_bulkhead_bore()),),
        provenance=(
            "Bulkhead_Fitted.stl is generated from Bulkhead.stl by assets/fit_bulkhead.py. "
            "The inner sleeve is narrowed to 16.8 x 5.2 mm at scale 2, giving the "
            "16.5 x 4.8 mm plug tab 0.15/0.20 mm clearance per side. Rendering and "
            "collision use this same mesh; the external barrel and boot are unchanged. "
            "The original explicit 90 g bulkhead mass is retained."
        ),
        build=free_bulkhead,
    )
)


def benched_plug() -> ObjectSpec:
    """The plug, resting on the bench where the reference assembly spawns it."""
    return ObjectSpec(
        name="plug",
        category="usb_c_plug",
        # The mesh the plug's record declares. The scale stays here: it is
        # a fact about this import, not about the part.
        asset=declared_asset(USB_C_PLUG, scale=SCALE),
        pose=point_from_surface(CELL, reference_assembly().plug_spawn),
        quat=PLUG_QUAT,
        density=2000.0,
        friction=0.35,
        contact_gap=2.0e-4,
        # The moving half of the pressure pair, and it is stated on both or
        # on neither -- see MATE_HYDROELASTIC, whose one constant is what
        # makes a scene where only one end asks impossible to write here.
        hydroelastic=MATE_HYDROELASTIC,
        # The other half of the pair, at the same scale. See MATE_SDF_VOXEL.
        sdf_voxel=MATE_SDF_VOXEL,
        # Warm against the port's blue: the frame's whole job is to show
        # where the plug is relative to the bore, which two cool greys a
        # centimetre apart cannot do. A bare STL states nothing, so without
        # this the plug takes a build-order palette colour.
        color=ACCENT_PLUG,
    )


USB_C_PLUG = register_asset(
    AssetRecord(
        name="usb_c_plug",
        summary="A USB-C plug: a 22 x 10 mm body with a 16.5 x 4.8 mm tab.",
        tags=("connector", "plug"),
        files={"mesh": "USB_C_Plug.stl"},
        features=(
            FeatureSpec(
                name="body",
                kind="grasp_box",
                payload={
                    # The narrowest half-extent is what a parallel jaw closes on,
                    # so the 5 mm here is the number the grip measure reads.
                    "half": [
                        float(REF.plug_half_x),
                        float(REF.plug_half_y),
                        float(reference_assembly().plug.length / 2),
                    ],
                    "center_local": [0.0, 0.0, float(reference_assembly().plug.com_z)],
                },
            ),
        ),
        provenance="Same reference as the port. Mesh is assets/usbc/USB_C_Plug.stl at scale 2.",
        build=benched_plug,
    )
)


# --------------------------------------------------------------------- reading
DEPTH_GATE = 0.8 * float(REF.bore_depth)
"""Tab-tip depth past the mouth that counts as inserted [m]: 10.4 mm of 13.

Here rather than in a task, because it is a fraction of *this connector's* bore.
The document states the resulting number so it can be read on its own, and
:func:`gates` is what keeps the two from drifting.
"""

LATERAL_GUARD = float(np.hypot(REF.port_cav_half_x, REF.port_cav_half_y))
"""Depth only counts inside the mouth's half-diagonal [m].

The geometric guard against scoring a miss that slid past the port face -- see
the ``insertion`` atom, which reports the offset precisely so a goal can apply
this.
"""


def gates() -> dict[str, float]:
    """The two numbers a goal over this connector has to state.

    Exposed so a document and a policy read the same values rather than each
    carrying a copy. A task *does* still write them out -- a goal whose
    thresholds are invisible is a goal nobody can review -- and the port's
    equivalence test asserts the document's literals against these.
    """
    return {"depth_gate": DEPTH_GATE, "lateral_guard": LATERAL_GUARD}


def measure_insertion(ctx, plug: str = "plug", port: str = "port", payload: dict | None = None) -> dict[str, float]:
    """Depth, lateral offset and mate gap for a live context.

    The convenience form of the ``insertion`` atom, for a graph tool that wants
    the numbers without building a goal clause. It calls the **same** function
    the atom's handler calls, against the **same** declared cavity, so a tool and
    the goal cannot disagree about how deep the plug is -- which they did before
    this port, as two transcriptions of one measurement.

    ``payload`` is the receptacle's own ``cavity`` payload; left ``None`` it is
    the block port's, which is only right for the bolted ``port`` variant. The
    caller that measures against a *named* receptacle passes that receptacle's
    declared feature -- mouth and axis are stated in the receptacle's local
    frame, so the port's payload against the bulkhead's pose is a bore rotated
    off its own slot, not a shared constant.
    """
    from robosimstudio.domains.rigid.measures.insertion import insertion  # noqa: PLC0415

    if payload is None:
        payload = USB_C_PORT.feature("bore").payload
    # The port is a welded static, so it has no `body_q` row and `ctx.pose`
    # cannot answer for it -- the compiler recorded where it was built instead.
    # Reading the wrong one does not fail: it places the bore at the origin and
    # reports a plug three centimetres from a hole that is a metre away.
    handle = ctx.handles.entity(port)
    pose = np.asarray(handle.xform if handle.body < 0 else ctx.pose(port), dtype=np.float64)
    rotation = quat_to_mat(np.asarray(pose[3:7], dtype=np.float64))
    world = {
        "mouth": np.asarray(pose[:3], dtype=np.float64) + rotation @ np.asarray(payload["mouth"], dtype=np.float64),
        "axis": rotation @ np.asarray(payload["axis"], dtype=np.float64),
        "depth": payload["depth"],
    }
    return insertion(ctx.pose(plug), world, tip_offset=float(payload["tip_offset"]))
