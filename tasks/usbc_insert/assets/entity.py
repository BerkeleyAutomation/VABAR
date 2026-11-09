# SPDX-License-Identifier: Apache-2.0
"""The suite's cables: articulated capsule chains off the plug and the boot.

Ported from the ``newton_tasks`` bimanual MJC sims' cable machinery -- the
architecture that matches this suite's solver. The AVBD variant builds its
cable as a rod; the MJC variant, which this suite's physics is, articulates it:
a chain of capsule links with small armature and joint friction, solved
natively by MuJoCo in the same model as everything else, so the cable's pull
on the plug and the plug's reaction on the cable are whatever the solver says
they are.

One deliberate departure from the reference's ball joints: each link bends on
a **revolute whose axis alternates** between the chain's two normals. A ball
joint is four quaternion coords, and 32 of them put the model at 151 --
past the 64-coord ceiling ``robot.ik.arm_ik`` holds Newton's LM solver
to, so every Cartesian move silently fell back to the FD solver, which sags
(measured: 7 mm of droop and 4 deg of tilt over the carry, planting the
insertion 5.5 mm under a slot with 1.6 mm of vertical clearance -- a failure
that looked exactly like cable drag and was not). Alternating hinges are one
coord each, keep the model at 55, bend every two segments in each plane, and
lock only torsion, which a jacketed cable barely has to give.

Two chains, one entity:

* **The plug's cable**, rooted on the plug's rear face, extending out the back
  (world ``-y`` at spawn). It rides the bench for its near span and drapes off
  the bench's rear edge to the table. Its weight is the honest load an
  insertion has to carry, and after release it is what droops the seated plug.
* **The bulkhead's rear cable**, rooted at the boot tip *on the bulkhead
  body* (named by ``port_object``) and running ``+y``, spawned straight and
  left to settle. The receptacle is a free body the support arm carries, so a
  chain rooted anywhere else is a chain the part visibly tears off at the
  boot the moment it lifts. The reference spawns its chain on a shelf tray
  because *its* pendulum transient levers the receptacle out of a cradle;
  this chain weighs 1.9 g against the bulkhead's 90 g and roots on the barrel
  axis the cradle bears, so it drapes from the boot to the table and no shelf
  is needed.

The chains are their own articulations. ``add_body`` closed the plug's
articulation at one free joint, and a chain whose root parent is a body of
another articulation is a structure the MuJoCo conversion handles (measured:
the chain follows a falling body with the joints intact). Contact softness for
the gram-scale links is the reference's: MuJoCo's default 20 ms solref, which
the rig's stiffening deliberately does not touch, plus zero torsional and
rolling friction so links slide freely on the bench.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import newton
import numpy as np
import warp as wp

from robosimstudio.assetlib import AssetRecord, register_asset
from robosimstudio.scene.entity import Entity, label_index
from robosimstudio.scene.frames import point_from_surface
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs import SimRequirements, contacts_with

from .rig import CONTACT_GAP
from .stage import BULK, CELL, REF, SCALE, bulkhead_boot_tip

__all__ = [
    "CABLES",
    "EVALUATION_CABLES",
    "PLUG_CABLE",
    "DETENT",
    "EntityUsbcCables",
    "EntityUsbcDetent",
    "SpecUsbcCables",
    "SpecUsbcDetent",
    "bulkhead_detent",
    "connector_cables",
]

CABLE_MU = 0.4
"""The reference's cable friction."""
CABLE_ARMATURE = 2.0e-5
"""Bend-joint armature [kg m^2]: the reference's stabiliser for gram-scale links."""
CABLE_JOINT_FRICTION = 2.0e-3
"""Bend-joint friction: what damps the free chain's pendulum.

It was suspected of pushing the released mate back open and cleared: 2e-3,
5e-4 and 1e-4 measured the same release give-back to the millimetre (1e-4 --
a chain limp enough to drape off the rear face -- measured *worse*), so the
authored value stands. The give-back was released-mate creep, an arrangement
property the detent entity now answers; see :class:`SpecUsbcDetent`."""
CABLE_COLOR = (0.90, 0.90, 0.90)
"""Jacket colour: the same white as the arms' tube shells.

The reference sims run a near-black jacket (0.16), which is what a real cable
usually is and what this shipped first. It reads badly here: against a dark
table and a dark-shelled robot a black cable disappears, and the 24-link chain
that does show up reads as a shadow across the mate. White separates it from
both, and matches ``embodiments.yam_look``'s tube shells so the scene looks
like one set."""


@dataclass(frozen=True, kw_only=True)
class SpecUsbcCables:
    """Both cables' knobs, at the reference sims' defaults.

    Lengths and the radius are **pre-scale** (multiplied by ``scale`` at
    build), the same convention the reference's arguments use. Both densities
    are the 450 kg/m^3 the bulkhead sim derived from a real 6 mm rubber
    jacket's ~30 g/m -- not the AVBD sim's 1400 default, whose ~21 g on this
    plug chain was measured here dragging the plug down through the jaws
    during the carry and planting the insertion on the bench, 12.4 mm under
    the bore. The same class of failure that sim's own note records (the
    inherited density prying the part out of its cradle), resolved the same
    way.

    ``port_root`` is the boot tip in the scene's work-surface frame with the
    chain's run direction as its rotation -- stated by the suite from the same
    reference numbers that place the port static, so the chain roots where the
    boot actually is.
    """

    name: str = "cables"
    entity: str = "usbc_cables"
    scale: float = 2.0
    plug_object: str = "plug"
    plug_links: int = 24
    plug_length: float = 0.18
    plug_radius: float = 0.0016
    plug_density: float = 450.0
    plug_rear_z: float = -0.034
    """The plug's rear face along its local insertion axis [m, **post-scale**]:
    ``stage.REF.plug_body_z0``, restated because a spec must build without the
    mesh loaded."""
    port_links: int = 8
    port_length: float = 0.05
    port_density: float = 450.0
    port_root: tuple[float, ...] = ()
    """Boot-tip xform (7-vector, work-surface frame): position, then a quat
    taking capsule-local ``+z`` to the chain's run direction. Empty builds no
    port chain."""
    port_object: str = ""
    """The receptacle body the rear chain roots on -- ``"bulkhead"`` for the
    bimanual variant, whose free receptacle the support arm carries and whose
    chain must therefore ride it. Empty roots the chain on the world, which is
    what a bolted panel-mount static is. Stated by the suite rather than
    guessed here: this used to be a probe for a body labelled ``port`` with a
    silent world fallback, and in the bulkhead scene -- whose receptacle the
    document names ``bulkhead`` -- the probe always missed, so the rear cable
    anchored to the air and the lifted part left its own boot's cable behind."""


def _host_articulation(builder: newton.ModelBuilder, parent: int) -> int | None:
    """The articulation of body ``parent``, which a chain rooted on it belongs to; ``None`` on the world.

    Newton 1.6 and later refuse a joint whose parent body sits in another
    articulation (``ModelBuilder._validate_joints``); the Newton this suite was
    measured on accepted it, and its MuJoCo conversion made the chain a subtree
    of the body it roots on. Labelling the chain's joints with that body's
    articulation is what gives the conversion the same tree today.
    """
    if parent < 0:
        return None
    host = next(
        (builder.joint_articulation[j] for j in range(len(builder.joint_child)) if builder.joint_child[j] == parent),
        -1,
    )
    return host if host >= 0 else None


@register_entity("usbc_cables")
class EntityUsbcCables(Entity):
    """Build both chains after the bodies they root on exist."""

    spec: SpecUsbcCables

    def __init__(self, spec: SpecUsbcCables, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.plug_cable_bodies: list[int] = []
        self.port_cable_bodies: list[int] = []
        self.bend_coords: list[int] = []

    def _link_cfg(self, builder: newton.ModelBuilder, density: float) -> newton.ModelBuilder.ShapeConfig:
        cfg = builder.default_shape_cfg.copy()
        cfg.mu = CABLE_MU
        cfg.gap = CONTACT_GAP
        cfg.density = float(density)
        cfg.mu_torsional = 0.0
        cfg.mu_rolling = 0.0
        # AVBD-parity contact gains from the reference; MuJoCo reads solref
        # instead, which stays at its soft default for these shapes on purpose.
        cfg.ke = 1.0e3
        cfg.kd = 300.0
        return cfg

    def _chain(
        self,
        builder: newton.ModelBuilder,
        *,
        parent: int,
        root_xform,
        root_anchor,
        links: int,
        seg_half: float,
        radius: float,
        density: float,
        tag: str,
    ) -> list[int]:
        """One articulated capsule chain, running along ``root_xform``'s +z.

        ``root_anchor`` is the first joint's anchor in the parent's frame --
        the rear face for the plug, the world-frame boot tip for the static
        bulkhead (``parent=-1`` makes the parent frame the world).
        """
        cfg = self._link_cfg(builder, density)
        root_parent = parent
        seg = 2.0 * seg_half
        bodies: list[int] = []
        joints: list[int] = []
        for i in range(links):
            xf = wp.transform_multiply(
                root_xform, wp.transform(wp.vec3(0.0, 0.0, float(seg_half + i * seg)), wp.quat_identity())
            )
            b = builder.add_link(xform=xf, label=f"{tag}{i}")
            builder.add_shape_capsule(
                b,
                radius=float(radius),
                half_height=float(seg_half),
                cfg=cfg,
                color=CABLE_COLOR,
                label=f"{tag}{i}",
            )
            pxf = root_anchor if i == 0 else wp.transform(wp.vec3(0.0, 0.0, float(seg_half)), wp.quat_identity())
            # Revolute with the bend axis alternating between the two normals
            # of the chain, not the reference's ball. A ball joint is 4
            # quaternion coords, and 32 of them put the model at 151 -- past
            # the 64-coord ceiling robot.ik.arm_ik holds the LM solver to,
            # so every Cartesian move fell back to the FD solver, which sags:
            # measured 7 mm of TCP droop and 4 deg of tilt over the carry,
            # planting the insertion 5.5 mm under a slot with 1.6 mm of
            # vertical clearance. Alternating hinges are 1 coord each (55
            # total, LM retained), bend every 2*seg in each plane, and lock
            # only torsion -- which a jacketed cable barely has to give.
            axis = (1.0, 0.0, 0.0) if i % 2 == 0 else (0.0, 1.0, 0.0)
            j = builder.add_joint_revolute(
                parent=parent,
                child=b,
                parent_xform=pxf,
                child_xform=wp.transform(wp.vec3(0.0, 0.0, float(-seg_half)), wp.quat_identity()),
                axis=axis,
                limit_lower=-2.0,
                limit_upper=2.0,
                armature=CABLE_ARMATURE,
                friction=CABLE_JOINT_FRICTION,
                label=f"{tag}{i}_bend",
            )
            joints.append(j)
            bodies.append(b)
            self.bend_coords.append(int(builder.joint_q_start[j]))
            parent = b
        builder.add_articulation(joints, label=tag)
        host = _host_articulation(builder, root_parent)
        if host is not None:
            # The chain keeps its own articulation range (its own FK pass, as
            # before) and joins the tree of the body it roots on; see
            # _host_articulation.
            for j in joints:
                builder.joint_articulation[j] = host
        return bodies

    # -------------------------------------------------------------- phase 1
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        S = float(spec.scale)

        # --- the plug's cable, rooted on its rear face ---------------------
        if spec.plug_links > 0:
            plug = label_index(list(builder.body_label), spec.plug_object)
            spawn = builder.body_q[plug]
            seg_half = 0.5 * (spec.plug_length * S) / spec.plug_links
            rear = float(spec.plug_rear_z)
            # The chain runs out the rear: plug-local -z, so the root frame is
            # the spawn pose flipped about x with its origin on the rear face.
            flip = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi)
            root = wp.transform_multiply(spawn, wp.transform(wp.vec3(0.0, 0.0, rear), flip))
            self.plug_cable_bodies = self._chain(
                builder,
                parent=plug,
                root_xform=root,
                root_anchor=wp.transform(wp.vec3(0.0, 0.0, rear), flip),
                links=int(spec.plug_links),
                seg_half=seg_half,
                radius=spec.plug_radius * S,
                density=spec.plug_density,
                tag="cable",
            )
            print(
                f"[usbc_cables] plug: {spec.plug_links} links, {spec.plug_length * S * 1e3:.0f} mm, "
                f"radius {spec.plug_radius * S * 1e3:.1f} mm, rooted on the rear face"
            )

        # --- the bulkhead's rear cable, rooted at the boot tip -------------
        if spec.port_links > 0 and spec.port_root:
            placed = self.placed(spec.port_root)
            root = wp.transform(wp.vec3(*[float(v) for v in placed[:3]]), wp.quat(*[float(v) for v in placed[3:7]]))
            # Root on the receptacle *body* when the spec names one (the
            # bimanual variant's free bulkhead -- the chain must ride the part
            # through the carry), and on the world when the receptacle is a
            # static. The anchor is the same boot-tip frame either way,
            # expressed in whichever parent it roots on. A named body that is
            # missing raises: the silent world fallback that stood here is how
            # the bulkhead's cable came to hang off a weld to the air.
            port = label_index(list(builder.body_label), spec.port_object) if spec.port_object else -1
            anchor = root if port < 0 else wp.transform_multiply(wp.transform_inverse(builder.body_q[port]), root)
            seg_half = 0.5 * (spec.port_length * S) / spec.port_links
            self.port_cable_bodies = self._chain(
                builder,
                parent=port,
                root_xform=root,
                root_anchor=anchor,
                links=int(spec.port_links),
                seg_half=seg_half,
                radius=spec.plug_radius * S,
                density=spec.port_density,
                tag="pcable",
            )
            where = f"the {spec.port_object} body" if port >= 0 else "the world"
            print(
                f"[usbc_cables] port: {spec.port_links} links, {spec.port_length * S * 1e3:.0f} mm "
                f"off the boot tip, rooted on {where}"
            )

    # -------------------------------------------------------------- phase 2
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        if self.plug_cable_bodies:
            masses = model.body_mass.numpy()
            cm = float(np.sum(masses[self.plug_cable_bodies]))
            print(f"[usbc_cables] plug cable mass {cm * 1e3:.1f} g")

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline, episode) -> None:
        """Straighten both chains, which is the configuration they were built at.

        Without this a cable is the one thing in the scene that remembers the
        last episode. The bodies it roots on are restored -- both connectors
        are ``ObjectSpec`` free bodies, put back by their own entities -- and
        the environment zeroes every joint velocity, but a bend angle is a
        *position*, so the chain stays draped wherever the previous episode
        left it and then yanks its root back there. Measured before this
        existed: episode 0 seated the plug, episode 1 ended 118 mm off the bore
        and episode 2 253 mm off, each worse than the last as the cable wound
        further out.
        """
        del episode
        if not self.bend_coords:
            return
        for state in (pipeline.state_0, pipeline.state_1):
            q = state.joint_q.numpy()
            q[self.bend_coords] = 0.0
            state.joint_q.assign(q)


# ---------------------------------------------------------------- the detent

DETENT_GAINS = (600.0, 7.0, 12.0, 0.5, 0.01, 0.03)
"""The detent weld's PD gains and ceilings, in launch order:
``kp`` [N/m], ``kd`` [N s/m], ``f_max`` [N], ``kr`` [N m/rad],
``kd_rot`` [N m s/rad], ``t_max`` [N m].

Sized like the cable latch's, for the same reason its docstring gives:
unclamped, a position latch is an energy pump with no ceiling, and the clamp
turns the worst case into *slip*, which is what a real detent does when
overpowered. ``f_max`` 12 N is a real USB-C shell's extraction-force scale
(the spec's 8-20 N), four orders above the millinewton standing loads a
seated mate carries, so the weld never saturates in ordinary episodes; ``kp``
against the ~20 g plug is near-critically damped by ``kd`` and comfortably
inside the 960 Hz substep. The ceilings earn their keep during the retract,
when the rising cage can brush the welded plug: a first cut at 25 N and a
full 1 N m of torque authority let that brush escalate into weld chatter,
measured whipping the gram-scale cable links off the plug's rear at 6 m/s.
0.03 N m is ~1 N at the plug's rear-face lever: enough to hold every honest
load, not enough to whip anything."""


@wp.kernel(enable_backward=False)
def _hold_mate(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    pair: wp.array(dtype=int),
    latched: wp.array(dtype=int),
    rel_tf: wp.array(dtype=wp.transform),
    captured_depth: wp.array(dtype=float),
    params: wp.array(dtype=float),
    gains: wp.array(dtype=float),
):
    """Engage at the click point, then weld -- both halves on device.

    Unlatched, this measures the mate the way the insertion atom does -- tab
    tip depth past the mouth along the receptacle's own axis, lateral offset,
    axis cosine -- and the substep the click criterion is met it captures the
    plug's pose in the receptacle's frame and sets the flag. Latched, it is
    the cable suite's ``_hold_segment`` in six degrees of freedom: a clamped
    PD wrench toward the captured pose, equal and opposite on both bodies, so
    the pair rides the support arm as one assembly. All state lives in device
    arrays because the substep loop is CUDA-graph captured: a baked float is
    forever, an array read is live.
    """
    plug = pair[0]
    port = pair[1]
    X_plug = body_q[plug]
    X_port = body_q[port]
    tip_w = wp.transform_point(X_plug, wp.vec3(0.0, 0.0, params[1]))
    tip_l = wp.transform_point(wp.transform_inverse(X_port), tip_w)
    depth = tip_l[0] - params[0]
    if latched[0] == 0:
        lat = wp.sqrt(tip_l[1] * tip_l[1] + tip_l[2] * tip_l[2])
        a_plug = wp.transform_vector(X_plug, wp.vec3(0.0, 0.0, 1.0))
        a_bore = wp.transform_vector(X_port, wp.vec3(1.0, 0.0, 0.0))
        if depth >= params[2] and lat <= params[3] and wp.dot(a_plug, a_bore) >= params[4]:
            rel_tf[0] = wp.transform_multiply(wp.transform_inverse(X_port), X_plug)
            captured_depth[0] = depth
            latched[0] = 1
        return
    # The ratchet. A real detent is a spring the plug clicks *past*: it never
    # resists going deeper, only coming back. Advancing the capture on a
    # deeper pose is what keeps the weld from fighting the rest of the seat
    # stroke -- the push carries 3 mm past the click point, and a fixed
    # capture turned that into a standing 1.8 N arm-vs-weld fight, paid out
    # later as chatter.
    #
    # **Depth only.** This used to re-capture the whole relative pose, and
    # that made it a tilt ratchet: the released mate oscillates at the
    # micron scale, any transient droop that coincides with a depth uptick
    # became the weld's new *target*, and the seated plug crept from its
    # ~1 degree latch attitude to 5.13-5.19 degrees within the verify's 48
    # settle steps -- 0.13-0.19 degrees past the goal's 5-degree clause, so
    # the episode score was a race between the rising hand's 50 mm sphere
    # and the creep (measured 2/4 with everything else fixed). A real
    # detent never re-sets its rest *angle* because the plug wiggled
    # deeper: slide the captured pose along the bore axis by exactly the
    # deepening and keep the captured rotation and lateral offset.
    if depth > captured_depth[0]:
        t_rel = wp.transform_get_translation(rel_tf[0])
        rel_tf[0] = wp.transform(
            wp.vec3(t_rel[0] + (depth - captured_depth[0]), t_rel[1], t_rel[2]),
            wp.transform_get_rotation(rel_tf[0]),
        )
        captured_depth[0] = depth
        return

    target = wp.transform_multiply(X_port, rel_tf[0])
    p_plug = wp.transform_get_translation(X_plug)
    p_port = wp.transform_get_translation(X_port)
    dp = wp.transform_get_translation(target) - p_plug

    v_port = wp.spatial_top(body_qd[port])
    w_port = wp.spatial_bottom(body_qd[port])
    v_carried = v_port + wp.cross(w_port, p_plug - p_port)
    v_rel = wp.spatial_top(body_qd[plug]) - v_carried
    force = gains[0] * dp - gains[1] * v_rel
    f_mag = wp.length(force)
    if f_mag > gains[2]:
        force = force * (gains[2] / f_mag)

    q_err = wp.transform_get_rotation(target) * wp.quat_inverse(wp.transform_get_rotation(X_plug))
    sign = wp.float32(1.0)
    if q_err[3] < 0.0:
        sign = wp.float32(-1.0)
    # Small-angle rotation vector: exact enough for a weld that holds its
    # error near zero, and free of the axis-angle singularity at identity.
    rv = wp.vec3(q_err[0], q_err[1], q_err[2]) * (2.0 * sign)
    w_rel = wp.spatial_bottom(body_qd[plug]) - w_port
    torque = gains[3] * rv - gains[4] * w_rel
    t_mag = wp.length(torque)
    if t_mag > gains[5]:
        torque = torque * (gains[5] / t_mag)

    wp.atomic_add(body_f, plug, wp.spatial_vector(force, torque))
    # Reaction about the port's own origin: the force acts at the plug origin.
    r = p_plug - p_port
    wp.atomic_add(body_f, port, wp.spatial_vector(-force, -torque - wp.cross(r, force)))


@dataclass(frozen=True, kw_only=True)
class SpecUsbcDetent:
    """The receptacle's retention spring, modeled as a criterion-gated weld.

    A real USB-C shell retains its plug with leaf-spring fingers at 8-20 N of
    extraction force. This contact stack provably cannot express that spring:
    rigid contact at the mate's 1e11 Pa/m pressure law stores nanometres of
    preload (a wedge holds nothing), catch geometry under ~0.3 mm cooks into
    cams at any affordable SDF resolution, and a receptacle *braced by an arm*
    is never microscopically still, so static friction never pins the released
    plug -- measured creep to ~7.8 mm / ~10 deg across twelve refuted
    variants: bore clearances 0.2/0.3/0.45 mm, re-entry lips, dovetail teeth,
    a 2.7 deg Morse taper, three cable stiffnesses, zero seat preload,
    contact_gap 4x smaller, a 5x slower release, impratio 10. The welded
    block-port variant holds the same fit with the same stack, which is the
    existence proof that grounding, not geometry, is what the arrangement
    lacks.

    So the detent is stated as what it is: when the tab is at click depth, on
    axis, and straight -- the state where real spring fingers snap home -- the
    plug and receptacle become one assembly, and stay one until reset. Entry
    is untouched contact physics, and entry is the task: threading a 0.85 mm
    window at the end of a bimanual carry. What this trades away, deliberately
    and on the record, is post-release retention as a *contact* phenomenon;
    a success here claims "seated to detent depth", not "held by friction".

    Thresholds are stated in the receptacle's own frame, mesh-free (the suite
    passes them from its asserted stage constants).
    """

    name: str = "detent"
    entity: str = "usbc_detent"
    plug_object: str = "plug"
    port_object: str = "bulkhead"
    mouth_x: float = -0.0045
    """The mouth plane along the receptacle's local barrel axis [m,
    post-scale]: ``stage.BULK.mouth_x * SCALE``, restated because a spec must
    build without the mesh loaded."""
    tip_z: float = 0.0133
    """The tab tip along the plug's local insertion axis [m, post-scale]:
    ``stage.REF.plug_tab_tip_z``."""
    engage_depth: float = 0.0115
    """Click depth [m]: 1.1 mm past the goal's 10.4 mm depth gate, 0.8 mm
    under the shallowest seat the gt graph measures (12.3 mm), so every real
    seat crosses it and nothing short of the gate ever latches."""
    engage_lat: float = 0.0015
    """On-axis gate [m]: the slot's own aperture bounds a tab this deep to
    ~1.2 mm of lateral offset, so this excludes nothing physical -- it exists
    to keep the criterion meaningful if the geometry ever loosens."""
    engage_tilt: float = 0.0523
    """Straightness gate [rad]: 3 degrees, against 0.25-0.36 measured at
    seat. Well under the goal's 5-degree clause: a plug hanging at the old
    droop angles can never latch."""


@register_entity("usbc_detent")
class EntityUsbcDetent(Entity):
    """One kernel on the rigid-force hook, and four device words of state."""

    def __init__(self, spec: SpecUsbcDetent, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.spec: SpecUsbcDetent = spec
        self._pair: wp.array | None = None
        self._latched: wp.array | None = None
        self._rel: wp.array | None = None
        self._captured_depth: wp.array | None = None
        self._params: wp.array | None = None
        self._gains: wp.array | None = None

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        labels = list(builder.body_label)
        self._plug_body = label_index(labels, self.spec.plug_object)
        self._port_body = label_index(labels, self.spec.port_object)

    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        import math  # noqa: PLC0415

        device = model.device
        spec = self.spec
        self._pair = wp.array([self._plug_body, self._port_body], dtype=int, device=device)
        self._latched = wp.zeros(1, dtype=int, device=device)
        self._rel = wp.zeros(1, dtype=wp.transform, device=device)
        self._captured_depth = wp.zeros(1, dtype=float, device=device)
        self._params = wp.array(
            [spec.mouth_x, spec.tip_z, spec.engage_depth, spec.engage_lat, math.cos(spec.engage_tilt)],
            dtype=float,
            device=device,
        )
        self._gains = wp.array(list(DETENT_GAINS), dtype=float, device=device)

    def contribute_hooks(self, hooks) -> None:
        model = self.model

        def hold(state, dt) -> None:
            del dt
            wp.launch(
                _hold_mate,
                dim=1,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    state.body_f,
                    self._pair,
                    self._latched,
                    self._rel,
                    self._captured_depth,
                    self._params,
                    self._gains,
                ],
                device=model.device,
            )

        hooks.rigid_force.append(hold)

    def reset(self, pipeline, episode) -> None:
        """Unclick. The captured pose is meaningless once the flag is down."""
        del pipeline, episode
        if self._latched is not None:
            self._latched.zero_()
        if self._captured_depth is not None:
            self._captured_depth.zero_()


# ---------------------------------------------------------------- the records
#
# What a document names. Each factory returns the spec its record's ``build``
# promises, placed from the same asserted stage constants that place the
# bore, so the chain roots where the boot actually is and the detent and the
# goal cannot disagree about where "seated" is.


def connector_cables() -> SpecUsbcCables:
    """A cable rooted on each connector, at the reference sims' defaults.

    They are not decoration: their weight is the honest load an insertion
    carries, and the plug's chain is what pulls on a released mate. They are
    also why the document stages doubled constraint ceilings.
    """
    # The chains run along their root's +z: rotate capsule-local +z onto world
    # +y, the reference sims' own spawn direction for both cables.
    run_quat = (-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))
    return SpecUsbcCables(
        scale=SCALE,
        plug_rear_z=REF.plug_body_z0,
        port_root=(*point_from_surface(CELL, bulkhead_boot_tip()), *run_quat),
        # Root the rear chain on the free receptacle, by the name the document
        # gives it. Left unstated the chain roots on the world, which is right
        # for a bolted receptacle and is, here, the bug where the carried
        # bulkhead tears visibly off its own cable.
        port_object="bulkhead",
    )


def bulkhead_detent() -> SpecUsbcDetent:
    """The receptacle's retention, at the click point the goal measures against.

    The mouth plane and tab tip the insertion atom reads, post-scale, from the
    same asserted constants that place the bore.
    """
    return SpecUsbcDetent(mouth_x=BULK.mouth_x * SCALE, tip_z=REF.plug_tab_tip_z)


CABLES = register_asset(
    AssetRecord(
        name="usbc_cables",
        summary=(
            "Two articulated capsule-chain cables: 180 mm off the plug's rear face and "
            "50 mm off the bulkhead's boot, each riding the connector it roots on."
        ),
        tags=("cable", "cord", "chain"),
        features=(),
        sim=SimRequirements(
            domain="rigid",
            contacts=contacts_with("surface", "fixtures", "objects", "hands"),
            notes=(
                "Alternating-hinge chains solved natively by MuJoCo in the one island every body "
                "of this cell is in; the links keep MuJoCo's soft default solref on purpose."
            ),
        ),
        provenance=(
            "Ported from the newton_tasks bimanual MJC sims' cable machinery; 450 kg/m^3 is the "
            "bulkhead sim's density for a real 6 mm rubber jacket, and the revolute-per-link "
            "articulation is this port's (see EntityUsbcCables)."
        ),
        build=connector_cables,
    )
)


def evaluation_plug_cable() -> SpecUsbcCables:
    """The evaluated 240 mm, 16-link plug cable with no receptacle tail."""
    return SpecUsbcCables(
        scale=SCALE,
        plug_rear_z=REF.plug_body_z0,
        plug_length=0.12,
        plug_links=16,
        port_links=0,
    )


def evaluation_bulkhead_cables() -> SpecUsbcCables:
    """The same plug cable plus the bulkhead's physical rear cable."""
    run_quat = (-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))
    return SpecUsbcCables(
        scale=SCALE,
        plug_rear_z=REF.plug_body_z0,
        plug_length=0.12,
        plug_links=16,
        port_root=(*point_from_surface(CELL, bulkhead_boot_tip()), *run_quat),
        port_object="bulkhead",
    )


PLUG_CABLE = register_asset(
    replace(
        CABLES,
        name="usbc_plug_cable",
        summary="A 240 mm, 16-link cable physically rooted on the USB-C plug.",
        build=evaluation_plug_cable,
    )
)

EVALUATION_CABLES = register_asset(
    replace(
        CABLES,
        name="usbc_evaluation_cables",
        summary=(
            "The evaluated 240 mm plug cable and the free bulkhead's articulated rear cable."
        ),
        build=evaluation_bulkhead_cables,
    )
)

DETENT = register_asset(
    AssetRecord(
        name="usbc_detent",
        summary=(
            "The receptacle's retention spring, as a criterion-gated weld: at click depth, on "
            "axis and straight, the plug and bulkhead become one assembly until reset."
        ),
        tags=("detent", "spring", "mechanism"),
        features=(),
        sim=SimRequirements(
            domain="rigid",
            contributes_hooks=("rigid_force",),
            notes="One kernel on the rigid-force hook and four device words of state; owns no body.",
        ),
        provenance=(
            "Stands in for the shield spring this contact stack cannot express; SpecUsbcDetent "
            "records the twelve refuted geometric variants. A success claims seated-to-click-depth."
        ),
        build=bulkhead_detent,
    )
)
