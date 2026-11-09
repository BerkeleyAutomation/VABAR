# SPDX-License-Identifier: Apache-2.0
"""USB-C jaw calibration and contact parameters.

Evaluated scenes retain the complete imported finger colliders and tune their
friction. A throwaway kinematic model measures the jaw command calibration;
it does not change live object dynamics. Legacy variants may explicitly use
trim_fingertips=True to replace the colliders with blade boxes. That option is
disabled for Easy, Medium and the bulkhead scene.

Fit contacts use MuJoCo solref (0.004, 1.0) and solimp
(0.95, 0.999, 0.0005, 0.5, 2). These are ordinary contact parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import newton
import numpy as np
import warp as wp

from robosimstudio.assetlib import AssetRecord, register_asset
from robosimstudio.core.math import quat_to_mat
from robosimstudio.domains.static import EntityStatic
from robosimstudio.scene.entity import Entity, label_index
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs import SimRequirements, contacts_with

__all__ = [
    "CONTACT_RIG",
    "RIG",
    "PortUsbc",
    "RigUsbc",
    "SpecUsbcRig",
    "contact_insertion_rig",
    "insertion_rig",
]

SOLREF = (0.004, 1.0)
"""MuJoCo solref (timeconst, dampratio) for the fit-critical shapes."""
SOLIMP = (0.95, 0.999, 0.0005, 0.5, 2.0)
"""MuJoCo solimp (d0, dmax, width, midpoint, power) for the same shapes."""
BLADE_MU = 0.35
CONTACT_GAP = 2.0e-4
FINGER_SPIN_FRICTION = 0.002
"""Torsional friction length for a finite finger contact patch [m]."""

INTERFERENCE = 0.002
"""Close command: squeeze 2 mm past the plug's width [m]."""
PINCH_CLEAR = 0.0004
"""Pinch command: stop 0.4 mm a side short of touching [m]."""
INSERT_INTERFERENCE = 0.0006
"""Eased grip for the insertion stroke [m past the width]."""


@dataclass(frozen=True, kw_only=True)
class SpecUsbcRig:
    """Contact model and jaw calibration settings."""

    name: str = "rig"
    entity: str = "usbc_rig"
    arm: str = "right"
    blade_len: float = 0.030
    """How much of the fingertip, measured from its lowest point along the
    gripper's reach axis, keeps collision [m]."""
    trim_fingertips: bool = False
    """Legacy blade-only collision approximation. False preserves the full
    imported fingertip colliders and only calibrates/stiffens their contacts.
    """
    finger_mu: float = 0.35
    """Sliding friction of the full fingertip collision surfaces."""
    insert_interference: float = INSERT_INTERFERENCE
    """Jaw gap reduction during insertion; physical contact sets grip force."""


def _qrot_batch(q_xyzw: np.ndarray, V: np.ndarray) -> np.ndarray:
    return V @ quat_to_mat(q_xyzw).T


@lru_cache(maxsize=4)
def _probe_blades(
    mjcf: str,
    arm: str,
    blade_len: float,
    joint_names: tuple[str, ...],
    finger_names: tuple[str, ...],
    tip_names: tuple[str, ...],
    ee_body: str,
    home_q: tuple[float, ...],
    grip_open: float,
):
    """Trim the fingertips and calibrate the jaw, on throwaway probe models.

    Everything about the robot arrives as an argument, from the registered
    embodiment: the asset path, the joint and body names, the home posture and
    the open command. It used to name the MJCF by literal path and import the
    posture from the embodiment module, which meant re-origining that asset in
    the scene refactor left this calibrating against a file the scene no longer
    builds from -- harmless while the two agreed, and undetectable when they
    stopped.

    Cached on all of it, since the result depends on exactly these and two
    finalizes per compile would otherwise be paid on every episode rebuild.

    Returns ``(blade_meshes, box_centers, box_half, fit_a, fit_b)`` where the
    meshes and centers are keyed by tip body name and everything is in each
    tip's own shape-local frame.
    """
    from newton.solvers import SolverMuJoCo  # noqa: PLC0415

    def build_probe(blade_meshes=None):
        probe = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(probe)
        probe.add_mjcf(mjcf, floating=False, enable_self_collisions=False, verbose=False)
        for j, label in enumerate(probe.joint_label):
            tail = label.split("/")[-1]
            if tail in joint_names and joint_names.index(tail) < len(home_q):
                probe.joint_q[probe.joint_q_start[j]] = home_q[joint_names.index(tail)]
            elif tail in finger_names:
                probe.joint_q[probe.joint_q_start[j]] = grip_open
        if blade_meshes is not None:
            for s in range(probe.shape_count):
                body = int(probe.shape_body[s])
                if body < 0 or probe.shape_source[s] is None:
                    continue
                body_name = probe.body_label[body].split("/")[-1]
                if body_name in blade_meshes:
                    probe.shape_source[s] = blade_meshes[body_name]
        return probe

    def fk(probe):
        pm = probe.finalize()
        ps = pm.state()
        newton.eval_fk(pm, pm.joint_q, pm.joint_qd, ps)
        return pm, ps, ps.body_q.numpy()

    # --- pass 1: cut the blades in the gripper frame at home ----------------
    probe = build_probe()
    pm, _, bq = fk(probe)
    gi = label_index(list(pm.body_label), ee_body)
    gq0, gp0 = bq[gi, 3:7], bq[gi, :3]

    blade_meshes: dict[str, newton.Mesh] = {}
    box_centers: dict[str, np.ndarray] = {}
    halves = []
    for s in range(probe.shape_count):
        body = int(probe.shape_body[s])
        if body < 0 or probe.shape_source[s] is None:
            continue
        body_name = probe.body_label[body].split("/")[-1]
        if body_name not in tip_names:
            continue
        xf = probe.shape_transform[s]
        xp = np.array([xf.p[0], xf.p[1], xf.p[2]], dtype=np.float64)
        xq = np.array([xf.q[0], xf.q[1], xf.q[2], xf.q[3]], dtype=np.float64)
        V = np.asarray(probe.shape_source[s].vertices, dtype=np.float64)
        Vb = _qrot_batch(xq, V) + xp
        Vw = _qrot_batch(bq[body, 3:7], Vb) + bq[body, :3]
        Vg = _qrot_batch(np.append(-gq0[:3], gq0[3]), Vw - gp0)
        keep = Vg[:, 2] <= float(Vg[:, 2].min()) + float(blade_len)
        faces = np.asarray(probe.shape_source[s].indices, dtype=np.int64).reshape(-1, 3)
        fkeep = faces[keep[faces].all(axis=1)]
        used = np.unique(fkeep)
        remap = -np.ones(len(V), dtype=np.int64)
        remap[used] = np.arange(len(used))
        blade_meshes[body_name] = newton.Mesh(
            np.ascontiguousarray(V[used], dtype=np.float32),
            np.ascontiguousarray(remap[fkeep].reshape(-1), dtype=np.int32),
        )
        lo, hi = V[used].min(axis=0), V[used].max(axis=0)
        box_centers[body_name] = 0.5 * (lo + hi)
        halves.append(0.5 * (hi - lo))
    if len(blade_meshes) != 2:
        raise RuntimeError(f"expected 2 fingertip meshes for {arm!r}, got {sorted(blade_meshes)}")
    # One shared half-extent so the two colliders are identical boxes; only the
    # centres differ, in each tip's own shape-local frame.
    box_half = np.maximum(np.max(np.stack(halves), axis=0), 2.0e-4)

    # --- pass 2: measure gap-vs-command on the trimmed colliders ------------
    probe = build_probe(blade_meshes)
    finger_coords = [
        int(probe.joint_q_start[j]) for j, label in enumerate(probe.joint_label) if label.split("/")[-1] in finger_names
    ]
    pm, ps, _ = fk(probe)
    gi = label_index(list(pm.body_label), ee_body)
    close_ax = np.array([0.0, 1.0, 0.0])  # gripper-local closing axis (measured)

    cmds = np.linspace(0.004, 0.040, 7)
    gaps = []
    q0 = pm.joint_q.numpy().copy()
    for c in cmds:
        q = q0.copy()
        for k in finger_coords:
            q[k] = c
        pm.joint_q.assign(wp.array(q, dtype=float))
        newton.eval_fk(pm, pm.joint_q, pm.joint_qd, ps)
        bq = ps.body_q.numpy()
        gq, gp = bq[gi, 3:7], bq[gi, :3]
        lo_side, hi_side = [], []
        for s in range(probe.shape_count):
            body = int(probe.shape_body[s])
            if body < 0 or probe.shape_source[s] is None:
                continue
            if probe.body_label[body].split("/")[-1] not in tip_names:
                continue
            xf = probe.shape_transform[s]
            xp = np.array([xf.p[0], xf.p[1], xf.p[2]], dtype=np.float64)
            xq = np.array([xf.q[0], xf.q[1], xf.q[2], xf.q[3]], dtype=np.float64)
            V = np.asarray(probe.shape_source[s].vertices, dtype=np.float64)
            Vw = _qrot_batch(bq[body, 3:7], _qrot_batch(xq, V) + xp) + bq[body, :3]
            Vg = _qrot_batch(np.append(-gq[:3], gq[3]), Vw - gp)
            proj = Vg @ close_ax
            (hi_side if proj.mean() > 0 else lo_side).append(proj)
        gaps.append(float(np.min(np.concatenate(hi_side)) - np.max(np.concatenate(lo_side))))
    fit_a, fit_b = np.polyfit(cmds, np.asarray(gaps), 1)
    return blade_meshes, box_centers, box_half, float(fit_a), float(fit_b)


@register_entity("usbc_rig")
class RigUsbc(Entity):
    """Build-time surgery plus the grip-command arithmetic that depends on it."""

    spec: SpecUsbcRig

    def __init__(self, spec: SpecUsbcRig, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.fit_a: float = 0.0
        self.fit_b: float = 0.0
        self.blade_shapes: list[int] = []

    # ------------------------------------------------------------- commands
    @property
    def grip_open(self) -> float:
        """This arm's finger command at full open [m], off the embodiment."""
        spec = self.embodiment.spec
        arm = next((a for a in spec.arms if a.name == self.spec.arm), spec.arms[0])
        return float(self.embodiment.hands[arm.name].open_width)

    def cmd_for_gap(self, gap: float) -> float:
        """The finger command whose measured collider gap is ``gap`` [m]."""
        return float(np.clip((float(gap) - self.fit_b) / self.fit_a, 0.0, self.grip_open))

    def grip_commands(self, plug_width: float) -> dict[str, float]:
        """The reference's derived commands for one plug width."""
        return {
            "open": self.grip_open,
            "pinch": self.cmd_for_gap(plug_width + 2.0 * PINCH_CLEAR),
            "close": self.cmd_for_gap(plug_width - INTERFERENCE),
            "insert": self.cmd_for_gap(plug_width - self.spec.insert_interference),
        }

    # -------------------------------------------------------------- phase 1
    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        arm = self.spec.arm
        # Off the registered embodiment, so the calibration and the scene are
        # built from the same asset by construction.
        from robosimstudio.paths import resolve_asset  # noqa: PLC0415

        spec = self.embodiment.spec
        arm_spec = next((a for a in spec.arms if a.name == arm), spec.arms[0])
        # The finger facts come off the hand, the joint and posture facts off the
        # arm. Splitting the probe's arguments this way is the whole point: the
        # calibration is of a *gripper* against a collider, and re-running it for
        # a different hand on the same arm should change only these three.
        hand = self.embodiment.hands.get(arm_spec.name)
        blade_meshes, box_centers, box_half, self.fit_a, self.fit_b = _probe_blades(
            str(resolve_asset(spec.asset)),
            arm,
            float(self.spec.blade_len),
            tuple(arm_spec.joint_names),
            tuple(hand.finger_names),
            tuple(b for side in hand.finger_bodies for b in side),
            hand.tcp_body or arm_spec.ee_body_name,
            tuple(float(v) for v in arm_spec.home_q),
            float(hand.open_width),
        )
        del blade_meshes  # calibration geometry; never installed on the full rig

        # The hand's own pad bodies, per side, rather than f"{arm}_tip_left":
        # that name is the linear_4310's and matches nothing on another gripper.
        tip_bodies = {}
        for side_index, side in enumerate(hand.finger_bodies):
            for name in side:
                tip_bodies[label_index(list(builder.body_label), name)] = (name, side_index)

        if not self.spec.trim_fingertips:
            # The imported housing is visual-only. Enable it on both hands;
            # retain the fingertip convex decomposition (its source mesh must
            # stay visual-only, since its hull bridges the open finger frame).
            housings = {}
            all_fingers = set()
            for h in self.embodiment.hands.values():
                if not h.tcp_body:
                    continue
                fingers = {label_index(list(builder.body_label), b) for side in h.finger_bodies for b in side}
                all_fingers.update(fingers)
                reference = next(
                    s
                    for s in range(builder.shape_count)
                    if int(builder.shape_body[s]) in fingers
                    and int(builder.shape_flags[s]) & int(newton.ShapeFlags.COLLIDE_SHAPES)
                )
                housings[label_index(list(builder.body_label), h.tcp_body)] = reference
            contact_shapes = []
            for shape in range(builder.shape_count):
                label = builder.shape_label[shape].split("/")[-1]
                body = int(builder.shape_body[shape])
                if body in housings and builder.shape_source[shape] is not None:
                    reference = housings[body]
                    builder.shape_flags[shape] |= int(newton.ShapeFlags.COLLIDE_SHAPES)
                    builder.shape_collision_group[shape] = builder.shape_collision_group[reference]
                    # Inherit this arm's existing self-collision exclusions.
                    # Otherwise the newly enabled housing contacts its own
                    # wrist bearing and prevents joint 6 from tracking.
                    filtered = [
                        other
                        for pair in builder._shape_collision_filter_pairs
                        if reference in pair
                        for other in pair
                        if other != reference
                    ]
                    for other in [reference, *filtered]:
                        if other != shape:
                            builder.add_shape_collision_filter_pair(shape, other)
                    contact_shapes.append(shape)
                elif body in all_fingers:
                    builder.shape_material_mu[shape] = self.spec.finger_mu
                    # Full mesh pads can meet a cylinder at a small patch.
                    # Three-dimensional point contact supplies no resistance
                    # to spinning about the closing axis. A finite friction
                    # patch adds that torque through ordinary condim=4 contact.
                    condim = builder.custom_attributes["mujoco:condim"]
                    if condim.values is None:
                        condim.values = {}
                    condim.values[shape] = 4
                    builder.shape_material_mu_torsional[shape] = FINGER_SPIN_FRICTION
                    contact_shapes.append(shape)
                elif label in ("plug", "port", "bulkhead"):
                    contact_shapes.append(shape)
                elif label == "bench":
                    builder.shape_material_mu[shape] = 0.4
                elif label == "table":
                    builder.shape_material_mu[shape] = BLADE_MU
            self._stiffen(builder, contact_shapes)
            print("[usbc_rig] full fingers and housings collide; contact/friction grasping")
            return

        # Fingertip meshes become visual-only; a fitted box per blade collides.
        # The box centre is in the *mesh-vertex* frame, so it composes with the
        # mesh shape's own transform -- dropped, the boxes land a wrist away
        # from the pads and the jaws close through the part without touching.
        stiff_shapes: list[int] = []
        tip_shape_xform = {}
        for s in range(builder.shape_count):
            body = int(builder.shape_body[s])
            if body in tip_bodies and builder.shape_source[s] is not None:
                builder.shape_collision_group[s] = 0
                tip_shape_xform[tip_bodies[body][0]] = builder.shape_transform[s]
        for body, (name, side_index) in tip_bodies.items():
            cfg = builder.default_shape_cfg.copy()
            cfg.density = 0.0
            cfg.mu = self.spec.finger_mu
            cfg.gap = CONTACT_GAP
            # Which side this pad is, from the hand's own ordering rather than
            # from a suffix on its name. finger_bodies is stated per side, so the
            # index *is* the answer and no other gripper has to be named
            # "..._tip_left" for this to work.
            side = "left" if side_index == 0 else "right"
            center = box_centers[name]
            shape = builder.add_shape_box(
                body=body,
                xform=wp.transform_multiply(
                    tip_shape_xform[name],
                    wp.transform(wp.vec3(*[float(v) for v in center]), wp.quat_identity()),
                ),
                hx=float(box_half[0]),
                hy=float(box_half[1]),
                hz=float(box_half[2]),
                cfg=cfg,
                label=f"blade_tip_{side}",
            )
            builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~int(newton.ShapeFlags.VISIBLE)
            self.blade_shapes.append(shape)
            stiff_shapes.append(shape)

        # The fit-critical contacts get the stiffened solref/solimp: the plug,
        # whichever receptacle this variant stages, and the blades. The bench and
        # table take the reference's frictions (0.4 / 0.35) in place of the
        # builder default.
        #
        # `bulkhead` IS A RECEPTACLE NAME, and leaving it out was not a choice.
        # Shapes are labelled with their entity's own name
        # (`EntityRigid.build`, `label=self.spec.name`), so this filter is a list
        # of the receptacles the suite ships: `port` is what the welded-block
        # documents call theirs and `bulkhead` is what the braced ones call
        # theirs. With only `port` listed, every variant staging a bulkhead --
        # `bulkhead`, and the evaluated `insert_medium` -- built its mate against
        # a receptacle still on MuJoCo's stock solref 0.02 while the plug going
        # into it had been stiffened to 0.004. A fit is a property of BOTH
        # surfaces, so half a stiffening is a soft bore: the plug settles into a
        # compliant wall and creeps back out of it, which is the ~7.8 mm / ~10
        # deg the detent spec measured across twelve contact variants and
        # attributed to grounding.
        for s in range(builder.shape_count):
            label_tail = builder.shape_label[s].split("/")[-1]
            if label_tail in ("plug", "port", "bulkhead"):
                stiff_shapes.append(s)
            elif label_tail == "bench":
                builder.shape_material_mu[s] = 0.4
            elif label_tail == "table":
                builder.shape_material_mu[s] = BLADE_MU
        self._stiffen(builder, stiff_shapes)
        print(
            f"[usbc_rig] blade box half {np.round(box_half * 1e3, 2)} mm, "
            f"jaw gap = {self.fit_a:.4f}*cmd {self.fit_b * 1e3:+.3f} mm"
        )

    def _stiffen(self, builder: newton.ModelBuilder, shapes: list[int]) -> None:
        """Write the reference's solref/solimp onto ``shapes``, and make MuJoCo read them.

        Newton's MuJoCo solver honours a raw ``mujoco:solref`` only on a shape
        whose ``mujoco:solref_mode`` is ``SOLREF_MODE_RAW``; every other mode
        derives the geom's solref from the shape's own ``ke``/``kd`` instead
        (``convert_solref``), which at this package's 2500/100 defaults lands
        on MuJoCo's stock 0.02/1. The value alone was being written here and
        silently discarded -- measured on the built ``usbc_insert/bulkhead``
        model, every blade and part geom read solref 0.02 while their solimp
        had arrived -- so the mode is set beside it.
        """
        from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW  # noqa: PLC0415
        from newton._src.solvers.mujoco.solver_mujoco import vec5  # noqa: PLC0415

        solref = builder.custom_attributes["mujoco:solref"]
        solref_mode = builder.custom_attributes["mujoco:solref_mode"]
        solimp = builder.custom_attributes["mujoco:geom_solimp"]
        for attr in (solref, solref_mode, solimp):
            if attr.values is None:
                attr.values = {}
        for s in shapes:
            solref.values[s] = wp.vec2(*SOLREF)
            solref_mode.values[s] = SOLREF_MODE_RAW
            solimp.values[s] = vec5(*SOLIMP)


@register_entity("usbc_port")
class PortUsbc(EntityStatic):
    """The receptacle: the vendored STL with the reference's lead-in chamfer.

    A suite-owned static because the generic mesh path loads the file as
    authored, and this one must not be: the reference's published behaviour was
    measured on a chamfered mouth (see ``stage.CHAMFER`` for the double-scale
    it inherits), and a receptacle without the lead-in demands the 0.15 mm
    approach accuracy raw. Material set to the reference's ``mu=0.35`` and the
    2e-4 contact gap the fit runs at.
    """

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        from robosimstudio.domains.static.entity import episode_xform  # noqa: PLC0415

        from .stage import CHAMFER, load_port  # noqa: PLC0415

        spec = self.spec
        mesh, _ = load_port(chamfer=float(getattr(spec, "chamfer", CHAMFER)))
        # Use the same episode pose as the compiler's cavity feature. Otherwise
        # the evaluator moves the bore while collision and RGB-D stay nominal.
        self.xform_now = episode_xform(spec, episode)
        placed = self.placed(self.xform_now)
        cfg = builder.default_shape_cfg.copy()
        cfg.mu = BLADE_MU  # the reference's 0.35, shared by plug, port and blades
        cfg.gap = CONTACT_GAP
        builder.add_shape_mesh(
            body=-1,
            xform=wp.transform(wp.vec3(*[float(v) for v in placed[:3]]), wp.quat(*[float(v) for v in placed[3:7]])),
            mesh=mesh,
            cfg=cfg,
            label=spec.name,
        )
        # Display is applied by Entity.build_into, once the range has closed.


# ------------------------------------------------------------------ the record
def insertion_rig() -> SpecUsbcRig:
    """The rig at the reference sim's defaults, on the arm that inserts."""
    return SpecUsbcRig()


def contact_insertion_rig() -> SpecUsbcRig:
    """Evaluation rig with contact/friction grasping and full hand collision."""
    return SpecUsbcRig(
        trim_fingertips=False,
        finger_mu=8.0,
        insert_interference=0.002,
    )


# The record last, because it names the factory that places the rig -- which
# is what lets ``(:objects rig - usbc_rig)`` in the document build it.
RIG = register_asset(
    AssetRecord(
        name="usbc_rig",
        summary=(
            "The insertion rig: full finger and housing collisions, calibrated jaw gap, "
            "and stiffened fit-critical contacts."
        ),
        tags=("rig", "gripper", "mechanism"),
        features=(),
        sim=SimRequirements(
            domain="rigid",
            contacts=contacts_with("objects", "fixtures"),
            notes=(
                "Owns no body of its own: it edits the embodiment's fingertip shapes and the "
                "plug's and port's contact softness in place, all inside the one MuJoCo island."
            ),
        ),
        provenance=(
            "The reference MJC sim's build-time surgery (--blade-collider 2, solref 0.004/1.0, "
            "solimp 0.95/0.999/0.0005/0.5/2), ported as a suite-owned extras entity."
        ),
        build=insertion_rig,
    )
)

CONTACT_RIG = register_asset(
    replace(
        RIG,
        name="usbc_contact_rig",
        summary="The insertion rig with full hand collision and high-friction contact grasping.",
        build=contact_insertion_rig,
    )
)
