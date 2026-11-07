# SPDX-License-Identifier: Apache-2.0
"""``cable_rod`` on SuperDex: the Cosserat centerline, adapted to what cable_route reads.

The SuperDex candidate of ``tools/fast_tune/cable_route.py``, which registers it as
``register_entity("cable_rod", engine="superdex")`` for an ``sdx:`` candidate and
points the variants' rod spec at ``cable_rod``. It lives here rather than in
``tasks/cable_route/assets`` because SuperDex did not win the sweep
(``docs/validation/fast_cable_route.md``); moving it into the suite and binding it
in ``assets/rod.py`` is all a SuperDex arrangement would need.

:class:`~robosimstudio_materials.rod.superdex.EntityRodSuperdex` builds the same
``RodSpec`` as a centerline of inactive particles integrated by SuperDex on the
CPU, beside the MuJoCo arms. These things the suite relies on are the capsule
chain's and not the library's foreign-engine rod's, and this class closes them
(plus two library seams, :func:`superdex_compat` and
:func:`velocity_reads_entity_first`, and the chain's scene friction and
fingertip filter, below):

* **The spawn shape.** ``PORT_BOARD``/``WEAVE_RIGID``/``CABLE_BOW`` draw the
  rod's bow, shift, harmonic and yaw as unit fractions in
  ``episode.overrides``; the capsule chain bends its straight spawn through
  :func:`~robosimstudio_materials.rod.entity._bow_cable`. The library's
  centerline lays the straight :meth:`RodSpec.rest_polyline` and lists the four
  ranges in ``unread_fields``. Here the same function bends the same straight
  nodes, so an episode's rod starts in the shape the capsule chain starts in --
  node for node, since the chain's capsule endpoints *are* these nodes. The
  stress-free shape is the spawn, as it is for the chain (whose segment frames
  are turned with the curve so the bow does not snap).
* **The stiffness.** A Cosserat rod wants ``EA``/``EI``/``GJ`` and the library
  derives them from ``young_modulus`` (5 MPa). This suite's rod states VBD joint
  gains instead (`tasks/cable_route/assets/cable.py:weave_bend` thins the reference's bend to
  material parity at 3.5 mm), and 5 MPa at 3.5 mm is ~44x the bend those gains
  mean. A rod joint's gain is a section stiffness over the pitch it spans, so
  ``EI = bend_ke * pitch``, ``EA = stretch_ke * pitch`` and
  ``GJ = twist_ke * pitch`` -- the same rod, in the engine's units.
* **The reader.** The routine and the cable tools call
  ``cable.positions(env.state.body_q.numpy())`` -- the capsule centres, one per
  segment. A centerline has no bodies; handed a ``body_q`` array this answers
  the same thing off the live particles (the midpoints of consecutive nodes),
  and ``hold_segment`` is the engine pin :meth:`hold` asks for.

What it does not close is written in ``docs/validation/fast_cable_route.md``:
one world only (no per-world pins), and one-way coupling when the arrangement
stages ``feedback="off"``.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from robosimstudio.specs.physics import HostActor
from robosimstudio_materials.rod.entity import _bow_cable
from robosimstudio_materials.rod.superdex import EntityRodSuperdex

__all__ = ["EntityCableSuperdex", "superdex_compat"]


_RENAMED = {
    # RoboSimStudio d9b2e44's rod builder name -> superdex-physics 1.0.1's.
    "use_visual_mesh_contact": "use_contact_skin",
    "visual_mesh_contact_element_type": "contact_skin_element_type",
}


def superdex_compat() -> None:
    """Alias the two ``RodActorParams`` fields superdex-physics 1.0.1 renamed, once per process.

    The pinned library's ``build_rod_actor`` sets ``use_visual_mesh_contact``
    (and ``visual_mesh_contact_element_type`` when ``skin_samples`` is stated);
    the 1.0.1 wheel calls them ``use_contact_skin`` and
    ``contact_skin_element_type`` and refuses the old names with an
    ``AttributeError`` at actor build. A property on the binding class maps the
    old spelling onto the new one. Nothing else is touched, and a wheel that
    still has the old names is left alone.
    """
    try:
        from superdex import physics
    except ImportError:
        return
    params = physics.experimental.RodActorParams
    for old, new in _RENAMED.items():
        if not hasattr(params, old) and hasattr(params, new):
            setattr(
                params,
                old,
                property(lambda self, _n=new: getattr(self, _n), lambda self, v, _n=new: setattr(self, _n, v)),
            )


def velocity_reads_entity_first() -> None:
    """Let ``(speed cable)`` read a bodiless rod: ask the entity before ``body_qd``.

    The goal binder's ``rod``/``velocity`` reader indexes ``body_qd`` by the
    chain's bodies unconditionally; its ``points`` reader already asks the entity
    first (``Entity.read``), and ``Entity.read``'s own docstring names the other
    forms as "the same guess waiting for the same second implementation". This
    is that second implementation: the reader is wrapped so an entity that
    answers ``read("velocity", ...)`` is used, and every other rod -- the capsule
    chain answers ``None`` -- reads exactly as before. Idempotent.
    """
    from robosimstudio_bench.goals import bind

    key = ("rod", "", "velocity")
    original = bind.READERS.get(key)
    if original is None or getattr(original, "entity_first", False):
        return

    def entity_first(snapshot, operand):
        got = bind._entity_reads(snapshot, operand, "velocity")
        return np.asarray(got, dtype=np.float64) if got is not None else original(snapshot, operand)

    entity_first.entity_first = True
    entity_first.__doc__ = original.__doc__
    bind.READERS[key] = entity_first


class EntityCableSuperdex(EntityRodSuperdex):
    """``cable_rod`` as a SuperDex Cosserat rod, starting where the capsule chain starts."""

    # The spawn ranges are read here; the rest of the library's list still holds.
    unread_fields = tuple(
        f for f in EntityRodSuperdex.unread_fields if f not in ("bow_range", "shift_range", "wave_range", "yaw_range")
    )

    def __init__(self, spec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self._spawned: np.ndarray | None = None
        self._pipeline = None

    # ------------------------------------------------------------------ spawn
    def _spawn_nodes(self, episode) -> np.ndarray:
        """This episode's nodes in world: the straight rest polyline, bent as the capsule chain bends it."""
        straight = super().nodes()
        if self.spec.closed or self.spec.nodes is not None:
            return straight  # the chain bends only a straight spawn
        quats = [wp.quat_identity()] * (len(straight) - 1)
        points, _ = _bow_cable([wp.vec3(*[float(v) for v in p]) for p in straight], quats, self.spec, episode)
        return np.asarray([[float(c) for c in p] for p in points], dtype=np.float64)

    def nodes(self) -> np.ndarray:
        return self._spawned if self._spawned is not None else super().nodes()

    def build(self, builder, episode) -> None:
        superdex_compat()  # before the composer builds the actor
        velocity_reads_entity_first()  # before the goal binds
        self._spawned = self._spawn_nodes(episode)
        self._filter_blades_from_scenery(builder)
        super().build(builder, episode)

    def _blade_labels(self) -> tuple[set[str], set[str]]:
        """``(tips, tips + palms)`` off the hands' declarations, as ``EntityRod._hand_body_labels`` reads them."""
        embodiment = getattr(self, "embodiment", None)
        if embodiment is None:
            return set(), set()
        wrists = {arm.name: arm.ee_body_name for arm in embodiment.spec.arms}
        tips: set[str] = set()
        palms: set[str] = set()
        for name, hand in (embodiment.hands or {}).items():
            if hand is None:
                continue
            for side in hand.finger_bodies:
                tips.update(label for label in side if label)
            palm = hand.tcp_body or wrists.get(name, "")
            if palm:
                palms.add(palm)
        return tips, tips | palms

    def _filter_blades_from_scenery(self, builder) -> None:
        """The capsule chain's fingertip-vs-scenery filter, which the library's centerline does not install.

        ``RodSpec.filter_fingertip_scenery`` (on for the weaves, off for the
        contact-only terminated tiers) lets the blades pass through the static
        shapes so a grasp can put the pads round the rod's equator on the bench
        -- ``routine.PAD_FLOOR_CLEAR`` is zero on that promise. The chain
        installs it in ``_decompose_working_tips``; the centerline lists the
        field as unread, so on SuperDex the hand stood on the bench above the
        rod's equator. Same pairs, same rule: every blade shape against every
        static shape.
        """
        if not self.spec.filter_fingertip_scenery:
            return
        _, blades = self._blade_labels()
        if not blades:
            return
        labels = list(builder.body_label)
        blade_bodies = {b for b in range(builder.body_count) if labels[b].split("/")[-1] in blades}
        statics = [t for t in range(builder.shape_count) if int(builder.shape_body[t]) < 0]
        for s in range(builder.shape_count):
            if int(builder.shape_body[s]) in blade_bodies:
                for t in statics:
                    builder.add_shape_collision_filter_pair(s, t)

    # -------------------------------------------------------------- stiffness
    def host_actor(self) -> HostActor:
        actor = super().host_actor()
        spec = self.spec
        if spec.closed or int(spec.segments) < 1:
            return actor
        pitch = float(spec.length) / int(spec.segments)
        gains = spec.drive_gains()
        params = dict(actor.params)
        params["axial_stiffness"] = (
            float(spec.axial_stiffness) if spec.axial_stiffness is not None else (float(gains["stretch_ke"]) * pitch)
        )
        bend = float(gains["bend_ke"]) * pitch
        params["flexural_stiffness"] = [bend, bend]
        params["torsional_stiffness"] = float(gains["twist_ke"]) * pitch
        return HostActor(kind=actor.kind, params=params)

    # -------------------------------------------------------------- materials
    def apply_model_materials(self, model) -> None:
        """The capsule chain's scene-friction pass, on the shapes the engine mirrors.

        ``EntityRod.apply_model_materials`` writes every shape to ``mu_base``,
        the fingertips to ``mu_gripper`` and the bench to ``mu_surface`` when
        the spec asks for ``scene_friction`` -- and the library's centerline
        lists that field as unread, so on SuperDex every shape kept 1.0. The
        mirrors take their friction off ``shape_material_mu`` when the bridge
        opens (this phase runs before it), and MuJoCo reads the same array for
        the arms against the bench, so writing the reference's values here is
        parity on both islands.
        """
        super().apply_model_materials(model)
        spec = self.spec
        if not spec.scene_friction:
            return
        mu = model.shape_material_mu.numpy()
        labels = list(model.shape_label)
        tips, _ = self._blade_labels()
        body_labels = list(model.body_label)
        tip_bodies = {b for b in range(model.body_count) if body_labels[b].split("/")[-1] in tips}
        shape_body = model.shape_body.numpy()
        for s in range(model.shape_count):
            if int(shape_body[s]) in tip_bodies:
                mu[s] = float(spec.mu_gripper)
            elif labels[s].split("/")[-1] in spec.surface_labels:
                mu[s] = float(spec.mu_surface)
            else:
                mu[s] = float(spec.mu_base)
        model.shape_material_mu.assign(mu)

    # ---------------------------------------------------------------- readers
    def on_scene_ready(self, pipeline) -> None:
        super().on_scene_ready(pipeline)
        self._pipeline = pipeline

    def _live_nodes(self) -> np.ndarray:
        state = getattr(self._pipeline, "state_0", None)
        if state is None:
            return super().positions(None)
        return super().positions(state)

    def positions(self, source=None) -> np.ndarray:
        """Node positions for a state or particle array; the capsule chain's segment centres for a ``body_q``.

        Callers written against the capsule chain hand a ``[bodies, 7]`` pose
        array. This rod has no bodies in it, so that call is answered off the
        live particles, as the chain would answer it: one point per segment, at
        its middle.
        """
        if isinstance(source, np.ndarray) and source.ndim == 2 and source.shape[1] == 7:
            nodes = self._live_nodes()
            return 0.5 * (nodes[:-1] + nodes[1:]) if len(nodes) > 1 else nodes
        return super().positions(source)

    def read(self, form: str, snapshot):
        """``points`` as the library's centerline answers it, and ``velocity`` off the nodes.

        ``[n, 6]`` like every velocity reader (linear, then a zero angular half),
        one row per node -- see :func:`velocity_reads_entity_first`.
        """
        if form != "velocity":
            return super().read(form, snapshot)
        qd = None
        probes = getattr(snapshot, "particles", None)
        if probes is not None and probes.get(self.name) is not None:
            velocities = getattr(probes.get(self.name), "velocities", None)
            if velocities is not None and len(velocities):
                qd = np.asarray(velocities, dtype=np.float64)[self.particles]
        if qd is None:
            raw = getattr(snapshot, "particle_qd", None)
            if raw is None or not len(raw) or not len(self.particles):
                return None
            qd = np.asarray(raw, dtype=np.float64)[self.particles]
        return np.hstack([qd, np.zeros_like(qd)])

    def hold_segment(self, on: bool, state=None, *, arm: str = "right", slot: int = 0, at=None, world: int = 0) -> int:
        """The capsule chain's latch call, answered with the engine pin :meth:`hold` takes."""
        if on and not self.spec.allow_grasp_latch:
            raise RuntimeError("This cable permits contact/friction grasps only; artificial grasp latches are disabled")
        return self.hold(on, state, arm=arm, slot=slot, at=at, world=world)
