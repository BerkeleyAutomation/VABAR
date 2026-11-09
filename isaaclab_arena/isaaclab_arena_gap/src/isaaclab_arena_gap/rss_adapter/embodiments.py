"""Arena embodiments as RoboSimStudio specs: Arena's asset, RoboSimStudio's registered hand.

Nothing about a hand is restated here. A resident hand is the registered spec
with its own asset and mount removed (Arena's asset already carries the hand),
its body labels mapped to the names Arena's export gives them, and its tool
centre re-expressed in the frame Arena's tool-centre body has.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from robosimstudio.robots.registry import get_embodiment, get_gripper, register_gripper
from robosimstudio.specs import ArmSpec, AssetSpec, EmbodimentSpec

from isaaclab_arena_gap.task_sources import fr3_robotiq_usd_path

FR3_HAND = "robotiq_2f85_panda"
# Arena's Isaac export prefixes the 2F-85's bodies; its joints keep MuJoCo Menagerie's names.
FR3_HAND_LABELS = {"base_mount": "robotiq_base", "left_pad": "robotiq_left_pad", "right_pad": "robotiq_right_pad"}
FR3_HAND_BASE_MOUNT_ORIGIN = (0.0, 0.0, 0.007)
"""Where MuJoCo Menagerie's ``base_mount`` sits in Arena's ``robotiq_base`` frame [m].

The two bodies share a name mapping but not an origin. Arena's export roots the
hand at the coupling's underside, the frame MuJoCo Menagerie's ``2f85.xml`` world is,
and that file stands ``base_mount`` 7 mm above it (``pos="0 0 0.007"``); the
axes agree. The driver pivots show it: 65.7045 mm along ``robotiq_base``'s z in
Arena's USD, 3.8 + 54.904 mm along ``base_mount``'s z in MuJoCo Menagerie. Without
this offset the registered 148.8 mm tool centre lands 7 mm short of the pinch
point, and every grasp drives the fingertips 7 mm deeper than the graph asked.
"""
# The FR3 shares the Panda's chain, so the Panda's kinematics files and home serve it,
# at the FR3's own joint stops (see fr3_kinematics).
FR3_KINEMATICS = "panda_robotiq"


def resident_hand(
    name: str, labels: Mapping[str, str], tcp_body_origin: Sequence[float] = (0.0, 0.0, 0.0)
) -> str:
    """Register RoboSimStudio's hand ``name`` as it sits inside an Arena asset; return its key.

    ``tcp_body_origin`` is where the registered ``tcp_body`` sits in the frame of
    the Arena body it maps to, with the same axes; the tool centre moves with it.
    """
    key = f"{name}@arena"
    try:
        get_gripper(key)
        return key
    except Exception:
        pass
    rss_gripper = get_gripper(name)
    mapped = lambda label: labels.get(label, label)  # noqa: E731
    register_gripper(
        replace(
            rss_gripper,
            name=key,
            asset=None,
            mount_body="",
            mount_xform=None,
            tcp_body=mapped(rss_gripper.tcp_body),
            tcp_offset=tuple(float(a) + float(b) for a, b in zip(rss_gripper.tcp_offset, tcp_body_origin, strict=True)),
            finger_bodies=tuple(tuple(mapped(n) for n in side) for side in rss_gripper.finger_bodies),
        )
    )
    return key


def _unstepped(actuation):
    """Registered gains for a model that never steps: Arena's drive modes are Arena's business."""
    return None if actuation is None else replace(actuation, require_position_mode=False)


def fr3_robotiq(base_pose: Sequence[float], home_q: Sequence[float] | None = None) -> EmbodimentSpec:
    """Arena's FR3 with its 2F-85, as the single arm ``right``, mounted at ``base_pose`` (xyz, xyzw).

    ``home_q`` defaults to AUTOLab's ``panda_robotiq`` home; see :func:`fr3_robotiq_home_like`.
    """
    from robosimstudio.paths import resolve_path

    from isaaclab_arena_gap.rss_adapter.fr3_kinematics import fr3_kinematics_urdf

    kinematics = get_embodiment(FR3_KINEMATICS)
    # The Panda's chain at the FR3's joint stops: the stops describe_arm reports and Arena simulates.
    urdf = str(fr3_kinematics_urdf(resolve_path(kinematics.curobo_urdf)))
    return EmbodimentSpec(
        name="arena_fr3_robotiq",
        asset=AssetSpec(path=str(fr3_robotiq_usd_path())),
        base_offset=tuple(float(v) for v in base_pose),
        collapse_fixed_joints=False,
        arms=(
            ArmSpec(
                name="right",
                joint_names=tuple(f"fr3_joint{i}" for i in range(1, 8)),
                gripper=resident_hand(FR3_HAND, FR3_HAND_LABELS, FR3_HAND_BASE_MOUNT_ORIGIN),
                home_q=tuple(home_q) if home_q is not None else kinematics.arms[0].home_q,
            ),
        ),
        gripper_actuation=_unstepped(get_gripper(FR3_HAND).actuation),
        arm_actuation=_unstepped(kinematics.arm_actuation),
        urdf=urdf,
        curobo_config=kinematics.curobo_config,
        curobo_urdf=urdf,
        curobo_time_scale=kinematics.curobo_time_scale,
        curobo_lock_joint=kinematics.curobo_lock_joint,
    )


def _writable_copy(spec: EmbodimentSpec) -> EmbodimentSpec:
    """``spec`` with its asset folder copied somewhere writable.

    Building an MJCF robot writes a stripped copy of the file beside it, so its
    relative mesh paths still resolve. The evaluator image installs RoboSimStudio's
    assets read-only, so a reference robot is built from a temporary copy.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    from robosimstudio.paths import resolve_asset

    source = Path(resolve_asset(spec.asset))
    folder = Path(tempfile.mkdtemp(prefix=f"rss_{spec.name}_")) / source.parent.name
    # File contents only: a copy that kept the image's read-only modes would refuse the write too.
    shutil.copytree(source.parent, folder, copy_function=shutil.copyfile)
    for directory, _, _ in os.walk(folder):
        os.chmod(directory, 0o755)
    return replace(spec, asset=replace(spec.asset, path=str(folder / source.name)))


def _home_tool_pose(spec: EmbodimentSpec):
    """``spec``'s home tool-centre pose in its own base frame, as (position, xyzw)."""
    import numpy as np
    import robosimstudio as rss
    from scipy.spatial.transform import Rotation

    scene = rss.Scene()
    scene.add(rss.robot(spec))
    sim = scene.build()
    name = next(iter(sim.handles.arms))
    base = sim.state.body_q.numpy()[sim.handles.arms[name].base_body]
    position, quat = sim.arm(name).home_pose
    to_base = Rotation.from_quat(base[3:7]).inv()
    return sim, sim.arm(name), to_base.apply(np.asarray(position) - base[:3]), (to_base * Rotation.from_quat(quat)).as_quat()


def fr3_robotiq_home_like(embodiment: str, base_pose: Sequence[float]) -> tuple[float, ...]:
    """The FR3 posture that puts the 2F-85's tool centre where ``embodiment`` rests its own at home.

    AUTOLab authors some FR3 tasks on another hand (syringe packing on the Franka
    hand), and its graphs derive carry heights from the resting tool pose. Arena
    keeps the 2F-85, so the same home joints would rest its longer tool 5 cm lower;
    matching the tool pose, base-relative, keeps the heights AUTOLab planned with.
    Both hands share RoboSimStudio's tool-frame convention, so the jaws match too.
    """
    import numpy as np

    _, _, position, quat = _home_tool_pose(_writable_copy(get_embodiment(embodiment)))
    sim, arm, _, _ = _home_tool_pose(fr3_robotiq(base_pose))
    seed = np.asarray(sim.model.joint_q.numpy(), dtype=np.float64)
    q = arm.solve_q_full(seed, np.asarray(base_pose[:3]) + position, quat)
    reached, _ = arm.fk_ee(q)
    if np.linalg.norm(np.asarray(reached) - np.asarray(base_pose[:3]) - position) > 1e-3:
        raise RuntimeError(f"no FR3 posture reaches {embodiment}'s home tool pose")
    return tuple(float(v) for v in np.asarray(q)[list(arm.arm_coords)])


YAM_HAND = "yam_linear_arena"
# Arena's default YAM hand is AUTOLab's arena hand; its spec names every part by
# arm, so a per-arm-prefixed asset (see usd.build_bimanual) needs no label mapping.
YAM_KINEMATICS = "yam_tool_hanging"
# The YAM tool-hanging embodiment: DLS IK (no URDF, by AUTOLab's own A/B), cuRobo on the vendor URDF.


def yam_bimanual(
    mounts: Mapping[str, Sequence[float]],
    gripper_variant: str | None = None,
    hand: str = YAM_HAND,
    kinematics: str = YAM_KINEMATICS,
) -> EmbodimentSpec:
    """Arena's two YAMs, ``left`` and ``right``, each at ``mounts[arm]`` (xyz, xyzw), on ``gripper_variant``.

    ``hand`` is the RoboSimStudio hand spec that variant is, and ``kinematics`` the
    AUTOLab embodiment whose IK, cuRobo and home postures the task was authored on.
    Every hand Arena's YAM carries names its parts as RoboSimStudio's spec does
    (``{arm}_tip_left``, ``{arm}_gripper``, ``{arm}_joint7``, ...) once each copy
    is prefixed with its arm, so no label mapping is needed.
    """
    import tempfile
    from pathlib import Path

    from isaaclab_arena_gap.rss_adapter.usd import build_bimanual
    from isaaclab_arena_gap.task_sources import yam_usd_path

    spec = get_embodiment(kinematics)
    usd = build_bimanual(
        yam_usd_path(), Path(tempfile.mkdtemp(prefix="arena_yam_")) / "yam_bimanual.usda", mounts, gripper_variant
    )
    authored = {arm.name: arm for arm in spec.arms}
    return EmbodimentSpec(
        name="arena_yam_bimanual",
        asset=AssetSpec(path=str(usd)),
        collapse_fixed_joints=False,
        arms=tuple(
            ArmSpec(
                name=arm,
                joint_names=tuple(f"{arm}_joint{i}" for i in range(1, 7)),
                # The kinematics' own end body (empty: RoboSimStudio derives it from the hand).
                ee_body_name=authored[arm].ee_body_name,
                gripper=hand,
                home_q=authored[arm].home_q,
            )
            for arm in mounts
        ),
        gripper_actuation=_unstepped(get_gripper(hand).actuation),
        arm_actuation=_unstepped(spec.arm_actuation),
        urdf=spec.urdf,
        ik_iterations=spec.ik_iterations,
        curobo_config=spec.curobo_config,
        curobo_urdf=spec.curobo_urdf,
        curobo_time_scale=spec.curobo_time_scale,
        curobo_lock_joint=spec.curobo_lock_joint,
    )


def hand_calibrations(task, names: Sequence[str]) -> dict:
    """The task entities ``names``, built beside AUTOLab's own robot for ``task``.

    For entities that calibrate the hand rather than populate the world: USB-C's
    ``rig`` measures the jaw command that gives each grip gap by probing the
    hand's colliders in AUTOLab's MJCF robot, which Arena's USD cannot stand in
    for. Arena carries the same hand, so the calibration transfers; nothing in it
    is a pose.
    """
    if not names:
        return {}
    import robosimstudio as rss

    specs = {spec.name: spec for spec in task.entities if spec.name in names}
    missing = sorted(set(names) - set(specs))
    if missing:
        raise KeyError(f"{task.full_name} declares no entity {missing}")
    scene = rss.Scene()
    scene.add(rss.robot(_writable_copy(get_embodiment(task.embodiment))))
    for spec in specs.values():
        scene.add(spec)
    sim = scene.build()
    return {name: sim.scene.by_name[name] for name in names}
