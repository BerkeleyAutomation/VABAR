"""Physical-frame and goal checks against AUTOLab main 34c6f8d0."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("torch")

import torch
from scipy.spatial.transform import Rotation

from ib_support import ASSETS_ROOT, ENVIRONMENTS_ROOT, load_scene, scene_params


def scene(name):
    return load_scene(f"tool_hanging/{name}_easy.yaml")


@pytest.mark.parametrize("name", ["scissors", "screwdriver", "plier"])
def test_rss_robot_mounts_and_home(name):
    config = scene(name)
    robot = config["embodiment"]["params"]
    for side, y in [("left", 0.17), ("right", -0.17)]:
        assert robot[f"{side}_mount_position"] == pytest.approx([0.3025, y, 0.81])
        assert robot[f"{side}_initial_joint_pose"] == pytest.approx(
            [0, 1.047, 1.047, 0, 0, 0]
        )
    assert config["background"]["params"]["initial_pose"]["position_xyz"][2] == 0.728


@pytest.mark.parametrize(
    "name,fixture", [("scissors", "scissor_hook_2")]
)
def test_reused_hook_shank_matches_rss_world_frame(name, fixture):
    config = scene(name)
    target = next(o for o in config["objects"] if o["id"] == fixture)
    # The decomposed hook preserves its cavity under native MuJoCo contacts.
    assert target["registry_name"] == "vabar__wrench_hook_2"
    pose = next(o for o in config["objects"] if o["id"] == fixture)["params"][
        "initial_pose"
    ]
    rod = config["task"]["subtasks"][0]["params"]["goals"][0]["rod"]
    rotation = Rotation.from_quat(pose["rotation_xyzw"])
    start = rotation.apply(rod["start_xyz"]) + pose["position_xyz"]
    end = rotation.apply(rod["end_xyz"]) + pose["position_xyz"]
    # records.py pegboard_hook: local X faces the robot; Y becomes world Z.
    assert start == pytest.approx([0.8189, -0.000001525, 1.132715762])
    assert end == pytest.approx([0.88, -0.000001525, 1.132715762])


def test_scissor_bows_follow_reused_mesh_frame():
    config = scene("scissors")
    goal = config["task"]["subtasks"][0]["params"]["goals"][0]
    rotation = next(
        r["params"]
        for r in config["relations"]
        if r["kind"] == "rotate_around_solution"
    )
    local_to_world = Rotation.from_euler(
        "xyz", [rotation["roll_rad"], 0, rotation["yaw_rad"]]
    )
    for loop, rss_point in zip(
        [goal["loop"], *goal["alternative_loops"]],
        [[0.0217, -0.0385, 0], [-0.02183, -0.03853, 0]],
        strict=True,
    ):
        assert local_to_world.apply(loop["center_xyz"]) == pytest.approx(
            Rotation.from_euler("z", np.pi / 2).apply(rss_point)
        )
        assert loop["radius_m"] == 0.024 + 0.005


def test_box_scores_rotated_tip_in_world_axis_region():
    from isaaclab_arena_vabar.tasks.tool_hanging.terminations import (
        HangingGoalSpec,
        _contained_in_fixture_box,
    )

    goal = HangingGoalSpec(
        "driver",
        "box",
        "driver",
        "box",
        "sensor",
        containment_min_xyz=(-0.067, -0.1252, -0.01),
        containment_max_xyz=(0.016, -0.0212, 0.141),
        containment_point_xyz=(0.175, 0, 0),
    )
    # Vertical screwdriver: its origin lies above the region, its tip inside.
    fixture = torch.tensor([[0.88, 0.0732, 0.85, 0, 0, 1, 0]], dtype=torch.float64)
    q = Rotation.from_euler("y", np.pi / 2).as_quat()
    tool = torch.tensor([[0.85, 0, 1.05, *q]], dtype=torch.float64)
    assert _contained_in_fixture_box(goal, tool, fixture).item()
    tool[:, 0] = 0.7
    assert not _contained_in_fixture_box(goal, tool, fixture).item()


def test_screwdriver_box_is_open_and_has_a_floor():
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(ASSETS_ROOT / "vabar__screwdriver_box/box.usda"))
    bounds = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            bounds.append(
                UsdGeom.BBoxCache(0, ["default"])
                .ComputeWorldBound(prim)
                .ComputeAlignedRange()
            )
    # Collider count parity with the RSS version is checked in test_tool_hanging_level2.
    assert bounds

    def solid(point):
        return any(
            all(
                lo <= x <= hi
                for lo, x, hi in zip(b.GetMin(), point, b.GetMax(), strict=True)
            )
            for b in bounds
        )

    assert solid((0.0325, 0.0732, 0.002))
    assert not solid((0.0325, 0.0732, 0.03))
    assert not solid((0.0325, 0.0732, 0.12))


def test_easy_tool_hanging_placement_regions_match_declared_tasks() -> None:
    for path in sorted((ENVIRONMENTS_ROOT / "tool_hanging").glob("*_easy.yaml")):
        config = scene(path.stem.removesuffix("_easy"))
        (tool_name,) = scene_params(config)["tools"]
        relations = [r for r in config["relations"] if r["subject"] == tool_name]

        assert any(
            r["kind"] == "on" and r["reference"] == "table" for r in relations
        ), path.stem
        assert all(r["kind"] != "at_position" for r in relations), path.stem
        limits = next(r["params"] for r in relations if r["kind"] == "position_limits")
        # The RSS version draws each planar offset uniformly in [-0.02, 0.02].
        assert limits["x_max"] - limits["x_min"] == pytest.approx(0.04), path.stem
        assert limits["y_max"] - limits["y_min"] == pytest.approx(0.04), path.stem
        if path.stem == "wrench_easy":
            rotation = next(
                r["params"] for r in relations if r["kind"] == "rotate_around_solution"
            )
            assert rotation["yaw_rad"] == pytest.approx(math.pi)


def test_plier_easy_restores_scoped_left_fingertip_friction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isaaclab_arena_vabar.environments.tool_hanging.registration import (
        ENVIRONMENTS as TOOL_HANGING_ENVIRONMENTS,
    )
    from isaaclab_arena_vabar.environments.tool_hanging.environment import (
        _apply_left_fingertip_torsional_friction,
    )
    from isaaclab_newton.physics import NewtonManager

    descriptor = next(
        item
        for item in TOOL_HANGING_ENVIRONMENTS
        if item.environment_name == "vabar_tool_hanging__plier_easy"
    )
    assert descriptor.left_fingertip_torsional_friction == pytest.approx(0.5)
    fingertips = [
        f"/World/envs/env_0/LeftRobot/arm/{finger}/Sphere{suffix}"
        for finger in ("lf_down", "rf_down")
        for suffix in ("", "_1", "_2", "_3", "_4", "_5")
    ]
    easy_labels = [
        "/World/envs/env_0/plier_1/mesh",
        "/World/envs/env_0/plier_support_left/mesh",
        *fingertips,
        "/World/envs/env_0/RightRobot/arm/rf_down/Sphere",
    ]
    builder = SimpleNamespace(
        shape_label=easy_labels,
        shape_material_mu_torsional=[0.1] * len(easy_labels),
    )
    monkeypatch.setattr(NewtonManager, "_builder", builder)

    _apply_left_fingertip_torsional_friction(
        0.5,
        descriptor.fingertip_friction_required_shape_tokens,
        descriptor.fingertip_friction_forbidden_shape_tokens,
    )

    assert builder.shape_material_mu_torsional[:2] == pytest.approx([0.1, 0.1])
    assert builder.shape_material_mu_torsional[2:14] == pytest.approx([0.5] * 12)
    assert builder.shape_material_mu_torsional[14] == pytest.approx(0.1)

    vector_fingertips = [
        label.replace("env_0", f"env_{env_id}")
        for env_id in range(2)
        for label in fingertips
    ]
    builder.shape_label = [*easy_labels[:2], *vector_fingertips]
    builder.shape_material_mu_torsional = [0.1] * len(builder.shape_label)
    _apply_left_fingertip_torsional_friction(
        0.5,
        descriptor.fingertip_friction_required_shape_tokens,
        descriptor.fingertip_friction_forbidden_shape_tokens,
    )
    assert builder.shape_material_mu_torsional[2:] == pytest.approx([0.5] * 24)

    builder.shape_label = [
        *easy_labels[:2],
        *fingertips,
        "/World/envs/env_1/table/collision",
    ]
    builder.shape_material_mu_torsional = [0.1] * len(builder.shape_label)
    with pytest.raises(AssertionError, match=r"invalid counts \{1: 0\}"):
        _apply_left_fingertip_torsional_friction(
            0.5,
            descriptor.fingertip_friction_required_shape_tokens,
            descriptor.fingertip_friction_forbidden_shape_tokens,
        )

    builder.shape_label = [
        "/World/envs/env_0/plier_1/mesh",
        "/World/envs/env_0/plier_2/mesh",
        "/World/envs/env_0/plier_support_left/mesh",
        "/World/envs/env_0/plier_support_right/mesh",
        *fingertips,
    ]
    builder.shape_material_mu_torsional = [0.1] * len(builder.shape_label)
    _apply_left_fingertip_torsional_friction(
        0.5,
        descriptor.fingertip_friction_required_shape_tokens,
        descriptor.fingertip_friction_forbidden_shape_tokens,
    )
    assert builder.shape_material_mu_torsional == pytest.approx(
        [0.1] * len(builder.shape_label)
    )



def _finger_collider_labels(env_id: int, robot: str) -> list[str]:
    """The crank hand's finger colliders as Newton labels them in a built scene."""
    root = f"/World/envs/env_{env_id}/{robot}/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"
    labels = []
    for side, short in (("left", "lf"), ("right", "rf")):
        rot = f"{root}/link_{side}_finger/{short}_rot"
        labels += [f"{rot}/Capsule", f"{rot}/Capsule_1"]
        down = f"{rot}/{short}_down"
        labels += [f"{down}/Sphere{suffix}" for suffix in ("", "_1", "_2", "_3", "_4", "_5")]
        labels += [f"{down}/Box", f"{down}/Box_1", f"{down}/Capsule", f"{down}/Capsule_1", f"{down}/Capsule_2"]
    return labels


def test_tool_hanging_finger_colliders_take_contact_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finger contacts use the hand's own solref/solimp, not an average with the tool's."""
    from isaaclab_arena_vabar.environments.tool_hanging.environment import (
        FINGER_CONTACT_PRIORITY,
        _apply_finger_contact_priority,
    )
    from isaaclab_newton.physics import NewtonManager
    from newton import ShapeFlags

    collide = int(ShapeFlags.COLLIDE_SHAPES)
    fingers = [
        label
        for env_id in range(2)
        for robot in ("LeftRobot", "RightRobot")
        for label in _finger_collider_labels(env_id, robot)
    ]
    others = [
        "/World/envs/env_0/LeftRobot/Geometry/arm/link_1/link_2/Capsule",
        "/World/envs/env_0/RightRobot/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6/Capsule",
        "/World/envs/env_0/scissors_1/Geometry/Visual_obj_00/StoredConvexCollision/ConvexHull_00",
        "/World/envs/env_0/scissor_hook_1/fixture/StoredConvexCollision/Hull_000",
    ]
    visual = "/World/envs/env_0/LeftRobot/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6/link_left_finger/lf_rot/visual"
    labels = [*others, *fingers, visual]
    builder = SimpleNamespace(
        shape_label=labels,
        shape_flags=[collide] * (len(labels) - 1) + [0],
        custom_attributes={"mujoco:geom_priority": SimpleNamespace(values=None)},
    )
    monkeypatch.setattr(NewtonManager, "_builder", builder)

    _apply_finger_contact_priority()

    values = builder.custom_attributes["mujoco:geom_priority"].values
    assert FINGER_CONTACT_PRIORITY > 0
    assert sorted(values) == list(range(len(others), len(others) + len(fingers)))
    assert set(values.values()) == {FINGER_CONTACT_PRIORITY}

    # A scene without scissors (a wrench, say) keeps MuJoCo's averaged finger contact.
    builder.shape_label = [label for label in labels if "/scissors_" not in label]
    builder.shape_flags = [collide] * len(builder.shape_label)
    builder.custom_attributes["mujoco:geom_priority"].values = None
    _apply_finger_contact_priority()
    assert builder.custom_attributes["mujoco:geom_priority"].values is None

    # A hand with a missing collider is refused rather than half prioritised.
    builder.shape_label = [*others, *fingers[1:]]
    builder.shape_flags = [collide] * len(builder.shape_label)
    builder.custom_attributes["mujoco:geom_priority"].values = None
    with pytest.raises(AssertionError, match="26 finger colliders"):
        _apply_finger_contact_priority()


def test_finger_contact_priority_registers_once_at_model_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isaaclab.physics import PhysicsEvent, PhysicsManager
    from isaaclab_arena_vabar.environments.tool_hanging.environment import (
        _apply_finger_contact_priority,
        _register_finger_contact_priority,
    )
    from isaaclab_newton.physics import NewtonManager

    callbacks = {}

    def register_callback(callback, event, *, name, wrap_weak_ref):
        callbacks[len(callbacks)] = (callback, event, wrap_weak_ref, name)

    monkeypatch.setattr(PhysicsManager, "_callbacks", callbacks)
    monkeypatch.setattr(NewtonManager, "register_callback", register_callback)

    from isaaclab_arena_vabar.environments.tool_hanging.registration import (
        ENVIRONMENTS as TOOL_HANGING_ENVIRONMENTS,
    )

    by_name = {item.environment_name: item for item in TOOL_HANGING_ENVIRONMENTS}
    for name in ("wrench_easy", "wrench_medium", "plier_easy", "screwdriver_medium"):
        _register_finger_contact_priority(by_name[f"vabar_tool_hanging__{name}"])
    assert callbacks == {}

    scissors = by_name["vabar_tool_hanging__scissors_medium"]
    _register_finger_contact_priority(scissors)
    _register_finger_contact_priority(scissors)

    assert len(callbacks) == 1
    callback, event, _, _ = callbacks[0]
    assert callback is _apply_finger_contact_priority
    assert event == PhysicsEvent.MODEL_INIT


def test_only_scissors_opt_into_finger_contact_priority() -> None:
    from isaaclab_arena_vabar.environments.tool_hanging.registration import (
        ENVIRONMENTS as TOOL_HANGING_ENVIRONMENTS,
    )

    opted = {
        item.environment_name
        for item in TOOL_HANGING_ENVIRONMENTS
        if item.finger_contact_priority
    }
    assert opted == {
        "vabar_tool_hanging__scissors_easy",
        "vabar_tool_hanging__scissors_medium",
    }
