"""Effective YAM camera rigs match AUTOLab's YAM station (b1c9012)."""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy.spatial.transform import Rotation

from ib_support import build_registered_environment

# YAM_STATION_RIG literals: (bases, fov_y, (near, far), eye, look_at, up), where
# bases is the RSS base-pair midpoint the eye is authored against.
_TOOL_HANGING_BASES = (0.3025, 0.0, 0.81)
_CABLE_BASES = (0.2525, 0.0, 0.75)
_YAM_VIEWS = {
    "top_camera": (
        _TOOL_HANGING_BASES,
        55.0,
        (0.01, 4.0),
        (0.3025, 0, 1.668),
        (0.8225, 0, 0.868),
        (1, 0, 0),
    ),
    "side_camera": (
        _TOOL_HANGING_BASES,
        50.0,
        (0.01, 4.0),
        (0.2625, 0.82, 1.868),
        (0.8325, 0, 0.948),
        (0, 0, 1),
    ),
    "cable_camera": (
        _CABLE_BASES,
        50.0,
        (0.01, 2.0),
        (0.9196, -0.06, 1.60),
        (0.5025, -0.06, 0.75),
        (0, 0, 1),
    ),
}
_LINK_SIX_D405 = ((-0.0017, 0.079729, 0.066021), (0.0, -0.42304971, 0.90610647, 0.0))


def _assert_intrinsics(camera, fov_y, clipping):
    assert (camera.width, camera.height) == (1280, 720)
    assert camera.update_period == pytest.approx(1 / 30)
    assert camera.offset.convention == "ros"
    assert camera.update_latest_camera_pose
    assert camera.data_types == ["rgb", "distance_to_image_plane"]
    spawn = camera.spawn
    assert math.degrees(
        2 * math.atan(spawn.vertical_aperture / (2 * spawn.focal_length))
    ) == pytest.approx(fov_y)
    assert camera.width / spawn.horizontal_aperture == pytest.approx(
        camera.height / spawn.vertical_aperture
    )
    assert spawn.clipping_range == pytest.approx(clipping)


def _assert_rig(rig, views, midpoint, wrists):
    assert set(rig.camera_names()) == {*views, *wrists}
    for name, (bases, fov_y, clipping, eye, look_at, up) in views.items():
        camera = getattr(rig, name)
        _assert_intrinsics(camera, fov_y, clipping)
        np.testing.assert_allclose(
            camera.offset.pos, np.subtract(eye, bases) + midpoint, atol=1e-9
        )
        # ROS optical: +z looks at the target, +y (image down) opposes up.
        rotation = Rotation.from_quat(camera.offset.rot)
        forward = np.subtract(look_at, eye) / np.linalg.norm(np.subtract(look_at, eye))
        np.testing.assert_allclose(rotation.apply((0, 0, 1)), forward, atol=1e-9)
        assert rotation.apply((1, 0, 0)) @ up == pytest.approx(0, abs=1e-9)
        assert rotation.apply((0, 1, 0)) @ up < 0
    for name, (suffix, (position, rotation)) in wrists.items():
        camera = getattr(rig, name)
        _assert_intrinsics(camera, 58.0, (0.005, 1.5))
        assert camera.prim_path.rsplit("/", 1)[0].endswith(suffix)
        np.testing.assert_allclose(camera.offset.pos, position, atol=1e-8)
        np.testing.assert_allclose(
            Rotation.from_quat(camera.offset.rot).as_matrix(),
            Rotation.from_quat(rotation).as_matrix(),
            atol=1e-8,
        )


_LINK_SIX = "/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"


@pytest.mark.parametrize(
    "name",
    [
        f"vabar_tool_hanging__{tool}_{level}"
        for tool in ("wrench", "plier", "scissors", "screwdriver")
        for level in ("easy", "medium")
    ],
)
def test_yam_task_factories_use_the_yam_station(name, stub_scene_module):
    embodiment = build_registered_environment(name, enable_cameras=True).embodiment
    midpoint = (
        np.asarray(embodiment.scene_config.left_robot.init_state.pos)
        + embodiment.scene_config.right_robot.init_state.pos
    ) / 2
    mount = (_LINK_SIX, _LINK_SIX_D405)
    wrists = {"left_wrist_camera": mount, "right_wrist_camera": mount}
    _assert_rig(embodiment.camera_config, _YAM_VIEWS, midpoint, wrists)


# AUTOLab's yam_station_usbc@v1 (tasks/usbc_insert/staging.py), the rig its USB-C
# tasks select: head and left-shoulder views at their own poses.
_USBC_BASES = (0.2525, 0.0, 0.75)
_USBC_VIEWS = {
    "top_camera": (
        _USBC_BASES,
        55.0,
        (0.01, 4.0),
        (0.2525, 0, 1.69),
        (0.7725, 0, 0.89),
        (1, 0, 0),
    ),
    "side_camera": (
        _USBC_BASES,
        50.0,
        (0.01, 4.0),
        (0.2125, 0.82, 1.89),
        (0.7825, 0, 0.97),
        (0, 0, 1),
    ),
}


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_usbc_uses_the_usbc_station(level, stub_scene_module):
    name = f"vabar_usbc_insertion__{level}"
    embodiment = build_registered_environment(name, enable_cameras=True).embodiment
    midpoint = (
        np.asarray(embodiment.scene_config.left_robot.init_state.pos)
        + embodiment.scene_config.right_robot.init_state.pos
    ) / 2
    # USB-C runs AUTOLab's linear_4310 hand, which carries its D405s on its gripper body.
    mount = (_LINEAR_HAND, _LINEAR_HAND_D405)
    wrists = {"left_wrist_camera": mount, "right_wrist_camera": mount}
    _assert_rig(embodiment.camera_config, _USBC_VIEWS, midpoint, wrists)


# yam_station_cable@v1 at its own poses, with the D405s on the linear_4310 hand.
_CABLE_STATION_VIEWS = {
    "top_camera": (
        _CABLE_BASES,
        55.0,
        (0.01, 4.0),
        (0.2525, 0, 1.69),
        (0.7725, 0, 0.89),
        (1, 0, 0),
    ),
    "cable_camera": (
        _CABLE_BASES,
        50.0,
        (0.01, 2.0),
        (0.9196, -0.06, 1.1357),
        (0.46, -0.06, 0.75),
        (0, 0, 1),
    ),
}
_LINEAR_HAND = f"{_LINK_SIX}/linear_4310_attach/Geometry/gripper"
_LINEAR_HAND_D405 = (
    (-0.123729, 0.0017, -0.070721),
    (0.640714032, 0.640714032, 0.29914132, 0.29914132),
)


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_cable_uses_the_cable_station(level, stub_scene_module):
    name = f"vabar_cable_routing__{level}"
    embodiment = build_registered_environment(name, enable_cameras=True).embodiment
    midpoint = (
        np.asarray(embodiment.scene_config.left_robot.init_state.pos)
        + embodiment.scene_config.right_robot.init_state.pos
    ) / 2
    wrists = {
        "left_wrist_camera": (_LINEAR_HAND, _LINEAR_HAND_D405),
        "right_wrist_camera": (_LINEAR_HAND, _LINEAR_HAND_D405),
    }
    _assert_rig(embodiment.camera_config, _CABLE_STATION_VIEWS, midpoint, wrists)


def test_yam_rig_follows_its_mounts():
    from isaaclab_arena_vabar.embodiments.bimanual_yam import (
        IndustrialBimanualYamEmbodiment,
    )

    left, right = (0.4, 0.3, 0.9), (0.4, -0.2, 0.9)
    embodiment = IndustrialBimanualYamEmbodiment(
        enable_cameras=True, left_mount_position=left, right_mount_position=right
    )
    wrists = {
        "left_wrist_camera": (_LINK_SIX, _LINK_SIX_D405),
        "right_wrist_camera": (_LINK_SIX, _LINK_SIX_D405),
    }
    midpoint = (np.asarray(left) + right) / 2
    _assert_rig(embodiment.camera_config, _YAM_VIEWS, midpoint, wrists)
