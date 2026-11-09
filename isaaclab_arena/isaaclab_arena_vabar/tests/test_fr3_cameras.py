"""Effective Franka camera rigs match AUTOLab's per-suite rigs (AUTOLab-GaP-NVIDIA-Collaboration 4a12c48)."""

import math

import numpy as np
import pytest
import yaml

pytest.importorskip("scipy")
pytest.importorskip("isaaclab")

from ib_support import REPO_ROOT, build_registered_environment
from scipy.spatial.transform import Rotation

# RSS CameraSpec literals: (width, height, fov_y, (near, far), eye, look_at,
# up), and the RSS Panda base position the eyes are authored against.
_GEAR = (
    (0.0, -0.115, 0.882),
    {
        "top_camera": (
            1280,
            960,
            50.0,
            (0.01, 4.0),
            (0.55, 0, 1.30),
            (0.55, 0, 0.77),
            (0, 1, 0),
        ),
        "iso_camera": (
            1280,
            720,
            40.0,
            (0.01, 4.0),
            (1.05, -0.50, 1.10),
            (0.53, 0, 0.77),
            (0, 0, 1),
        ),
        "chain_camera": (
            640,
            480,
            40.0,
            (0.01, 4.0),
            (0.62, 0.34, 0.83),
            (0.62, -0.02, 0.77),
            (0, 0, 1),
        ),
        "front_camera": (
            640,
            480,
            45.0,
            (0.01, 4.0),
            (0.18, -0.05, 1.02),
            (0.62, -0.01, 0.78),
            (0, 0, 1),
        ),
    },
)
_TOOL_SORTING = (
    (-0.5, -0.1, 0.912),
    {
        "top_camera": (
            1280,
            960,
            52.0,
            (0.01, 3.0),
            (0.10, 0.02, 2.05),
            (0.10, 0.02, 0.80),
            (-1, 0, 0),
        ),
        "oblique_camera": (
            640,
            480,
            45.0,
            (0.01, 5.0),
            (1.30, -0.80, 1.60),
            (0.10, 0.02, 0.82),
            (0, 0, 1),
        ),
    },
)
_SYRINGE = (
    (-0.5, -0.1, 0.912),
    {
        "top_camera": (
            1280,
            960,
            52.0,
            (0.01, 3.0),
            (0.10, 0.02, 1.75),
            (0.10, 0.02, 0.80),
            (1, 0, 0),
        ),
        "oblique_camera": (
            1280,
            960,
            45.0,
            (0.01, 5.0),
            (1.3213, 0.0826, 1.5159),
            (-0.1495, 0.0711, 1.0465),
            (0, 0, 1),
        ),
    },
)
_WRIST_CLIPPING = (0.005, 1.5)
_ROBOTIQ_TCP_Z = 0.157
_HAND = "/robotiq_attach/Geometry/robotiq_base"
_BASE = "/Robot/Geometry/base/fr3_link0"


def _panda_wrist_mount():
    """Recompute the RSS ``_panda_wrist_mount`` (tasks/tool_sort/staging.py)."""
    eye = np.array([0.0, 0.1, -0.07])
    # franka_hand_robosuite: TCP offset zero, closing axis +x.
    target = np.array([0.0, 0.0, 0.05])
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, (1.0, 0.0, 0.0))
    right /= np.linalg.norm(right)
    look_at = np.column_stack((right, np.cross(forward, right), forward))
    landscape_roll = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return eye, Rotation.from_matrix(look_at @ landscape_roll)


def test_panda_wrist_mount_matches_the_rss_literal():
    eye, rotation = _panda_wrist_mount()
    np.testing.assert_allclose(eye, (0.0, 0.1, -0.07))
    np.testing.assert_allclose(
        rotation.as_quat(),
        (0.0, -0.3404252637530182, 0.9402715776831118, 0.0),
        atol=1e-12,
    )


def _assert_intrinsics(camera, width, height, fov_y, clipping):
    assert (camera.width, camera.height) == (width, height)
    assert camera.offset.convention == "ros"
    assert camera.update_latest_camera_pose
    assert camera.data_types == ["rgb", "distance_to_image_plane"]
    assert camera.renderer_cfg.renderer_type == "isaac_rtx"
    spawn = camera.spawn
    assert math.degrees(
        2 * math.atan(spawn.vertical_aperture / (2 * spawn.focal_length))
    ) == pytest.approx(fov_y)
    assert camera.width / spawn.horizontal_aperture == pytest.approx(
        camera.height / spawn.vertical_aperture
    )
    assert spawn.clipping_range == pytest.approx(clipping)


def _assert_rig(embodiment, station, wrist):
    rig = embodiment.camera_config
    rss_base, views = station
    assert set(rig.camera_names()) == {*views, *(("wrist_camera",) if wrist else ())}
    # Every Franka scene stands the FR3 base where the RSS Panda stands
    # relative to the work surface, with no yaw, so base-relative views land
    # on the same part of the scene.
    assert embodiment.initial_pose.rotation_xyzw[:3] == pytest.approx((0, 0, 0))
    for name, (width, height, fov_y, clipping, eye, look_at, up) in views.items():
        camera = getattr(rig, name)
        _assert_intrinsics(camera, width, height, fov_y, clipping)
        assert camera.prim_path.rsplit("/", 1)[0].endswith(_BASE)
        np.testing.assert_allclose(
            camera.offset.pos, np.subtract(eye, rss_base), atol=1e-9
        )
        # ROS optical: +z looks at the target, +y (image down) opposes up.
        rotation = Rotation.from_quat(camera.offset.rot)
        forward = np.subtract(look_at, eye) / np.linalg.norm(np.subtract(look_at, eye))
        np.testing.assert_allclose(rotation.apply((0, 0, 1)), forward, atol=1e-9)
        assert rotation.apply((1, 0, 0)) @ up == pytest.approx(0, abs=1e-9)
        assert rotation.apply((0, 1, 0)) @ up < 0
    if wrist:
        width, height, tcp_standoff = wrist
        camera = rig.wrist_camera
        _assert_intrinsics(camera, width, height, 60.0, _WRIST_CLIPPING)
        assert camera.prim_path.rsplit("/", 1)[0].endswith(_HAND)
        eye, rotation = _panda_wrist_mount()
        np.testing.assert_allclose(
            camera.offset.pos, eye + (0, 0, tcp_standoff), atol=1e-12
        )
        np.testing.assert_allclose(
            Rotation.from_quat(camera.offset.rot).as_matrix(),
            rotation.as_matrix(),
            atol=1e-12,
        )
        # Beside the closing (x) axis, looking forward (+z) through the jaws.
        view = Rotation.from_quat(camera.offset.rot).apply((0, 0, 1))
        assert view[2] > 0 and view[1] < 0 and view[0] == pytest.approx(0)


def test_scenes_place_the_fr3_base_where_rss_places_the_panda():
    import re

    from ib_support import ASSETS_ROOT, load_scene

    table = (
        ASSETS_ROOT
        / "vabar__fr3_workcell_table/vabar__fr3_workcell_table.usda"
    ).read_text()
    surface = table[table.index('def Xform "placement_surface"') :]
    low, high = (
        np.array([float(v) for v in corner.split(",")])
        for corner in re.search(
            r"extent = \[\(([^)]*)\), \(([^)]*)\)\]", surface
        ).groups()
    )
    np.testing.assert_allclose(
        ((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, high[2]),
        (0.6, 0.115, -0.132),
        atol=1e-6,
    )

    # Surface frame (x, y, top z) relative to the robot base. RSS Panda:
    # surface o (-0.6, -0.115, 0.132), so the surface is (0.6, 0.115, -0.132)
    # from the base. Arena: base (-0.5, -0.1, 0.912); the FR3 workcell table
    # (same pose) has its placement surface centred 0.6 ahead, 0.115 left and
    # 0.132 below its origin -> world surface (0.1, 0.015, 0.78).
    for relative_yaml in (
        "gear_insertion/easy.yaml",
        "gear_insertion/easy_pair.yaml",
        "gear_insertion/medium_train.yaml",
        "tool_sorting/easy_1.yaml",
        "tool_sorting/easy_2.yaml",
        "tool_sorting/easy_3.yaml",
        "syringe_disposal/single.yaml",
        "syringe_disposal/both.yaml",
        "syringe_disposal/designated.yaml",
        "syringe_disposal/cluttered.yaml",
    ):
        scene = load_scene(relative_yaml)
        robot = scene["embodiment"]["params"]["initial_pose"]
        table = scene["background"]["params"]["initial_pose"]
        assert scene["background"]["registry_name"] == "vabar__fr3_workcell_table"
        assert robot["position_xyz"] == table["position_xyz"] == [-0.5, -0.1, 0.912]
        assert abs(robot["rotation_xyzw"][3]) == 1.0


@pytest.mark.parametrize(
    ("name", "station", "wrist"),
    [
        *[
            (f"vabar_gear_insertion__{suffix}", _GEAR, None)
            for suffix in ("easy", "easy_pair", "medium_train")
        ],
        # panda_robotiq: the mount rides the Robotiq base_mount at the flange.
        *[
            (f"vabar_tool_sorting__easy_{index}", _TOOL_SORTING, (320, 240, 0.0))
            for index in (1, 2, 3)
        ],
        # Franka hand: the mount rides the hand's TCP, here the Robotiq's.
        *[
            (
                f"vabar_syringe_disposal__{suffix}",
                _SYRINGE,
                (1280, 960, _ROBOTIQ_TCP_Z),
            )
            for suffix in ("single", "both", "designated", "cluttered")
        ],
    ],
)
def test_franka_task_factories_use_the_autolab_suite_rig(
    name, station, wrist, stub_scene_module
):
    embodiment = build_registered_environment(name, enable_cameras=True).embodiment
    _assert_rig(embodiment, station, wrist)


def test_fr3_requires_a_camera_rig_with_cameras():
    from isaaclab_arena_vabar.embodiments.fr3_robotiq_2f85 import (
        IndustrialFr3Robotiq2f85Embodiment,
    )

    with pytest.raises(ValueError, match="camera_rig"):
        IndustrialFr3Robotiq2f85Embodiment(enable_cameras=True)
    assert IndustrialFr3Robotiq2f85Embodiment().camera_config is None


# RSS camera names per graph. gear_mesh/gap_perception reads `overhead`
# (scripts/sight.py); tool_sort/gap_perception_loop reads `overhead` (workflow.json);
# syringe_packing/gap_perception reads `overhead`, `eye_in_hand` and
# `agentview` (scripts/). The rest of each rig is streamed as the RSS version does.
_GRAPH_ALIASES = {
    "gear_mesh/gap_perception": {
        "top_camera": "overhead",
        "iso_camera": "iso",
        "chain_camera": "chain",
        "front_camera": "front",
    },
    "tool_sort/gap_perception_loop": {
        "top_camera": "overhead",
        "oblique_camera": "agentview",
        "wrist_camera": "eye_in_hand",
    },
    "syringe_packing/gap_perception": {
        "top_camera": "overhead",
        "oblique_camera": "agentview",
        "wrist_camera": "eye_in_hand",
    },
}
# The ground-truth oracle graphs stream the same rig as their perception pairs.
_GRAPH_ALIASES["tool_sort/gt"] = _GRAPH_ALIASES["tool_sort/gap_perception_loop"]
_GRAPH_ALIASES["syringe_packing/gt"] = _GRAPH_ALIASES["syringe_packing/gap_perception"]


@pytest.mark.parametrize(
    "config",
    sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for directory in ("gear_insertion", "syringe_disposal", "tool_sorting")
        for path in (REPO_ROOT / "isaaclab_arena_gap/experiment_configs" / directory).glob(
            "*.yaml"
        )
        if "usbc" not in path.name
    ),
)
def test_franka_experiments_stream_every_rig_camera_under_its_rss_name(
    config, stub_scene_module
):
    document = yaml.safe_load((REPO_ROOT / config).read_text())
    if "shared" in document:
        # A per-seed campaign streams the shared rig; its runs pin layouts and seeds only.
        run = document["shared"]
        for seed_run in document["runs"].values():
            assert not any("camera" in key for key in seed_run.get("policy", {})), config
    else:
        (run,) = document["runs"].values()
    policy = run["policy"]
    streamed = {
        policy[f"gap_{slot}camera"]: policy[f"gap_{slot}bridge_camera"]
        for slot in ("", "wrist_", "aux_")
        if policy.get(f"gap_{slot}camera")
    }
    streamed.update(policy.get("gap_extra_cameras", {}))
    assert streamed == _GRAPH_ALIASES[policy["gap_graph"]]
    environment = build_registered_environment(
        run["environment"]["type"], enable_cameras=True
    )
    assert set(streamed) == set(environment.embodiment.camera_config.camera_names())
