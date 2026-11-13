"""AUTOLab Franka camera rigs from AUTOLab-GaP-NVIDIA-Collaboration main (4a12c48).

One rig per suite, copied from the rig each RSS suite stages:

- gear mesh: ``gear_mesh_views@v2`` in tasks/gear_mesh/staging.py;
- tool sorting Level 1: ``tool_sort_agentview@v3`` in tasks/tool_sort/staging.py,
  named by profile ``vab_industrial_robotiq@v1``;
- syringe: ``sharps_agentview@v1`` in tasks/syringe_packing/staging.py, named by
  profile ``sharps_vab_industrial@v1`` (the same rig, formerly ``agentview@v2``).
"""

import math
from dataclasses import dataclass

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from scipy.spatial.transform import Rotation

_BASE_PRIM = "{ENV_REGEX_NS}/Robot/Geometry/base/fr3_link0"
_HAND_PRIM = (
    f"{_BASE_PRIM}/fr3_link1/fr3_link2/fr3_link3/fr3_link4/"
    "fr3_link5/fr3_link6/fr3_link7/robotiq_attach/Geometry/robotiq_base"
)


@dataclass(frozen=True)
class _View:
    """One RSS look-at camera, in RSS world coordinates."""

    width: int
    height: int
    fov_y: float
    clipping: tuple[float, float]
    eye: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class _Wrist:
    """One RSS eye-in-hand camera: ``xform`` (xyz, xyzw) in the hand frame."""

    width: int
    height: int
    fov_y: float
    clipping: tuple[float, float]
    xform: tuple[float, float, float, float, float, float, float]


# RSS Panda base positions. Fixed views keep their offset from this point
# and ride the FR3 base. Gear: bench_075_light's surface (0.6, 0, 0.75) composed
# with panda_robotiq's base_offset (-0.6, -0.115, 0.132). VAB: ``robot0_base``
# in every assets/vab_tool_sort export, which equals robosuite_bench's surface
# (0.1, 0.015, 0.78) composed with the same base_offset.
_GEAR_PANDA_BASE = (0.0, -0.115, 0.882)
_VAB_PANDA_BASE = (-0.5, -0.1, 0.912)

_GEAR_VIEWS = {
    "top_camera": _View(
        1280,
        960,
        50.0,
        (0.01, 4.0),
        (0.55, 0.0, 1.30),
        (0.55, 0.0, 0.77),
        (0.0, 1.0, 0.0),
    ),
    "iso_camera": _View(
        1280, 720, 40.0, (0.01, 4.0), (1.05, -0.50, 1.10), (0.53, 0.0, 0.77)
    ),
    "chain_camera": _View(
        640, 480, 40.0, (0.01, 4.0), (0.62, 0.34, 0.83), (0.62, -0.02, 0.77)
    ),
    "front_camera": _View(
        640, 480, 45.0, (0.01, 4.0), (0.18, -0.05, 1.02), (0.62, -0.01, 0.78)
    ),
}
_TOOL_SORTING_VIEWS = {
    "top_camera": _View(
        1280,
        960,
        52.0,
        (0.01, 3.0),
        (0.10, 0.02, 2.05),
        (0.10, 0.02, 0.80),
        (-1.0, 0.0, 0.0),
    ),
    "oblique_camera": _View(
        640, 480, 45.0, (0.01, 5.0), (1.30, -0.80, 1.60), (0.10, 0.02, 0.82)
    ),
}
_SYRINGE_VIEWS = {
    "top_camera": _View(
        1280,
        960,
        52.0,
        (0.01, 3.0),
        (0.10, 0.02, 1.75),
        (0.10, 0.02, 0.80),
        (1.0, 0.0, 0.0),
    ),
    "oblique_camera": _View(
        1280,
        960,
        45.0,
        (0.01, 5.0),
        (1.3213, 0.0826, 1.5159),
        (-0.1495, 0.0711, 1.0465),
    ),
}

# ``PANDA_WRIST_MOUNT``: camera pose (ROS optical, xyz then xyzw) in the frame
# ``parent_frame="work"`` resolves to. That frame approaches along +z and closes
# along x, as does ``robotiq_base``.
_PANDA_WRIST_MOUNT = (
    0.0,
    0.1,
    -0.07,
    0.0,
    -0.3404252637530182,
    0.9402715776831118,
    0.0,
)
# Tool sorting stages panda_robotiq, whose "work" frame is the Robotiq
# ``base_mount`` at the flange: the same origin and axes as ``robotiq_base``.
_TOOL_SORTING_WRIST = _Wrist(320, 240, 60.0, (0.005, 1.5), _PANDA_WRIST_MOUNT)
# Syringe stages the Franka hand, whose "work" frame is its TCP. The mount is
# kept relative to the Robotiq TCP, 0.157 m along ``robotiq_base`` +z.
_ROBOTIQ_TCP_Z = 0.157
_SYRINGE_WRIST = _Wrist(
    1280,
    960,
    60.0,
    (0.005, 1.5),
    (
        _PANDA_WRIST_MOUNT[0],
        _PANDA_WRIST_MOUNT[1],
        _PANDA_WRIST_MOUNT[2] + _ROBOTIQ_TCP_Z,
        *_PANDA_WRIST_MOUNT[3:],
    ),
)


def _look_at_rotation(view: _View) -> tuple[float, ...]:
    """RSS ``look_at_matrix``: columns right, down, forward (ROS optical)."""
    eye = np.asarray(view.eye)
    forward = np.asarray(view.look_at) - eye
    forward /= np.linalg.norm(forward)
    up = np.asarray(view.up) / np.linalg.norm(view.up)
    if abs(forward @ up) > 0.999:
        up = np.array((0.0, 1.0, 0.0))
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.column_stack((right, down, forward))
    return tuple(float(value) for value in Rotation.from_matrix(matrix).as_quat())


def _camera(path, width, height, fov_y, clipping, position, rotation):
    vertical_aperture = 3.6
    return CameraCfg(
        prim_path=path,
        width=width,
        height=height,
        update_latest_camera_pose=True,
        data_types=["rgb", "distance_to_image_plane"],
        renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=vertical_aperture / (2 * math.tan(math.radians(fov_y / 2))),
            horizontal_aperture=vertical_aperture * width / height,
            vertical_aperture=vertical_aperture,
            clipping_range=clipping,
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation, convention="ros"),
    )


def _fixed(name, views, base):
    """A world-fixed RSS view, mounted on the FR3 base at its base offset."""
    view = views[name]
    return _camera(
        f"{_BASE_PRIM}/{name}",
        view.width,
        view.height,
        view.fov_y,
        view.clipping,
        tuple(float(e - b) for e, b in zip(view.eye, base, strict=True)),
        _look_at_rotation(view),
    )


def _wrist(wrist: _Wrist):
    return _camera(
        f"{_HAND_PRIM}/wrist_camera",
        wrist.width,
        wrist.height,
        wrist.fov_y,
        wrist.clipping,
        wrist.xform[:3],
        wrist.xform[3:],
    )


@configclass
class Fr3GearMeshCameraCfg(ArenaCameraCfg):
    """``gear_mesh_views@v2``: overhead, iso, chain and front; no wrist camera."""

    top_camera: CameraCfg = _fixed("top_camera", _GEAR_VIEWS, _GEAR_PANDA_BASE)
    iso_camera: CameraCfg = _fixed("iso_camera", _GEAR_VIEWS, _GEAR_PANDA_BASE)
    chain_camera: CameraCfg = _fixed("chain_camera", _GEAR_VIEWS, _GEAR_PANDA_BASE)
    front_camera: CameraCfg = _fixed("front_camera", _GEAR_VIEWS, _GEAR_PANDA_BASE)


@configclass
class Fr3ToolSortingCameraCfg(ArenaCameraCfg):
    """``tool_sort_agentview@v3``: overhead, agentview and eye-in-hand."""

    top_camera: CameraCfg = _fixed("top_camera", _TOOL_SORTING_VIEWS, _VAB_PANDA_BASE)
    oblique_camera: CameraCfg = _fixed(
        "oblique_camera", _TOOL_SORTING_VIEWS, _VAB_PANDA_BASE
    )
    wrist_camera: CameraCfg = _wrist(_TOOL_SORTING_WRIST)


@configclass
class Fr3SyringeCameraCfg(ArenaCameraCfg):
    """``sharps_agentview@v1`` (formerly ``agentview@v2``): overhead, agentview and eye-in-hand, all 1280 x 960."""

    top_camera: CameraCfg = _fixed("top_camera", _SYRINGE_VIEWS, _VAB_PANDA_BASE)
    oblique_camera: CameraCfg = _fixed(
        "oblique_camera", _SYRINGE_VIEWS, _VAB_PANDA_BASE
    )
    wrist_camera: CameraCfg = _wrist(_SYRINGE_WRIST)


FR3_CAMERA_RIGS = {
    "gear_mesh": Fr3GearMeshCameraCfg,
    "tool_sorting": Fr3ToolSortingCameraCfg,
    "syringe": Fr3SyringeCameraCfg,
}
"""Rigs a scene selects with the FR3 embodiment's ``camera_rig``."""
