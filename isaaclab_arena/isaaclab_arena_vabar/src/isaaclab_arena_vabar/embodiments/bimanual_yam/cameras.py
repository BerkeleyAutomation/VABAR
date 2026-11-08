"""YAM station camera rigs derived from AUTOLab-GaP-NVIDIA-Collaboration (b1c9012).

Source: ``YAM_STATION_RIG`` in tasks/{tool_hanging,usbc_insert,cable_route}/staging.py
on the ``yam-station-cameras`` branch.
YAM scenes share :class:`YamCameraCfg`; cable and USB-C swap in their stations' views.
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

_WIDTH, _HEIGHT = 1280, 720
_LINK_SIX_SUFFIX = "/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"
_WRIST_FOV_Y = 58.0
_WRIST_CLIPPING = (0.005, 1.5)
# D405 optical pose (ROS axes, xyzw) relative to the canonical YAM link 6.
_LINK_SIX_WRIST_POSITION = (-0.0017, 0.079729, 0.066021)
_LINK_SIX_WRIST_ROTATION = (0.0, -0.42304971, 0.90610647, 0.0)
# The D405 on the linear_4310 hand's gripper body, as on AUTOLab's I2RT arm.
_LINEAR_HAND_SUFFIX = f"{_LINK_SIX_SUFFIX}/linear_4310_attach/Geometry/gripper"
_LINEAR_HAND_WRIST_POSITION = (-0.123729, 0.0017, -0.070721)
_LINEAR_HAND_WRIST_ROTATION = (0.640714032, 0.640714032, 0.29914132, 0.29914132)


@dataclass(frozen=True)
class _View:
    """One RSS look-at camera, in RSS world coordinates."""

    fov_y: float
    clipping: tuple[float, float]
    eye: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)


# RSS base-pair midpoints. Fixed views keep their offset from this point.
_TOOL_HANGING_BASES = (0.3025, 0.0, 0.81)
_TOOL_HANGING_VIEWS = {
    "top_camera": _View(
        55.0, (0.01, 4.0), (0.3025, 0.0, 1.668), (0.8225, 0.0, 0.868), (1.0, 0.0, 0.0)
    ),
    "side_camera": _View(
        50.0, (0.01, 4.0), (0.2625, 0.82, 1.868), (0.8325, 0.0, 0.948)
    ),
}
_CABLE_BASES = (0.2525, 0.0, 0.75)
# The cable station's oblique view, raised from 0.386 m to 0.85 m above the bases so
# it also looks over the tool-hanging pegboard; same heading and lateral offset.
_CABLE_VIEWS = {
    "cable_camera": _View(
        50.0, (0.01, 2.0), (0.9196, -0.06, 1.60), (0.5025, -0.06, 0.75)
    ),
}
# AUTOLab's yam_station_cable@v1 at its own poses: its overhead differs from tool
# hanging's, and its oblique view is not raised over a pegboard.
_CABLE_STATION_VIEWS = {
    "top_camera": _View(
        55.0, (0.01, 4.0), (0.2525, 0.0, 1.69), (0.7725, 0.0, 0.89), (1.0, 0.0, 0.0)
    ),
    "cable_camera": _View(
        50.0, (0.01, 2.0), (0.9196, -0.06, 1.1357), (0.46, -0.06, 0.75)
    ),
}
# Each fixed view of the shared YAM rig, with the base midpoint it is authored
# against: tool hanging's overhead and side views plus the cable station's view.
_YAM_VIEWS = {
    "top_camera": (_TOOL_HANGING_BASES, _TOOL_HANGING_VIEWS["top_camera"]),
    "side_camera": (_TOOL_HANGING_BASES, _TOOL_HANGING_VIEWS["side_camera"]),
    "cable_camera": (_CABLE_BASES, _CABLE_VIEWS["cable_camera"]),
}


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


def _camera(path, fov_y, clipping, position, rotation):
    vertical_aperture = 3.6
    return CameraCfg(
        prim_path=path,
        width=_WIDTH,
        height=_HEIGHT,
        update_period=1 / 30,
        update_latest_camera_pose=True,
        data_types=["rgb", "distance_to_image_plane"],
        renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=vertical_aperture / (2 * math.tan(math.radians(fov_y / 2))),
            horizontal_aperture=vertical_aperture * _WIDTH / _HEIGHT,
            vertical_aperture=vertical_aperture,
            clipping_range=clipping,
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation, convention="ros"),
    )


def _fixed(bases, view, name):
    offset = tuple(e - b for e, b in zip(view.eye, bases, strict=True))
    return _camera(
        f"{{ENV_REGEX_NS}}/{name}",
        view.fov_y,
        view.clipping,
        offset,
        _look_at_rotation(view),
    )


def _link_six_wrist(side):
    root = "LeftRobot" if side == "left" else "RightRobot"
    return _camera(
        f"{{ENV_REGEX_NS}}/{root}{_LINK_SIX_SUFFIX}/{side}_wrist_camera",
        _WRIST_FOV_Y,
        _WRIST_CLIPPING,
        _LINK_SIX_WRIST_POSITION,
        _LINK_SIX_WRIST_ROTATION,
    )


@configclass
class _YamStationCameraCfg(ArenaCameraCfg):
    """Fixed views follow the base-pair midpoint; wrist views ride the arm."""

    def _fixed_views(self) -> dict[str, tuple[tuple[float, float, float], _View]]:
        raise NotImplementedError

    def set_robot_mount_positions(self, left, right) -> None:
        """Keep each fixed view where the RSS scene has it relative to the bases."""
        midpoint = (np.asarray(left, dtype=float) + np.asarray(right, dtype=float)) / 2
        for name, (bases, view) in self._fixed_views().items():
            getattr(self, name).offset.pos = tuple(
                float(value) for value in midpoint + view.eye - np.asarray(bases)
            )


@configclass
class YamCameraCfg(_YamStationCameraCfg):
    """Shared YAM rig: overhead, left-shoulder, cable and two D405 wrists.

    Overhead and side views are ``yam_station_tool_hanging@v1``; the cable view is
    ``yam_station_cable@v1`` raised over the pegboard. Both D405s sit on link 6.
    """

    top_camera: CameraCfg = _fixed(*_YAM_VIEWS["top_camera"], "top_camera")
    side_camera: CameraCfg = _fixed(*_YAM_VIEWS["side_camera"], "side_camera")
    cable_camera: CameraCfg = _fixed(*_YAM_VIEWS["cable_camera"], "cable_camera")
    left_wrist_camera: CameraCfg = _link_six_wrist("left")
    right_wrist_camera: CameraCfg = _link_six_wrist("right")

    def _fixed_views(self):
        return _YAM_VIEWS


def linear_hand_wrist(side):
    """A D405 on ``side``'s linear_4310 gripper body."""
    root = "LeftRobot" if side == "left" else "RightRobot"
    return _camera(
        f"{{ENV_REGEX_NS}}/{root}{_LINEAR_HAND_SUFFIX}/{side}_wrist_camera",
        _WRIST_FOV_Y,
        _WRIST_CLIPPING,
        _LINEAR_HAND_WRIST_POSITION,
        _LINEAR_HAND_WRIST_ROTATION,
    )


@configclass
class YamCableCameraCfg(_YamStationCameraCfg):
    """AUTOLab's ``yam_station_cable@v1`` rig for the linear_4310 hand."""

    top_camera: CameraCfg = _fixed(
        _CABLE_BASES, _CABLE_STATION_VIEWS["top_camera"], "top_camera"
    )
    cable_camera: CameraCfg = _fixed(
        _CABLE_BASES, _CABLE_STATION_VIEWS["cable_camera"], "cable_camera"
    )
    left_wrist_camera: CameraCfg = linear_hand_wrist("left")
    right_wrist_camera: CameraCfg = linear_hand_wrist("right")

    def _fixed_views(self):
        return {name: (_CABLE_BASES, view) for name, view in _CABLE_STATION_VIEWS.items()}


# AUTOLab's ``yam_station_usbc@v1`` (tasks/usbc_insert/staging.py), the rig its USB-C
# tasks select: a head camera and a left-shoulder camera at their own poses, each
# higher than tool hanging's. Its wrist D405 rides the hand.
_USBC_BASES = (0.2525, 0.0, 0.75)
_USBC_STATION_VIEWS = {
    "top_camera": _View(
        55.0, (0.01, 4.0), (0.2525, 0.0, 1.69), (0.7725, 0.0, 0.89), (1.0, 0.0, 0.0)
    ),
    "side_camera": _View(
        50.0, (0.01, 4.0), (0.2125, 0.82, 1.89), (0.7825, 0.0, 0.97)
    ),
}


@configclass
class YamUsbcCameraCfg(_YamStationCameraCfg):
    """AUTOLab's ``yam_station_usbc@v1`` head and side views.

    The scene copies the wrist D405s from the embodiment, which mounts them on
    the selected hand.
    """

    top_camera: CameraCfg = _fixed(
        _USBC_BASES, _USBC_STATION_VIEWS["top_camera"], "top_camera"
    )
    side_camera: CameraCfg = _fixed(
        _USBC_BASES, _USBC_STATION_VIEWS["side_camera"], "side_camera"
    )
    left_wrist_camera: CameraCfg = _link_six_wrist("left")
    right_wrist_camera: CameraCfg = _link_six_wrist("right")

    def _fixed_views(self):
        return {name: (_USBC_BASES, view) for name, view in _USBC_STATION_VIEWS.items()}
