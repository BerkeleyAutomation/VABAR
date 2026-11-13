"""Durable black-box scoring semantics for benchmark-owned tasks."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch


class _TorchValue:
    def __init__(self, value: torch.Tensor):
        self.torch = value


def _asset(position, quaternion=(0.0, 0.0, 0.0, 1.0), velocity=None):
    position = torch.tensor([position], dtype=torch.float32)
    return SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=_TorchValue(position),
            root_link_quat_w=_TorchValue(
                torch.tensor([quaternion], dtype=torch.float32)
            ),
            root_lin_vel_w=_TorchValue(
                torch.tensor([velocity or (0.0, 0.0, 0.0)], dtype=torch.float32)
            ),
            root_com_vel_w=_TorchValue(torch.zeros((1, 6), dtype=torch.float32)),
        )
    )


def _usbc_goal():
    from isaaclab_arena_vabar.tasks.usbc_insertion.geometry import Bore
    from isaaclab_arena_vabar.tasks.usbc_insertion.terminations import UsbcGoalSpec

    return UsbcGoalSpec(
        bore=Bore(
            mouth=(-0.0045, 0.0, 0.0),
            axis=(1.0, 0.0, 0.0),
            depth=0.0276,
            half_extents=(0.0098, 0.004),
            tip_offset=0.0133,
        ),
        depth_min=0.0104,
        lateral_max=0.0087931792,
        tilt_max=None,
        speed_max=0.05,
        tcp_distance_min=0.05,
        require_released=True,
        plug_name="plug",
        port_name="port",
    )


def _usbc_hand():
    from isaaclab_arena_vabar.tasks.usbc_insertion.terminations import (
        UsbcWorkHandSpec,
    )

    return UsbcWorkHandSpec(
        robot_name="right_robot",
        tcp_body_name="link_6",
        tcp_offset_xyz=(0.0, 0.0, 0.0),
        gripper_action_name="right_gripper_action",
        gripper_joint_name="left_finger",
        span_m=0.075048,
        open_joint_m=0.037524,
        grasp_width_m=0.010,
    )


class _ActionManager:
    def __init__(self, target: float):
        self.term = SimpleNamespace(
            processed_actions=torch.tensor([[target]], dtype=torch.float32)
        )

    def get_term(self, name: str):
        assert name == "right_gripper_action"
        return self.term


def _usbc_env(
    position,
    quaternion,
    velocity=(0.0, 0.0, 0.0),
    *,
    measured_gripper=0.037524,
    commanded_gripper=0.037524,
    tcp_position=(-0.1, 0.0, 0.0),
):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_names=["left_finger"],
            body_names=["link_6"],
            joint_pos=_TorchValue(
                torch.tensor([[measured_gripper]], dtype=torch.float32)
            ),
            body_link_pos_w=_TorchValue(
                torch.tensor([[tcp_position]], dtype=torch.float32)
            ),
            body_link_quat_w=_TorchValue(
                torch.tensor([[(1.0, 0.0, 0.0, 0.0)]], dtype=torch.float32)
            ),
        )
    )
    return SimpleNamespace(
        scene={
            "plug": _asset(position, quaternion, velocity),
            "port": _asset((0.0, 0.0, 0.0)),
            "right_robot": robot,
        },
        action_manager=_ActionManager(commanded_gripper),
    )


_ALIGNED_WXYZ = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
_SEATED = (0.012, 0.0, 0.0)


@pytest.mark.parametrize(
    ("position", "quaternion", "env_changes", "accepted"),
    [
        pytest.param(_SEATED, _ALIGNED_WXYZ, {}, True, id="seated-aligned-resting"),
        # Easy has no tilt clause: an untilted plug seated at the bore is accepted.
        pytest.param(
            (0.0253, 0.0, 0.0133), (1.0, 0.0, 0.0, 0.0), {}, True, id="no-tilt-gate"
        ),
        pytest.param((-0.010, 0.0, 0.0), _ALIGNED_WXYZ, {}, False, id="shallow"),
        pytest.param((0.012, 0.02, 0.0), _ALIGNED_WXYZ, {}, False, id="off-axis"),
        pytest.param(
            _SEATED, _ALIGNED_WXYZ, {"velocity": (1, 0, 0)}, False, id="moving"
        ),
        pytest.param(
            _SEATED,
            _ALIGNED_WXYZ,
            {"measured_gripper": 0.005, "commanded_gripper": 0.004},
            False,
            id="still-gripped-at-plug-width",
        ),
        pytest.param(
            _SEATED,
            _ALIGNED_WXYZ,
            {"measured_gripper": 0.001, "commanded_gripper": 0.0},
            True,
            id="empty-stalled-jaws-are-not-a-grasp",
        ),
        pytest.param(
            _SEATED,
            _ALIGNED_WXYZ,
            {"tcp_position": (-0.02, 0.0, 0.0)},
            False,
            id="work-tcp-inside-clearance",
        ),
    ],
)
def test_usbc_seated_predicate(position, quaternion, env_changes, accepted) -> None:
    from isaaclab_arena_vabar.tasks.usbc_insertion.terminations import (
        usbc_insertion_seated,
    )

    result = usbc_insertion_seated(
        _usbc_env(position, quaternion, **env_changes), _usbc_goal(), _usbc_hand()
    ).item()
    assert result is accepted


def test_cable_centerline_uses_endpoints_and_newton_quaternion_order() -> None:
    from isaaclab_arena_vabar.tasks.cable_routing.geometry import capsule_centerline

    half = torch.tensor([0.006] * 50, dtype=torch.float64)
    bodies = torch.zeros((2, 50, 7), dtype=torch.float64)
    centers = torch.arange(50, dtype=torch.float64) * 0.012 + 0.006
    bodies[0, :, 2] = centers
    bodies[0, :, 6] = 1.0  # XYZW identity: local Z stays world Z.
    bodies[1, :, 0] = centers
    bodies[1, :, 4] = bodies[1, :, 6] = 2**-0.5  # Rotate local Z onto world X.
    points = capsule_centerline(bodies, half)

    assert points.shape == (2, 51, 3)
    expected = torch.arange(51, dtype=torch.float64) * 0.012
    torch.testing.assert_close(points[0, :, 2], expected)
    torch.testing.assert_close(points[1, :, 0], expected)
    torch.testing.assert_close(points[:, 0], torch.zeros((2, 3), dtype=torch.float64))
    torch.testing.assert_close(
        points[1, :, 1:], torch.zeros((51, 2), dtype=torch.float64)
    )


def test_tool_hanging_threading_geometry_accepts_crossing_and_rejects_miss() -> None:
    from isaaclab_arena_vabar.tasks.tool_hanging.geometry import (
        LoopFeature,
        RodFeature,
        threaded,
    )

    loop = LoopFeature((0, 0, 0), (0, 0, 1), 0.03)
    rod = RodFeature((0, 0, -0.04), (0, 0, 0.04), 0.002)
    identity = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    missed = identity.clone()
    missed[0, 0] = 0.05
    assert threaded(loop, rod, identity, identity).tolist() == [True]
    assert threaded(loop, rod, identity, missed).tolist() == [False]


def test_tool_hanging_encloses_uses_either_handle_bow() -> None:
    from isaaclab_arena_vabar.tasks.tool_hanging.geometry import (
        LoopFeature,
        RodFeature,
        encloses,
    )

    bow = LoopFeature((-1.0, 0.0, 0.0), (0, 0, 1), 0.03)
    other_bow = LoopFeature((1.0, 0.0, 0.0), (0, 0, 1), 0.03)
    hook = RodFeature((0, 0, -0.04), (0, 0, 0.04), 0.002)
    identity = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    on_other_bow = identity.clone()
    on_other_bow[0, 0] = -1.0

    assert not encloses(bow, hook, on_other_bow, identity).item()
    assert encloses(other_bow, hook, on_other_bow, identity).item()


def test_tool_hanging_encloses_grants_autolab_enclosure_tolerance() -> None:
    from isaaclab_arena_vabar.tasks.tool_hanging.geometry import (
        LoopFeature,
        RodFeature,
        encloses,
        threaded,
    )

    # Wrench easy's ring on its hook shank: the ring plane is normal to the shank.
    ring_r, shank_r = 0.0102, 0.00249
    shank = RodFeature((0.0, 0.0, 0.0), (0.0611, 0.0, 0.0), shank_r)
    strict = LoopFeature((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), ring_r)
    tolerant = LoopFeature((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), ring_r, enclosure_tolerance_m=0.008)
    rings = torch.tensor(
        [
            # Resting on the shank, 3 mm past the usable radius r_loop - r_rod.
            [0.03, 0.0, -(ring_r - shank_r + 0.003), 0.0, 0.0, 0.0, 1.0],
            # Fallen to the table.
            [0.03, 0.0, -0.25, 0.0, 0.0, 0.0, 1.0],
            # Slid 40 mm past the shank's free end.
            [-0.04, 0.0, -(ring_r - shank_r), 0.0, 0.0, 0.0, 1.0],
        ]
    )
    hook = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)

    assert encloses(strict, shank, rings, hook).tolist() == [False, False, False]
    assert encloses(tolerant, shank, rings, hook).tolist() == [True, False, False]
    # AUTOLab's tolerance belongs to encloses alone.
    assert threaded(tolerant, shank, rings, hook).tolist() == threaded(
        strict, shank, rings, hook
    ).tolist()


def test_tool_hanging_rejects_invalid_enclosure_tolerance() -> None:
    from isaaclab_arena_vabar.tasks.tool_hanging.geometry import LoopFeature
    from isaaclab_arena_vabar.tasks.tool_hanging.task import _goal_spec

    with pytest.raises(ValueError, match="enclosure tolerance"):
        LoopFeature((0, 0, 0), (0, 0, 1), 0.01, enclosure_tolerance_m=-0.001)
    raw = {
        "tool": "wrench",
        "fixture": "hook",
        "loop": {
            "center_xyz": (0.0, 0.0, 0.0),
            "normal_xyz": (1.0, 0.0, 0.0),
            "radius_m": 0.0102,
            "enclosure_tolerance_m": 0.008,
        },
        "rod": {
            "start_xyz": (0.0, 0.0, 0.0),
            "end_xyz": (0.0611, 0.0, 0.0),
            "radius_m": 0.00249,
        },
    }
    with pytest.raises(ValueError, match="only to the encloses predicate"):
        _goal_spec(raw, {"wrench": object()}, {"hook": object()})
    goal = _goal_spec(
        {**raw, "predicate": "encloses"}, {"wrench": object()}, {"hook": object()}
    )
    assert goal.loop.enclosure_tolerance_m == 0.008


def test_tool_hanging_parses_alternative_bows_and_applies_physical_gates() -> None:
    from isaaclab_arena_vabar.tasks.tool_hanging.task import _goal_spec
    from isaaclab_arena_vabar.tasks.tool_hanging.terminations import (
        tool_hanging_goal_completion,
    )

    tool = _asset((-1.0, 0.0, 0.0))
    fixture = _asset((0.0, 0.0, 0.0))
    goal = _goal_spec(
        {
            "tool": "scissors",
            "fixture": "hook",
            "predicate": "encloses",
            "loop": {
                "center_xyz": (-1.0, 0.0, 0.0),
                "normal_xyz": (0.0, 0.0, 1.0),
                "radius_m": 0.03,
            },
            "alternative_loops": [
                {
                    "center_xyz": (1.0, 0.0, 0.0),
                    "normal_xyz": (0.0, 0.0, 1.0),
                    "radius_m": 0.03,
                }
            ],
            "rod": {
                "start_xyz": (0.0, 0.0, -0.04),
                "end_xyz": (0.0, 0.0, 0.04),
                "radius_m": 0.002,
            },
            "contact_force_n": 0.2,
            "speed_m_s": 0.08,
        },
        {"scissors": tool},
        {"hook": fixture},
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(force_matrix_w=torch.tensor([[[[0.3, 0.0, 0.0]]]]))
    )
    env = SimpleNamespace(
        scene={
            "scissors": tool,
            "hook": fixture,
            goal.contact_sensor_name: sensor,
        }
    )

    assert tool_hanging_goal_completion(env, goals=(goal,)).item()
    sensor.data.force_matrix_w.zero_()
    assert not tool_hanging_goal_completion(env, goals=(goal,)).item()
    sensor.data.force_matrix_w[..., 0] = 0.3
    tool.data.root_lin_vel_w.torch[..., 0] = 0.1
    assert not tool_hanging_goal_completion(env, goals=(goal,)).item()

    goal.predicate = "unknown"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unsupported tool-hanging predicate"):
        tool_hanging_goal_completion(env, goals=(goal,))


def test_tool_hanging_scores_a_non_finite_tool_pose_as_incomplete() -> None:
    from isaaclab_arena_vabar.tasks.tool_hanging.task import _goal_spec
    from isaaclab_arena_vabar.tasks.tool_hanging.terminations import (
        tool_hanging_goal_completion,
    )

    tool = _asset((0.0, 0.0, 0.0))
    fixture = _asset((0.0, 0.0, 0.0))
    goal = _goal_spec(
        {
            "tool": "wrench",
            "fixture": "hook",
            "predicate": "encloses",
            "loop": {
                "center_xyz": (0.0, 0.0, 0.0),
                "normal_xyz": (0.0, 0.0, 1.0),
                "radius_m": 0.03,
            },
            "rod": {
                "start_xyz": (0.0, 0.0, -0.04),
                "end_xyz": (0.0, 0.0, 0.04),
                "radius_m": 0.002,
            },
            "contact_force_n": 0.2,
            "speed_m_s": 0.08,
        },
        {"wrench": tool},
        {"hook": fixture},
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(force_matrix_w=torch.tensor([[[[0.3, 0.0, 0.0]]]]))
    )
    env = SimpleNamespace(
        scene={"wrench": tool, "hook": fixture, goal.contact_sensor_name: sensor}
    )
    assert tool_hanging_goal_completion(env, goals=(goal,)).item()

    tool.data.root_link_pos_w.torch[0, 2] = float("nan")
    assert not tool_hanging_goal_completion(env, goals=(goal,)).item()


@pytest.mark.parametrize("field", ("predicate", "alternative_loops"))
def test_tool_hanging_rejects_loop_options_without_loop_geometry(field: str) -> None:
    from isaaclab_arena_vabar.tasks.tool_hanging.task import _goal_spec

    raw = {
        "tool": "tool",
        "fixture": "fixture",
        "containment": {
            "minimum_xyz": (-0.1, -0.1, -0.1),
            "maximum_xyz": (0.1, 0.1, 0.1),
        },
        field: "encloses" if field == "predicate" else [],
    }
    with pytest.raises(ValueError, match="require loop/rod geometry"):
        _goal_spec(raw, {"tool": object()}, {"fixture": object()})


def _gear_mesh_term(
    num_envs=1,
    *,
    gears=(((0.032, 0.0, 0.008), 0.85),),
    target_offsets_xyz=((0.032, 0.0, 0.008),),
    gear_teeth=(20,),
    tcp_xyz=(0.10, 0.0, 0.008),
    button_pressed=False,
):
    """Build a gear-mesh success term over ``(position, spin)`` gear states."""
    from isaaclab_arena_vabar.tasks.gear_insertion.terminations import gear_mesh_success
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)

    class _Board:
        def __init__(self):
            self.data = SimpleNamespace(
                joint_names=["pinion_joint", "button_joint"],
                joint_pos=torch.zeros((num_envs, 2)),
                root_pos_w=torch.zeros((num_envs, 3)),
                root_quat_w=identity.clone(),
            )
            self.data.joint_pos[:, 1] = -0.005 if button_pressed else 0.0
            self.velocity_target = None

        def set_joint_velocity_target(self, target, *, joint_ids):
            self.velocity_target = (target.clone(), joint_ids)

    gear_assets = {
        ("gear" if len(gears) == 1 else f"gear_{index}"): SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([position]).repeat(num_envs, 1),
                root_quat_w=identity.clone(),
                root_com_vel_w=torch.tensor(
                    [[0.0, 0.0, 0.0, 0.0, 0.0, spin]]
                ).repeat(num_envs, 1),
            )
        )
        for index, (position, spin) in enumerate(gears)
    }
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["robotiq_base"],
            body_pos_w=torch.tensor([[tcp_xyz]]).repeat(num_envs, 1, 1),
            body_quat_w=identity[:, None, :].clone(),
        )
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=0.25,
        scene={"board": _Board(), "robot": robot, **gear_assets},
    )
    params = {
        "board_asset_cfg": SceneEntityCfg("board"),
        "gear_asset_cfgs": [SceneEntityCfg(name) for name in gear_assets],
        "robot_asset_cfg": SceneEntityCfg("robot"),
        "tcp_body_name": "robotiq_base",
        "tcp_offset_xyz": (0.0, 0.0, 0.157),
        "target_offsets_xyz": list(target_offsets_xyz),
        "button_latch_m": 0.005,
        "drive_speed_rad_s": 4.0,
        "spin_fraction": 0.3,
        "gear_teeth": gear_teeth,
        "spin_window_s": 0.5,
        "hold_time_s": 1.0,
    }
    cfg = TerminationTermCfg(func=gear_mesh_success, params=params)
    return gear_mesh_success(cfg, env), env, params


def test_gear_mesh_latches_motor_and_requires_release_and_full_hold() -> None:
    term, env, params = _gear_mesh_term()

    assert not term(env, **params).item()
    target, joint_ids = term.board.velocity_target
    assert target.tolist() == [[0.0]]
    assert joint_ids == [0]

    term.board.data.joint_pos[0, 1] = -0.005
    for _ in range(3):
        assert not term(env, **params).item()
    assert term(env, **params).item()
    target, _ = term.board.velocity_target
    assert target.tolist() == [[4.0]]

    term.board.data.joint_pos[0, 1] = 0.0
    term.robot.data.body_pos_w[0, 0] = term.gear.data.root_pos_w[0] - torch.tensor(
        [0.0, 0.0, 0.157]
    )
    assert not term(env, **params).item()
    assert term.success_steps.tolist() == [0]
    target, _ = term.board.velocity_target
    assert target.tolist() == [[4.0]]


def test_gear_mesh_rejects_sub_ratio_rotation_and_reset_clears_latch() -> None:
    term, env, params = _gear_mesh_term()
    term.board.data.joint_pos[0, 1] = -0.005
    term.gear.data.root_com_vel_w[0, 5] = 0.839

    for _ in range(5):
        assert not term(env, **params).item()
    assert term.success_steps.tolist() == [0]

    term.reset()
    term.board.data.joint_pos[0, 1] = 0.0
    assert not term.latched.item()
    assert not term(env, **params).item()
    target, _ = term.board.velocity_target
    assert target.tolist() == [[0.0]]


def test_gear_mesh_requires_assembly_before_start() -> None:
    term, env, params = _gear_mesh_term()
    term.gear.data.root_pos_w[0, 0] += 0.1
    term.board.data.joint_pos[0, 1] = -0.005
    assert not term(env, **params).item()

    term.gear.data.root_pos_w[0, 0] -= 0.1
    for _ in range(5):
        assert not term(env, **params).item()
    assert term.latched.item()
    assert not term.started_after_seating.item()


@pytest.mark.parametrize(
    ("gear_teeth", "threshold"),
    [(16, 1.05), (20, 0.84), (24, 0.70)],
)
def test_gear_mesh_spin_threshold_scales_with_selected_teeth(
    gear_teeth: int, threshold: float
) -> None:
    term, env, params = _gear_mesh_term()
    params["gear_teeth"] = (gear_teeth,)
    term.board.data.joint_pos[0, 1] = -0.005
    term.gear.data.root_com_vel_w[0, 5] = threshold - 0.001
    for _ in range(5):
        assert not term(env, **params).item()
    assert term.success_steps.tolist() == [0]

    term.reset()
    term.gear.data.root_com_vel_w[0, 5] = threshold + 0.001
    for _ in range(3):
        assert not term(env, **params).item()
    assert term(env, **params).item()


def test_gear_mesh_matches_rss_unsigned_turning_gate() -> None:
    term, env, params = _gear_mesh_term()
    term.board.data.joint_pos[0, 1] = -0.005
    term.gear.data.root_com_vel_w[0, 5] = -0.85

    for _ in range(3):
        assert not term(env, **params).item()
    assert term(env, **params).item()


def test_gear_mesh_allows_interchangeable_repeated_tooth_sizes() -> None:
    offsets = ((0.03, 0.00, 0.008), (0.07, 0.01, 0.008), (0.11, -0.01, 0.008))
    # The two 16-tooth gears swap stations; each still meets its own spin gate.
    term, env, params = _gear_mesh_term(
        2,
        gears=((offsets[2], 1.1), (offsets[1], 0.9), (offsets[0], 1.1)),
        target_offsets_xyz=offsets,
        gear_teeth=(16, 20, 16),
        tcp_xyz=(0.5, 0.5, 0.5),
        button_pressed=True,
    )

    for _ in range(3):
        assert not term(env, **params).any()
    assert term(env, **params).all()


def _tool_sort_object(
    *,
    root=(0.0, 0.0, 0.1),
    center=(0.0, 0.0, 0.1),
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
):
    return SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([root], dtype=torch.float32),
            root_com_pos_w=torch.tensor([center], dtype=torch.float32),
            root_lin_vel_w=torch.tensor([linear], dtype=torch.float32),
            root_ang_vel_w=torch.tensor([angular], dtype=torch.float32),
        )
    )


def _tool_sort_term(
    *,
    root=(0.0, 0.0, 0.1),
    center=(0.0, 0.0, 0.1),
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
    gripper_position=0.0,
):
    from isaaclab_arena_vabar.tasks.tool_sorting.task import (
        objects_centered_in_regions_and_released,
    )
    from isaaclab.managers import TerminationTermCfg
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={
            "hammer": _tool_sort_object(
                root=root,
                center=center,
                linear=linear,
                angular=angular,
            ),
            "bin": SimpleNamespace(
                data=SimpleNamespace(
                    root_pos_w=torch.zeros((1, 3)),
                    root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                )
            ),
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    joint_names=["left_driver_joint"],
                    joint_pos=torch.tensor([[gripper_position]], dtype=torch.float32),
                )
            ),
        },
    )
    params = {
        "object_names": ["hammer"],
        "region_names": ["bin"],
        "bounds_xyzxyz": [(-0.2, -0.2, 0.0, 0.2, 0.2, 0.2)],
        "linear_velocity_threshold": 0.01,
        "angular_velocity_threshold": 0.05,
        "robot_asset_name": "robot",
        "gripper_joint_name": "left_driver_joint",
        "gripper_open_position_threshold": 0.1,
    }
    predicate = TrueForConsecutiveStepsCfg(
        predicate=TerminationTermCfg(
            func=objects_centered_in_regions_and_released,
            params=params,
        ),
        required_steps=10,
    )
    tracker = ProgressTracker(
        [CompletionCriteria(name="settled", predicate_sequence=[predicate])],
        num_envs=1,
        device="cpu",
        env=env,
    )
    return tracker, env


def _tool_sort_step(tracker, env, step):
    tracker.step(env, step_index=torch.tensor([step]))
    return tracker.is_complete().item()


def test_tool_sort_uses_center_of_mass_and_requires_a_settled_dwell() -> None:
    tracker, env = _tool_sort_term(root=(0.4, 0.0, 0.1))

    for step in range(9):
        assert not _tool_sort_step(tracker, env, step)
    assert _tool_sort_step(tracker, env, 9)


def test_tool_sort_requires_release_and_resets_dwell_on_motion() -> None:
    tracker, env = _tool_sort_term(gripper_position=0.9)
    for step in range(10):
        assert not _tool_sort_step(tracker, env, step)

    env.scene["robot"].data.joint_pos[0, 0] = 0.0
    for step in range(10, 15):
        assert not _tool_sort_step(tracker, env, step)
    env.scene["hammer"].data.root_ang_vel_w[0, 2] = 0.1
    assert not _tool_sort_step(tracker, env, 15)

    env.scene["hammer"].data.root_ang_vel_w[0, 2] = 0.0
    for step in range(16, 25):
        assert not _tool_sort_step(tracker, env, step)
    assert _tool_sort_step(tracker, env, 25)


def test_tool_sort_reset_clears_the_dwell_counter() -> None:
    tracker, env = _tool_sort_term()
    for step in range(5):
        assert not _tool_sort_step(tracker, env, step)

    tracker.reset([0])

    for step in range(9):
        assert not _tool_sort_step(tracker, env, step)
    assert _tool_sort_step(tracker, env, 9)


@pytest.mark.parametrize("num_envs", [1, 2, 5])
def test_gear_mesh_motor_targets_are_independent_per_environment(num_envs):
    term, env, params = _gear_mesh_term(num_envs)
    term.board.data.joint_pos[::2, 1] = -0.005
    term(env, **params)
    target, joint_ids = term.board.velocity_target
    expected = torch.zeros((num_envs, 1))
    expected[::2] = 4.0
    assert target.shape == (num_envs, 1)
    assert joint_ids == [term.pinion_joint]
    torch.testing.assert_close(target, expected)

    term.board.data.joint_pos[:, 1] = 0.0
    term(env, **params)
    torch.testing.assert_close(term.board.velocity_target[0], expected)

    term.reset([0])
    term(env, **params)
    expected[0] = 0.0
    torch.testing.assert_close(term.board.velocity_target[0], expected)


def test_gear_mesh_targets_and_speed_gates_are_per_environment():
    term, env, params = _gear_mesh_term(2)
    params["target_offsets_xyz"] = [[[0.032, 0.0, 0.008]], [[0.08, 0.0, 0.008]]]
    params["gear_teeth"] = [[16], [24]]
    term.gear.data.root_pos_w[1] = torch.tensor([0.08, 0.0, 0.008])
    # 0.85 rad/s passes a 24-tooth station's 0.7 gate, but fails 16 teeth's 1.05.
    term(env, **params)
    term.board.data.joint_pos[:, 1] = -0.006
    for _ in range(5):
        result = term(env, **params)
    assert result.tolist() == [False, True]
    # A different environment's target must never satisfy this one.
    term.gear.data.root_pos_w[1] = torch.tensor([0.032, 0.0, 0.008])
    assert term(env, **params).tolist() == [False, False]
    term.reset([1])
    assert term.latched.tolist() == [True, False]


def test_tool_sort_counts_a_parked_object_as_absent() -> None:
    from isaaclab_arena_vabar.tasks.tool_sorting.task import (
        objects_centered_in_regions_and_released,
    )

    _, env = _tool_sort_term(center=(0.35, 0.88, 0.79), linear=(0.0, 0.0, 0.0))
    env.scene = _SceneWithOrigins(env.scene, torch.zeros((1, 3)))
    params = {
        "object_names": ["hammer"],
        "region_names": ["bin"],
        "bounds_xyzxyz": [(-0.2, -0.2, 0.0, 0.2, 0.2, 0.2)],
        "linear_velocity_threshold": 0.01,
        "angular_velocity_threshold": 0.05,
        "robot_asset_name": "robot",
        "gripper_joint_name": "left_driver_joint",
        "gripper_open_position_threshold": 0.1,
    }
    assert not objects_centered_in_regions_and_released(env, **params).item()
    parked = {**params, "parked_bounds_xyzxyz": (0.2, 0.8, 0.7, 0.5, 0.95, 0.9)}
    assert objects_centered_in_regions_and_released(env, **parked).item()
    # Environment-local: the same object in a clone offset by 10 m is still parked.
    env.scene.env_origins = torch.tensor([[10.0, 0.0, 0.0]])
    env.scene["hammer"].data.root_com_pos_w[0, 0] += 10.0
    assert objects_centered_in_regions_and_released(env, **parked).item()


class _SceneWithOrigins(dict):
    def __init__(self, entries, env_origins):
        super().__init__(entries)
        self.env_origins = env_origins
