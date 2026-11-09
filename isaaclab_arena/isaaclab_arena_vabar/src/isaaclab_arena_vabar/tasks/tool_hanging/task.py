"""Reusable Arena task configuration for hanging tools from fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from isaaclab.managers import TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_newton.sensors.contact_sensor.contact_sensor_cfg import (
    ContactSensorCfg as NewtonContactSensorCfg,
)

from isaaclab_arena_vabar.numeric import is_finite_real

from .geometry import LoopFeature, RodFeature
from .metrics import ToolHangingFractionMetric
from .terminations import HandTcp, HangingGoalSpec, tool_hanging_held_success


YAM_TCPS = tuple(
    HandTcp(robot_name=f"{side}_robot", body_name="link_6", offset_xyz=(0.0, -0.044, 0.13))
    for side in ("left", "right")
)
"""Both YAM hands' tool centres: RoboSimStudio's ``yam_linear_arena`` grip point on ``link_6``,
the hand the RSS adapter registers for Arena's tool-hanging crank hand."""

RELEASE_HOLD_S = 1.0
"""The RSS version's ``(hold-for 1.0 ...)`` around each tool's clause."""


class ToolHangingTask(TaskBase):
    """Score graph-resolved tools and fixtures against numeric loop/rod goals.

    As the RSS version scores tool hanging: every goal -- hung, in contact with its fixture,
    at rest and released (no hand within the goal's ``hands_clear_m``) -- holds for
    :data:`RELEASE_HOLD_S` of simulated time, and success ends the episode.
    """

    def __init__(
        self,
        tools: list[Asset],
        fixtures: list[Asset],
        goals: Sequence[Mapping[str, Any]],
        episode_length_s: float | None = None,
        task_description: str | None = None,
        hold_s: float = RELEASE_HOLD_S,
        hands: tuple[HandTcp, ...] = YAM_TCPS,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.tools = tuple(tools)
        self.fixtures = tuple(fixtures)
        tool_by_name = _assets_by_name("tools", self.tools)
        fixture_by_name = _assets_by_name("fixtures", self.fixtures)
        self.goal_specs = tuple(
            _goal_spec(goal, tool_by_name, fixture_by_name) for goal in goals
        )
        if not self.goal_specs:
            raise ValueError("ToolHangingTask requires at least one goal")
        _validate_goal_coverage(self.tools, self.fixtures, self.goal_specs)
        self.contact_sensor_names = tuple(goal.contact_sensor_name for goal in self.goal_specs)
        self.contact_sensors = tuple(
            _newton_fixture_contact_sensor(
                tool_by_name[goal.tool_name],
                tuple(fixture_by_name[name] for name in goal.fixture_names),
            )
            for goal in self.goal_specs
        )
        self.hold_s = _threshold("hold_s", hold_s)
        self.hands = tuple(hands)
        self.scene_config = self._make_scene_cfg()
        self.termination_cfg = TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="tool_hanging",
                    predicate_sequence=[
                        TerminationTermCfg(
                            func=tool_hanging_held_success,
                            params={"goals": self.goal_specs, "hands": self.hands, "hold_s": self.hold_s},
                        )
                    ],
                )
            ],
        )

    def _make_scene_cfg(self):
        fields: list[tuple[str, type[ContactSensorCfg], ContactSensorCfg]] = [
            (name, type(sensor), sensor)
            for name, sensor in zip(self.contact_sensor_names, self.contact_sensors, strict=True)
        ]
        return make_configclass("ToolHangingSceneCfg", fields)()

    def get_scene_cfg(self) -> Any:
        return self.scene_config

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def get_events_cfg(self) -> Any:
        return None

    def get_mimic_env_cfg(self, arm_mode) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ToolHangingFractionMetric(self.goal_specs, self.hands)]


def _newton_fixture_contact_sensor(tool: Asset, fixtures: tuple[Asset, ...]) -> NewtonContactSensorCfg:
    """Filter a dynamic tool body against fixed fixture collision shapes."""
    if not fixtures:
        raise ValueError("tool-hanging contact sensor requires at least one fixture")
    base_cfg = tool.get_contact_sensor_cfg(contact_against_object=fixtures[0])
    fixture_shape_exprs = [
        f"{fixture.get_prim_path().rstrip('/')}(?:/.*)?" for fixture in fixtures
    ]
    if isinstance(base_cfg, NewtonContactSensorCfg):
        base_cfg.filter_prim_paths_expr = []
        base_cfg.filter_shape_prim_expr = fixture_shape_exprs
        return base_cfg
    base_cfg.filter_prim_paths_expr = []
    return NewtonContactSensorCfg.from_base_cfg(
        base_cfg,
        filter_shape_prim_expr=fixture_shape_exprs,
    )


def _assets_by_name(label: str, assets: Sequence[Asset]) -> dict[str, Asset]:
    by_name = {asset.name: asset for asset in assets}
    if len(by_name) != len(assets):
        raise ValueError(f"{label} must have unique names")
    return by_name


def _validate_goal_coverage(
    tools: Sequence[Asset], fixtures: Sequence[Asset], goals: Sequence[HangingGoalSpec]
) -> None:
    tool_names = {tool.name for tool in tools}
    fixture_names = {fixture.name for fixture in fixtures}
    goal_tool_names = [goal.tool_name for goal in goals]
    goal_fixture_names = {
        fixture_name for goal in goals for fixture_name in goal.fixture_names
    }
    if (
        len(goal_tool_names) != len(tool_names)
        or set(goal_tool_names) != tool_names
        or len(set(goal_tool_names)) != len(goal_tool_names)
        or goal_fixture_names != fixture_names
    ):
        raise ValueError(
            "tool-hanging goals must cover every tool exactly once and every fixture"
        )


def _threshold(name: str, value: Any) -> float:
    if not is_finite_real(value):
        raise ValueError(f"tool-hanging {name} must be a finite non-boolean real number")
    if value < 0.0:
        raise ValueError("tool-hanging thresholds must be non-negative")
    return float(value)


def _goal_spec(
    raw: Mapping[str, Any], tools: Mapping[str, Asset], fixtures: Mapping[str, Asset]
) -> HangingGoalSpec:
    try:
        tool_name = raw["tool"]
        fixture_name = raw.get("fixture")
    except KeyError as error:
        raise ValueError(f"tool-hanging goal is missing {error.args[0]!r}") from error
    if not isinstance(tool_name, str) or tool_name not in tools:
        raise ValueError(f"tool-hanging goal references unknown tool {tool_name!r}")
    fixture_names_raw = raw.get("fixtures")
    if (fixture_name is None) == (fixture_names_raw is None):
        raise ValueError("tool-hanging goal must define exactly one of fixture or fixtures")
    if fixture_names_raw is not None:
        if (
            not isinstance(fixture_names_raw, Sequence)
            or isinstance(fixture_names_raw, (str, bytes))
            or not fixture_names_raw
            or len(set(fixture_names_raw)) != len(fixture_names_raw)
            or any(not isinstance(name, str) or name not in fixtures for name in fixture_names_raw)
        ):
            raise ValueError("tool-hanging goal references invalid alternative fixtures")
        fixture_names = tuple(fixture_names_raw)
        fixture_name = fixture_names[0]
    else:
        if not isinstance(fixture_name, str) or fixture_name not in fixtures:
            raise ValueError(f"tool-hanging goal references unknown fixture {fixture_name!r}")
        fixture_names = (fixture_name,)
    containment_raw = raw.get("containment")
    loop_raw = raw.get("loop")
    rod_raw = raw.get("rod")
    predicate_raw = raw.get("predicate", "threaded")
    alternative_loops_raw = raw.get("alternative_loops", ())
    geometry_count = sum((
        containment_raw is not None,
        loop_raw is not None or rod_raw is not None,
    ))
    if geometry_count != 1:
        raise ValueError("tool-hanging goal must define exactly one geometry predicate")
    if containment_raw is not None and (
        "predicate" in raw or "alternative_loops" in raw
    ):
        raise ValueError(
            "tool-hanging predicate and alternative_loops require loop/rod geometry"
        )
    loop = rod = None
    alternative_loops: tuple[LoopFeature, ...] = ()
    containment_min = containment_max = None
    containment_point = (0.0, 0.0, 0.0)
    if containment_raw is not None:
        if not isinstance(containment_raw, Mapping):
            raise ValueError("tool-hanging containment must be a numeric mapping")
        containment_min = _point("containment minimum_xyz", containment_raw.get("minimum_xyz"))
        containment_max = _point("containment maximum_xyz", containment_raw.get("maximum_xyz"))
        containment_point = _point(
            "containment point_xyz", containment_raw.get("point_xyz", (0.0, 0.0, 0.0))
        )
        if any(low >= high for low, high in zip(containment_min, containment_max, strict=True)):
            raise ValueError("tool-hanging containment bounds must be ordered")
    else:
        if not isinstance(loop_raw, Mapping) or not isinstance(rod_raw, Mapping):
            raise ValueError("tool-hanging goal loop and rod must be numeric mappings")
        if predicate_raw not in {"threaded", "encloses"}:
            raise ValueError("tool-hanging predicate must be threaded or encloses")
        if (
            not isinstance(alternative_loops_raw, Sequence)
            or isinstance(alternative_loops_raw, (str, bytes))
            or any(not isinstance(item, Mapping) for item in alternative_loops_raw)
        ):
            raise ValueError(
                "tool-hanging alternative_loops must be a sequence of mappings"
            )
        loop = LoopFeature(**dict(loop_raw))
        rod = RodFeature(**dict(rod_raw))
        alternative_loops = tuple(
            LoopFeature(**dict(item)) for item in alternative_loops_raw
        )
        if predicate_raw != "encloses" and any(
            candidate.enclosure_tolerance_m for candidate in (loop, *alternative_loops)
        ):
            raise ValueError(
                "tool-hanging enclosure_tolerance_m applies only to the encloses predicate"
            )
    owner_names = set(tools) | set(fixtures)
    loop_owner_name = raw.get("loop_owner", tool_name)
    rod_owner_name = raw.get("rod_owner", fixture_name)
    if not isinstance(loop_owner_name, str) or loop_owner_name not in owner_names:
        raise ValueError(
            f"tool-hanging goal references unknown loop owner {loop_owner_name!r}"
        )
    if not isinstance(rod_owner_name, str) or rod_owner_name not in owner_names:
        raise ValueError(
            f"tool-hanging goal references unknown rod owner {rod_owner_name!r}"
        )
    contact_force_n = _threshold("contact_force_n", raw.get("contact_force_n", 0.2))
    speed_m_s = _threshold("speed_m_s", raw.get("speed_m_s", 0.15))
    hands_clear_raw = raw.get("hands_clear_m")
    hands_clear_m = None if hands_clear_raw is None else _threshold("hands_clear_m", hands_clear_raw)
    release_point = _point("release_point_xyz", raw.get("release_point_xyz", (0.0, 0.0, 0.0)))
    return HangingGoalSpec(
        tool_name=tool_name,
        fixture_name=fixture_name,
        loop_owner_name=loop_owner_name,
        rod_owner_name=rod_owner_name,
        contact_sensor_name=(
            f"contact_sensor_{tool_name}_{'_or_'.join(fixture_names)}"
            if len(fixture_names) > 1
            else f"contact_sensor_{tool_name}_{fixture_name}"
        ),
        loop=loop,
        rod=rod,
        predicate=predicate_raw,
        alternative_loops=alternative_loops,
        containment_min_xyz=containment_min,
        containment_max_xyz=containment_max,
        containment_point_xyz=containment_point,
        contact_force_n=contact_force_n,
        speed_m_s=speed_m_s,
        hands_clear_m=hands_clear_m,
        release_point_xyz=release_point,
        alternative_fixture_names=fixture_names if len(fixture_names) > 1 else (),
    )


def _point(name: str, value: Any) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"tool-hanging {name} must contain three values")
    result: list[float] = []
    for item in value:
        if not is_finite_real(item):
            raise ValueError(f"tool-hanging {name} must contain finite real values")
        result.append(float(item))
    return tuple(result)  # type: ignore[return-value]
