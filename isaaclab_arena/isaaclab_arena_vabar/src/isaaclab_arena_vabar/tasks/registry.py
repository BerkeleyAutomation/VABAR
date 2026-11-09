"""Stable task-registration seam for the industrial benchmark package."""

from __future__ import annotations

from isaaclab_arena.assets.registries import TaskRegistry

from isaaclab_arena_vabar.registry import registered_component

from .gear_insertion import GearMeshTask
from .tool_hanging.task import ToolHangingTask
from .tool_sorting import ObjectsCenteredInRegionsTask
from .tool_sorting.regions_task import ObjectsInRegionsTask
from .usbc_insertion.task import UsbcInsertionTask


def register_tasks() -> None:
    """Register benchmark-owned tasks, rejecting type collisions.

    Explicit rather than decorator-driven, matching the assets, embodiments and
    environments seams. A ``@register_task`` decorator would only fire if some
    other module happened to import the task, which is how an environment can end
    up unreachable while looking registered.
    """

    registry = TaskRegistry()
    for task, task_name in (
        (GearMeshTask, GearMeshTask.__name__),
        (ToolHangingTask, "IndustrialToolHangingTask"),
        (ObjectsCenteredInRegionsTask, ObjectsCenteredInRegionsTask.__name__),
        (ObjectsInRegionsTask, ObjectsInRegionsTask.__name__),
        (UsbcInsertionTask, UsbcInsertionTask.__name__),
    ):
        name = task_name
        existing = registered_component(registry, name)
        if existing is not None:
            if existing is not task:
                raise RuntimeError(
                    f"conflicting task registration for {name!r}: {existing!r}"
                )
            continue
        registry.register(task, name)
