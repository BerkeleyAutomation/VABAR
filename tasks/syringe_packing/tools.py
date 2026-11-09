# SPDX-License-Identifier: Apache-2.0
"""Tool sort's privileged tools: the shared goal cursor, plus the one thing only
it needs.

Two halves now, and the split is where the reuse is.

**The cursor is not here any more.** "Which of this problem's goals should I do
next" is the ``.bddl`` file's ``(:goal)`` conjunction read against the
evaluator's verdict, and nothing in that sentence is about tools or bins -- so
:class:`~robosimstudio_policy_agent.sim_tools.GoalCursor` lives beside the other
tools that read nothing but the goal tree and the snapshot. This suite's copy
was the source it was promoted from, unchanged in behaviour: the same
containment pairs in document order, the same two-pass retry cap, the same
``attempt`` the grasp node fans its yaw on. Every suite that needs it registers
the promoted one, which is the point of moving it.

**``sim.get_container`` stays**, because it is not a fact about goals. It reads
the export's own container sidecar to say whether a region belongs to a sealed
box with a hole in its lid, and only this suite ships one. The generic ``sim.*``
set deliberately knows nothing about what task is running; a tool that reads one
export's ``containers`` block is exactly the kind of thing
:attr:`~robosimstudio.specs.TaskSpec.tools` exists to contribute.

Together they are what turn one workflow into six. Without the cursor a graph has
to name its object and its region as literals, which is why the shipped ``gt``
graph used to be a hammer graph that four of the six variants could not run at
all: they contain no ``hammer_0``.
"""

from __future__ import annotations

from typing import Any

from robosimstudio_policy_agent.sim_tools import register_goal_cursor

__all__ = ["SharpsDisposalTools", "register_syringe_packing_tools"]


def register_syringe_packing_tools(registry, env, bus) -> None:
    """Register this suite's privileged tools on a GaP tool registry.

    Args:
        registry: A ``gap_core.tools.ToolRegistry``.
        env: The toolkit-layer environment.
        bus: The bus whose snapshot carries the evaluator's metrics.
    """
    register_goal_cursor(registry, env, bus)
    tools = SharpsDisposalTools(env, bus)
    registry.register_callable(
        "sim.get_container",
        tools.get_container,
        summary="Whether a goal region is a container with a drop hole, and where its rim and aperture are.",
        tags=("privileged",),
    )


class SharpsDisposalTools:
    """The suite's own tool implementation -- the container lookup, and nothing else."""

    def __init__(self, env, bus) -> None:
        self.env = env
        self.bus = bus

    def get_container(self, region_name: str) -> dict[str, Any]:
        """Whether a goal region belongs to a container, and how to fill it.

        A compartment and a container are not the same destination and cannot be
        filled the same way. A compartment is open from above, so releasing a
        tool over it works. A sharps box is a sealed shell whose lid is solid
        everywhere except one hole: a tool let go flat over it lands *on* the
        lid, which is what ``syringe_packing/tray_pair`` did on its first end-to-end run
        -- the graph completed every node, reported success, and scored zero.

        The reference fills one by standing the tool on end over the hole and
        letting go (``run_tool_sort.post_upright``), which needs the two numbers
        this returns: where the hole is, and how high the rim it must clear is.
        Both are recorded by the export, so this is a lookup rather than a
        measurement.

        Returns ``is_container`` false for an ordinary compartment region, which
        is the common case and the one a caller should treat as "release
        normally".
        """
        containers = (self.env.handles.meta or {}).get("containers") or {}
        for name, entry in containers.items():
            if region_name != f"{name}_region":
                continue
            ax, ay, radius = (float(v) for v in entry["aperture"])
            return {
                "region_name": region_name,
                "container_name": name,
                "is_container": True,
                # World, because that is the frame a waypoint is given in.
                "aperture": {"x": ax, "y": ay, "radius": radius},
                "rim_z": float(entry["rim_z"]),
            }
        return {
            "region_name": region_name,
            "container_name": "",
            "is_container": False,
            "aperture": {"x": 0.0, "y": 0.0, "radius": 0.0},
            "rim_z": 0.0,
        }
