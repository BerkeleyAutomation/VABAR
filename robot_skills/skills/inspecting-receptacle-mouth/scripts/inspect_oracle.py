"""Where the bore is, from the simulator, at the empty-hand look station.

The privileged twin of ``inspect_vision``. It exists so the slot has a pair --
the graph runs the same schedule either way, and the difference between the two
runs is the perception and nothing else.

It returns the same three keys the ground read publishes, so whichever body runs
here simply *refines* the scene frame that the grasp and the stroke go on to
use. The oracle's refinement is exact by construction; what it is really for is
telling you how much of a vision run's failure is the reading and how much is
the schedule.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    inspect_body: str
    scene_mouth_point: list
    scene_bore_axis: list
    inspect_mouth_point: list
    inspect_bore_axis: list
    inspect_moved_mm: float


def run(
    ctx: NodeContext,
    scene_mouth_point: list | None = None,
    scene_bore_axis: list | None = None,
    scene_receptacle_held: bool = False,
) -> Output:
    del scene_receptacle_held
    state = ctx.tool("sim.perceive_mate")
    if str(state.get("route")) != "found":
        # NOT an abort. The bore was already read from the bench, and that read
        # is what the schedule was built on; this station only sharpens it. A
        # station that sees nothing leaves the frame it was handed.
        print("[inspect:oracle] no mate state; keeping the bench read", flush=True)
        return _kept("oracle", scene_mouth_point, scene_bore_axis)
    mouth = [float(v) for v in state["mouth_point"]]
    moved = _dist_mm(mouth, scene_mouth_point)
    print(
        f"[inspect:oracle] mouth {_mm(mouth)} ({moved:.2f} mm from the bench read)",
        flush=True,
    )
    return {
        "route": "found",
        "inspect_body": "oracle",
        "scene_mouth_point": mouth,
        "scene_bore_axis": [float(v) for v in state["bore_axis"]],
        # See the vision twin: `inspect_*` is this station's own read, empty
        # when there was none, so the re-look can tell it from a carried frame.
        "inspect_mouth_point": mouth,
        "inspect_bore_axis": [float(v) for v in state["bore_axis"]],
        "inspect_moved_mm": moved,
    }


def _kept(body: str, mouth, axis) -> Output:
    return {
        "route": "found",
        "inspect_body": body,
        "scene_mouth_point": [float(v) for v in (mouth or [])],
        "scene_bore_axis": [float(v) for v in (axis or [])],
        "inspect_mouth_point": [],
        "inspect_bore_axis": [],
        "inspect_moved_mm": 0.0,
    }


def _dist_mm(a, b) -> float:
    if not a or not b:
        return 0.0
    return float(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=False)) ** 0.5 * 1e3)


def _mm(v) -> str:
    return "(" + ", ".join(f"{float(x) * 1e3:.1f}" for x in v) + ") mm"
