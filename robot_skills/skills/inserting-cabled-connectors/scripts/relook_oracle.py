"""Where the mate is *now*, read out of the scene.

The **oracle body** of the ``relook`` slot, and it is the same tool the ground
slot called -- because the honest thing about a re-look is not that it uses a
different instrument, it is that it happens at a different *time*. Between the
two reads the support arm has closed 3 mm of interference a side onto a free
90 g receptacle that its cradle does not stop from sliding, and the work hand
has closed on the plug and moved it about a millimetre in the jaws against a
0.15 mm-a-side fit.

Both of those are the whole difference between this graph and an open-loop
script. The reference's own stroke seats 3.6 mm of 13 because it plans from
where the parts were before anything touched them.

``scene_mouth_point`` and ``scene_bore_axis`` are declared as inputs and used
only to report how far the receptacle moved. This body does not need either --
it can measure the whole frame -- and it declares them anyway, because the pair
has to declare one interface and its vision twin genuinely needs both: it cannot
re-measure the bore's *height* from a top-down wrist view, and it cannot sign
its own axis once the plug has left the bench. Carrying the keys on both sides
is what keeps that asymmetry visible instead of hidden in a shape difference.
"""

import math
from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    relook_body: str
    mate_plug_point: list
    mate_mouth_point: list
    mate_bore_axis: list
    mate_plug_axis: list
    mate_plug_up: list
    mate_moved_mm: float


NOTHING: Output = {
    "route": "lost",
    "relook_body": "oracle",
    "mate_plug_point": [],
    "mate_mouth_point": [],
    "mate_bore_axis": [],
    "mate_plug_axis": [],
    "mate_plug_up": [],
    "mate_moved_mm": 0.0,
}


def run(
    ctx: NodeContext,
    scene_mouth_point: list | None = None,
    scene_bore_axis: list | None = None,
    scene_plug_half_h: float = 0.0,
    inspect_mouth_point: list | None = None,
    inspect_bore_axis: list | None = None,
    label: str = "relook",
) -> Output:
    state = ctx.tool("sim.perceive_mate")
    if str(state.get("route")) != "found":
        return dict(NOTHING)  # type: ignore[return-value]
    mouth = [float(v) for v in state["mouth_point"]]
    moved = _dist_mm(mouth, scene_mouth_point)
    turned = _turn_deg(state["bore_axis"], scene_bore_axis)
    print(
        f"[{label}:oracle] mouth {_mm(mouth)} ({moved:.2f} mm and {turned:.2f} deg from the "
        f"ground read)  plug in the jaws {_mm(state['plug_point'])}",
        flush=True,
    )
    return {
        "route": "found",
        "relook_body": "oracle",
        # The plug's LIVE pose, which after the close is its pose in the jaws.
        # Re-basing the stroke on it is this graph's whole correction.
        "mate_plug_point": [float(v) for v in state["plug_point"]],
        "mate_mouth_point": mouth,
        "mate_bore_axis": [float(v) for v in state["bore_axis"]],
        "mate_plug_axis": [],
        "mate_plug_up": [],
        "mate_moved_mm": moved,
    }


def _turn_deg(a, b) -> float:
    """How far the axis swung between the two reads [deg]. Zero when unstated."""
    if not a or not b:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=True))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, dot)))))


def _dist_mm(a, b) -> float:
    if not a or not b:
        return 0.0
    return float(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5 * 1e3)


def _mm(v) -> str:
    return "(" + ", ".join(f"{float(x) * 1e3:.1f}" for x in v) + ") mm"
