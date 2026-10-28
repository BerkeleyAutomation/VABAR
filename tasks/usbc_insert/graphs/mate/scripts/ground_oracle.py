"""Where the two halves of the mate are, read out of the scene.

The **oracle body** of the ``ground`` slot: the plug's grasp station and heading,
the bore's mouth and axis, a free receptacle's barrel, and whether the
receptacle is free at all. One tool call, because ``sim.perceive_mate`` was
written to return exactly the keys a camera could earn -- see its docstring for
why that constraint shaped it rather than the other way round.

**What it deliberately does not return.** No depth past the mouth, no lateral
error, no verdict. Those are ``sim.get_insertion_state``'s, they are the
evaluator's own quantities, and a perception slot that handed them over would
let the graph route on the score. This body knows where things are; whether the
task is done is a different question and it is asked at the end.

The ``scene_`` prefix on every output is not decoration. Subgraph inputs bind by
**name** out of one flat cross-subgraph namespace, last writer wins, so the
``relook`` slot -- which refines two of these same quantities from the wrist --
has to write under names of its own or it would silently overwrite the ground
read that the stroke still needs the rest of.
"""

import re
from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    scene_body: str
    scene_plug_point: list
    scene_plug_yaw_deg: float
    scene_plug_half_h: float
    scene_mouth_point: list
    scene_bore_axis: list
    scene_barrel_point: list
    scene_receptacle_held: bool


NOTHING: Output = {
    "route": "not_found",
    "scene_body": "oracle",
    "scene_plug_point": [],
    "scene_plug_yaw_deg": 0.0,
    "scene_plug_half_h": 0.0,
    "scene_mouth_point": [],
    "scene_bore_axis": [],
    "scene_barrel_point": [],
    "scene_receptacle_held": False,
}
"""Every declared key, empty. A body that returned a *subset* on its miss path
would leave a downstream binding reading whatever was last written under that
name rather than nothing at all."""


def run(ctx: NodeContext, instruction: str = "") -> Output:
    state = ctx.tool("sim.perceive_mate")
    if str(state.get("route")) != "found":
        return dict(NOTHING)  # type: ignore[return-value]
    # Through the tool, for the reason its docstring gives: the runtime's typed
    # observation drops the episode's command on the floor, so a body that read
    # it there would silently compare against the empty string and never warn.
    if not str(instruction).strip():
        instruction = str(ctx.tool("usbc.instruction").get("instruction") or "")
    # The one place this body can check its twin's homework. `receptacle_held`
    # is the key the vision body cannot see and has to read out of the
    # instruction instead, and here the scene knows the answer -- so a
    # disagreement is the language convention drifting from the cell, which is
    # exactly the failure a run under the other body would report as a missing
    # brace on a receptacle that needed one.
    said = _language_says_held(instruction)
    if said is not None and said != bool(state["held"]):
        print(
            f"[ground:oracle] WARNING: the instruction {instruction!r} reads held={said} and the "
            f"scene says held={state['held']}; the vision body will get this wrong",
            flush=True,
        )
    print(
        f"[ground:oracle] plug {_mm(state['plug_point'])} yaw {state['plug_yaw_deg']:.1f} deg  "
        f"mouth {_mm(state['mouth_point'])} axis {_round(state['bore_axis'])}  "
        f"receptacle {state['receptacle']!r} held={state['held']}",
        flush=True,
    )
    return {
        "route": "found",
        "scene_body": "oracle",
        "scene_plug_point": [float(v) for v in state["plug_point"]],
        "scene_plug_yaw_deg": float(state["plug_yaw_deg"]),
        "scene_plug_half_h": float(state["plug_half_h"]),
        "scene_mouth_point": [float(v) for v in state["mouth_point"]],
        "scene_bore_axis": [float(v) for v in state["bore_axis"]],
        "scene_barrel_point": [float(v) for v in state["barrel_point"]],
        "scene_receptacle_held": bool(state["held"]),
    }


HELD_WORDS = ("holding", "held", "bulkhead")
"""The words ``ground_vision`` reads ``receptacle_held`` out of, restated here
so this body can check them against what the scene actually is."""


def _language_says_held(instruction: str) -> bool | None:
    """What the instruction claims about the receptacle, or ``None`` if silent."""
    text = str(instruction or "").strip().lower()
    if not text:
        return None
    return any(re.search(rf"\b{word}\b", text) for word in HELD_WORDS)


def _mm(v) -> str:
    return "(" + ", ".join(f"{float(x) * 1e3:.1f}" for x in v) + ") mm"


def _round(v) -> str:
    return "(" + ", ".join(f"{float(x):+.2f}" for x in v) + ")"
