"""What the episode actually achieved, read off the evaluator's own verdict.

Runs once, after the cursor says there is nothing left it will attempt. It waits
first: the goal's other two clauses are borne load and rest, and a tool released
a moment ago is still swinging on its hook -- reading the score at that instant
reports a hang that has not finished happening. Then it reports the evaluator's
reward rather than recomputing one, because a graph that scores itself is a
graph that can be wrong about the only number anybody reads.

``hung`` is counted off the published geometry metrics rather than off the
reward, so a partial episode says which half worked. These documents group their
clauses one per tool under ``(fraction)``, so half a reward is one tool hung and
the count says which.

**Every placement atom is counted.** ``screwdrivers`` states ``threaded`` and
``wrenches``/``scissors`` state ``encloses``; reading one prefix reported
``clauses: 0`` on the other two and made every episode there look like a scene
with nothing to hang. The first two are ``HANG_ATOMS`` in the suite's
``tools.py``; ``containment`` is the third, which the level-1 documents state as
a region rather than a threading. All three are kept as literals here because a
graph script is read as shipped source.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str
    reward: float
    success: bool
    hung: int
    clauses: int


def run(ctx: NodeContext, settle_steps: int = 90) -> Output:
    ctx.tool("robot.wait_steps", steps=int(settle_steps))
    verdict = ctx.tool("sim.check_success")
    metrics = verdict.get("metrics") or {}
    geometry = [
        bool(v) for k, v in metrics.items() if str(k).startswith(("threaded/", "encloses/", "containment/"))
    ]
    hung = sum(1 for value in geometry if value)
    return {
        "route": "hung" if hung else "none",
        "reward": float(verdict["reward"]),
        "success": bool(verdict["success"]),
        "hung": hung,
        "clauses": len(geometry),
    }
