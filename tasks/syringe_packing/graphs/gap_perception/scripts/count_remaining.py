"""Keep weak detections uncertain; require two views before a negative result."""

import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

# Helpers shared by this graph's scripts live beside them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception_checks import xyz, detect, rod_like, distinct, expanded_bounds, summary  # noqa: E402


SCORE_MIN = 0.08
AWAY_FROM_APERTURE_M = 0.12


class Output(TypedDict):
    remaining: int
    status: str
    centers: list
    evidence: list


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    aperture: dict[str, float],
    rim_z: float,
    object_description: str = "medical syringe",
) -> Output:
    center = xyz(aperture)
    hand = xyz(ctx.tool("robot.get_ee_pose")["pose"]["position"])
    weak = []
    strong = []
    views = 0

    def eligible(d):
        return (
            d["center"][2] < rim_z
            and np.linalg.norm(d["center"][:2] - center[:2]) > AWAY_FROM_APERTURE_M
            and np.linalg.norm(d["center"] - hand) > 0.14
            and rod_like(d)
        )

    for camera in sorted(observation.get("cameras", []), key=lambda c: c["name"] != "overhead"):
        if camera["name"] not in ("overhead", "agentview"):
            continue
        views += 1
        items = [d for d in detect(ctx, camera, prompt=object_description) if eligible(d)]
        accepted = [d for d in items if d["score"] >= SCORE_MIN]
        if not accepted and items:
            crop = expanded_bounds(items, camera)
            recovered = [d for d in detect(ctx, camera, crop=crop, prompt=object_description) if eligible(d)]
            items += recovered
            accepted = [d for d in items if d["score"] >= SCORE_MIN]
        strong += accepted
        weak += [d for d in items if d["score"] < SCORE_MIN]
        if strong:
            found = distinct(strong)
            return {
                "remaining": len(found),
                "status": "more",
                "centers": [summary(d)["center"] for d in found],
                "evidence": [summary(d) for d in found],
            }
    if weak or views < 2:
        return {"remaining": -1, "status": "uncertain", "centers": [], "evidence": [summary(d) for d in distinct(weak)]}
    return {"remaining": 0, "status": "none", "centers": [], "evidence": [{"negative_views": views}]}

