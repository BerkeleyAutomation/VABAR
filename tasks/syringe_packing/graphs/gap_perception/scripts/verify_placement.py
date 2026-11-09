"""Inspect the cleared box from two views; absence is not proof of delivery."""

import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

# Helpers shared by this graph's scripts live beside them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception_checks import xyz, detect, crop_for_world_box, summary  # noqa: E402


ABOVE_RIM_TOLERANCE_M = 0.010
"""How far above the rim a syringe's points must reach to count as left above the
aperture. In sweep s7, t04 and t07 aborted after the unjam push over syringes standing
on the pile with their tops only +3 and +4 mm above the rim, 13-17 mm off the
hole's centre. Their centres were 70 mm inside the box, where the containment goal
counts them. The previous 3 mm margin made a full box read as a failed delivery."""


class Output(TypedDict):
    verified: bool
    status: str
    evidence: str
    observations: list
    ignored_robot_masks: int


def run(ctx: NodeContext, observation: dict[str, Any], aperture: dict[str, float], rim_z: float) -> Output:
    hand = xyz(ctx.tool("robot.get_ee_pose")["pose"]["position"])
    center = xyz(aperture)
    strong = []
    weak = []
    ignored = 0
    views = 0
    for camera in observation.get("cameras", []):
        if camera["name"] not in ("overhead", "agentview"):
            continue
        crop = crop_for_world_box(camera, center + [-0.11, -0.11, -0.12], center + [0.11, 0.11, 0.15])
        views += 1
        for d in detect(ctx, camera, crop=crop):
            if np.linalg.norm(d["center"][:2] - center[:2]) > 0.10:
                continue
            # A segmentation on the flat lid is not a syringe protruding above it.
            if np.mean(np.abs(d["points"][:, 2] - rim_z) < 0.001) > 0.90:
                continue
            # The live hand envelope, not a stale Franka planner sphere model.
            if np.linalg.norm(d["center"] - hand) < 0.14:
                ignored += 1
                continue
            (strong if d["score"] >= 0.03 else weak).append(d)
    if views < 2:
        raise RuntimeError("placement uncertain: two calibrated inspection views are required")

    def above_rim(d):
        above = d["points"][:, 2] > rim_z + ABOVE_RIM_TOLERANCE_M
        return np.count_nonzero(above) >= 8 and np.mean(above) >= 0.20

    outside = [d for d in strong if above_rim(d)]
    if outside:
        return {
            "verified": False,
            "status": "blocked",
            "evidence": "syringe remains above the aperture after release",
            "observations": [summary(d) for d in outside],
            "ignored_robot_masks": ignored,
        }
    if any(above_rim(d) for d in weak):
        return {
            "verified": False,
            "status": "uncertain",
            "evidence": "weak above-rim evidence requires another observation",
            "observations": [summary(d) for d in weak],
            "ignored_robot_masks": ignored,
        }
    if strong:
        return {
            "verified": True,
            "status": "verified",
            "evidence": "visible syringe centers are below the rim",
            "observations": [summary(d) for d in strong],
            "ignored_robot_masks": ignored,
        }
    return {
        "verified": False,
        "status": "clear",
        "evidence": "no visible obstruction in two views; delivery is not established by absence",
        "observations": [],
        "ignored_robot_masks": ignored,
    }

