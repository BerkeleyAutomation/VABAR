"""How many syringes are still to be posted: the ones the overhead camera sees
lying away from the sharps box.

The loop's condition. A syringe that has been posted is inside the opaque box
and out of sight; one lying on the lid is a failed post the verifier already
rejected. So what remains is every syringe SAM finds whose depth cloud sits
farther than :data:`AWAY_FROM_APERTURE_M` from the aperture -- in the tray, on
the bench -- and below the box's rim. ``status`` is ``"more"`` or ``"none"``,
the subgraph's exit, which the workflow routes back to perception or to
``done``.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

SCORE_MIN = 0.08
AWAY_FROM_APERTURE_M = 0.12


class Output(TypedDict):
    remaining: int
    status: str
    centers: list[dict[str, float]]


def run(ctx: NodeContext, observation: dict[str, Any], aperture: dict[str, float], rim_z: float,
        object_description: str = "syringe") -> Output:
    overhead = next(c for c in observation["cameras"] if c.get("name") == "overhead")
    result = ctx.tool("sam3.segment_text", image=overhead["rgb"], query=object_description, max_results=4)
    aperture_xy = np.array([float(aperture["x"]), float(aperture["y"])])
    centers: list[dict[str, float]] = []
    for mask, score in zip(result.get("masks") or [], result.get("scores") or [], strict=False):
        if float(score) < SCORE_MIN:
            continue
        cloud = ctx.tool(
            "geometry.mask_to_world_points", mask=np.asarray(mask, dtype=np.uint8),
            depth=overhead["depth"], intrinsics=overhead["intrinsics"], camera_pose=overhead["pose"],
        )["points"]
        points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        if len(points) < 30:
            continue
        center = np.median(points, axis=0)
        if float(np.linalg.norm(center[:2] - aperture_xy)) <= AWAY_FROM_APERTURE_M or float(center[2]) >= float(rim_z):
            continue
        # one syringe, one count: SAM can return the same object twice
        if any(np.linalg.norm(center[:2] - np.array([c["x"], c["y"]])) < 0.03 for c in centers):
            continue
        centers.append({"x": float(center[0]), "y": float(center[1]), "z": float(center[2])})
    print(f"[remaining] {len(centers)} syringe(s) still away from the box")
    return {"remaining": len(centers), "status": "more" if centers else "none", "centers": centers}
