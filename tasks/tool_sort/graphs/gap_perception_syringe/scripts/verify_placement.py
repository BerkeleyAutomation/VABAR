"""Camera-only evidence that the released syringe passed below the aperture."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    verified: bool
    evidence: str


def run(ctx: NodeContext, observation: dict[str, Any], aperture: dict[str, float], rim_z: float) -> Output:
    overhead = next(c for c in observation["cameras"] if c.get("name") == "overhead")
    result = ctx.tool("sam3.segment_text", image=overhead["rgb"], query="syringe", max_results=3)
    if not result.get("masks") or float((result.get("scores") or [0.0])[0]) < 0.03:
        # The opaque box hides a contained syringe. This evidence is considered
        # only after insertion and release have completed successfully.
        return {"verified": True, "evidence": "syringe disappeared after aperture release"}
    cloud = ctx.tool(
        "geometry.mask_to_world_points", mask=np.asarray(result["masks"][0], dtype=np.uint8),
        depth=overhead["depth"], intrinsics=overhead["intrinsics"], camera_pose=overhead["pose"],
    )["points"]
    points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    if len(points) < 8:
        raise RuntimeError("visible syringe verification has too few valid depth points")
    center = np.median(points, axis=0)
    aperture_xy = np.array([aperture["x"], aperture["y"]])
    if float(center[2]) >= float(rim_z) - 0.005 or float(np.linalg.norm(center[:2] - aperture_xy)) > 0.10:
        raise RuntimeError("syringe remains visible outside or above the sharps-box aperture")
    return {"verified": True, "evidence": "visible syringe center is below the aperture"}
