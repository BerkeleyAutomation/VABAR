"""Camera-only evidence that the released syringe passed below the aperture."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    verified: bool
    evidence: str


def run(ctx: NodeContext, observation: dict[str, Any], aperture: dict[str, float], rim_z: float) -> Output:
    overhead = next(c for c in observation["cameras"] if c.get("name") == "overhead")
    result = ctx.tool("sam3.segment_text", image=overhead["rgb"], query="syringe", max_results=4)
    # Every visible syringe is judged, not only the top-scored one: with a
    # second syringe still lying in the tray (the two-syringe task), the
    # top mask is that one, 30 cm from the box, and it must not read as the
    # posted syringe "remaining visible outside the aperture". Only a syringe
    # near the aperture counts -- above the rim it is lying on the lid.
    aperture_xy = np.array([aperture["x"], aperture["y"]])
    seen_near = 0
    for mask, score in zip(result.get("masks") or [], result.get("scores") or [], strict=False):
        if float(score) < 0.03:
            continue
        cloud = ctx.tool(
            "geometry.mask_to_world_points", mask=np.asarray(mask, dtype=np.uint8),
            depth=overhead["depth"], intrinsics=overhead["intrinsics"], camera_pose=overhead["pose"],
        )["points"]
        points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        if len(points) < 8:
            continue
        center = np.median(points, axis=0)
        if float(np.linalg.norm(center[:2] - aperture_xy)) > 0.10:
            continue  # another syringe, elsewhere on the bench
        seen_near += 1
        if float(center[2]) >= float(rim_z) - 0.005:
            raise RuntimeError("syringe remains visible outside or above the sharps-box aperture")
    if seen_near:
        return {"verified": True, "evidence": "visible syringe center is below the aperture"}
    # The opaque box hides a contained syringe. This evidence is considered
    # only after insertion and release have completed successfully.
    return {"verified": True, "evidence": "syringe disappeared after aperture release"}
