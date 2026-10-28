"""Discover a labelled 2x2 destination layout — local override for gap_perception_loop.

Identical to the bundle's discover_regions.py except it also outputs
``destination_bounds_json``: the 2D image-space bounding box of the destination
container detected by DINO. select_pair.py uses this to crop to the source bin
(the complementary region) instead of running a separate DINO call that
frequently finds the robot gripper.
"""

import json
import logging
import os
import re
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sorting_cv import box as _box  # noqa: E402
from sorting_cv import camera as _camera  # noqa: E402


class Output(TypedDict, total=False):
    layout_json: str
    destination_bounds_json: str
    attempted_json: str
    bin_floor_z: float


logger = logging.getLogger(__name__)

_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")


def _specific_container_box(
    detections: list[dict[str, Any]], image_shape: tuple[int, ...]
) -> dict[str, Any]:
    image_height, image_width = image_shape[:2]
    container_min_width = 0.25 * image_width
    container_min_height = 0.25 * image_height

    # The destination bin is on the right half of the image; the source bin is on
    # the left. Reject any candidate whose left edge starts in the left 40% of
    # the frame — that box spans the entire fixture rather than the destination.
    right_half_x1 = 0.40 * image_width

    # Prefer explicit destination/compartment/four-compartment labels — these
    # describe the whole container. Pick the largest to get the full bin footprint
    # rather than one half or one compartment.
    destination_groups = []
    for detection in detections:
        box = _box(detection)
        width = float(box["x2"]) - float(box["x1"])
        height = float(box["y2"]) - float(box["y1"])
        label = str(detection.get("label", "")).lower()
        if (
            float(box["x1"]) >= right_half_x1
            and width >= container_min_width
            and height >= container_min_height
            and ("destination" in label or "compartment" in label or "four" in label)
        ):
            destination_groups.append((width * height, box))
    if destination_groups:
        return max(destination_groups, key=lambda item: item[0])[1]

    # Fall back to "printed label" detections — pick the largest to capture the
    # full container rather than an individual compartment sub-region.
    label_min_width = 0.18 * image_width
    label_min_height = 0.20 * image_height
    label_groups = []
    for detection in detections:
        box = _box(detection)
        width = float(box["x2"]) - float(box["x1"])
        height = float(box["y2"]) - float(box["y1"])
        label = str(detection.get("label", "")).lower()
        if (
            float(box["x1"]) >= right_half_x1
            and width >= label_min_width
            and height >= label_min_height
            and "printed label" in label
        ):
            label_groups.append((width * height, box))
    if label_groups:
        return max(label_groups, key=lambda item: item[0])[1]

    return _box(max(detections, key=lambda item: float(item.get("score", 0.0))))


def _read_labels(
    ctx: NodeContext, image: Any, layout_description: str, attempts: int = 1
) -> dict[str, str]:
    answer = ""
    for attempt in range(max(1, int(attempts))):
        answer = ctx.tool(
            "vlm.query",
            image=image,
            prompt=(
                f"Locate {layout_description}. Read the printed label in each of its four "
                "compartments. Use IMAGE coordinates, not world directions. Reply exactly: "
                "LAYOUT: top-left=<label>; top-right=<label>; "
                "bottom-left=<label>; bottom-right=<label>."
            ),
        )["text"]
        found: dict[str, str] = {}
        for position in _POSITIONS:
            match = re.search(rf"{position}\s*=\s*([^;\n.]+)", str(answer), re.I)
            if match:
                found[position] = re.sub(r"\s+", " ", match.group(1).strip()).lower()
        if len(found) == 4 and len(set(found.values())) == 4:
            return found
        if attempt + 1 < max(1, int(attempts)):
            logger.warning("[discover_regions] re-asking for the layout labels: %r", answer)
    raise ValueError(f"could not read four unique destination labels: {answer!r}")


def _region_obb(
    ctx: NodeContext, camera: dict[str, Any], bounds: dict[str, Any], position: str
) -> tuple[dict[str, Any], float]:
    """Returns (obb, floor_z). floor_z is the median Z of the bin floor points."""
    x1 = int(round(float(bounds["x1"])))
    y1 = int(round(float(bounds["y1"])))
    x2 = int(round(float(bounds["x2"])))
    y2 = int(round(float(bounds["y2"])))
    inset_x = max(3, int(0.08 * (x2 - x1)))
    inset_y = max(3, int(0.08 * (y2 - y1)))
    middle_x, middle_y = (x1 + x2) // 2, (y1 + y2) // 2
    left, top = position.endswith("left"), position.startswith("top")
    xa, xb = (x1 + inset_x, middle_x - inset_x) if left else (middle_x + inset_x, x2 - inset_x)
    ya, yb = (y1 + inset_y, middle_y - inset_y) if top else (middle_y + inset_y, y2 - inset_y)
    mask = np.zeros(np.asarray(camera["depth"]).shape[:2], dtype=np.uint8)
    mask[max(0, ya) : max(0, yb), max(0, xa) : max(0, xb)] = 255
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=mask,
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        raise ValueError(f"too few depth points for destination region {position}")
    floor_cut = np.quantile(points[:, 2], 0.45)
    floor = points[points[:, 2] <= floor_cut + 0.004]
    if len(floor) < 10:
        floor = points
    floor_z = float(np.median(floor[:, 2]))
    low, high = np.quantile(floor, [0.10, 0.90], axis=0)
    center = 0.5 * (low + high)
    extent = np.maximum(0.5 * (high - low), [0.025, 0.025, 0.002])
    obb = {
        "center": dict(zip(("x", "y", "z"), map(float, center), strict=True)),
        "extent": dict(zip(("x", "y", "z"), map(float, extent), strict=True)),
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    return obb, floor_z


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    layout_description: str,
    camera_name: str = "overhead",
    max_read_attempts: int = 1,
    seed_attempt_log: bool = False,
) -> Output:
    camera = _camera(observation, camera_name)
    image = camera["rgb"]
    labels = _read_labels(ctx, image, layout_description, max_read_attempts)
    detections = (
        ctx.tool(
            "grounding-dino.detect", image=image, query=layout_description, box_threshold=0.10
        ).get("detections")
        or []
    )
    if not detections:
        raise ValueError(f"could not localize {layout_description!r}")
    bounds = _specific_container_box(detections, np.asarray(image).shape)
    floor_z_values = []
    regions = []
    for position, label in labels.items():
        obb, floor_z = _region_obb(ctx, camera, bounds, position)
        floor_z_values.append(floor_z)
        regions.append({"label": label, "image_position": position, "obb": obb})
    return {
        "layout_json": json.dumps(regions),
        "destination_bounds_json": json.dumps(bounds),
        "attempted_json": "[]" if seed_attempt_log else "",
        "bin_floor_z": float(np.median(floor_z_values)),
    }
