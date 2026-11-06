"""Tool-sort–specific object selection: DINO → tournament → verify → label-assign → SAM3.

Custom script for gap_perception_loop. Candidate for promotion to
open-robot-skills/perceiving-sorting-pairs once the pipeline is validated.

Selection pipeline:
  1. DINO detects candidate items in the source-bin crop using remaining labels.
  2. Containment-NMS removes sub-region fragments (logos, labels).
  3. Bin-bounds and depth filters remove the gripper and crop-margin noise.
  4. Spent-position filter removes detections already in the attempt log.
  5. Pairwise VLM tournament picks the most accessible candidate.
  6. Close-up verify rejects the robot arm / gripper if the tournament misfired.
  7. Label-assign maps the winner to a destination label.
  8. SAM3 segments the winner box.
  9. geometry.exclude_robot_points removes FK-near robot points from the cloud.
"""

import json
import logging
import os
import re
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

# Resolve siblings from the bundle (open-robot-skills perceiving-sorting-pairs).
# The runtime puts the bundle's scripts/ on sys.path when staging; the fallback
# here handles running the script directly or outside the bundle loader.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from sorting_cv import bounds as _bounds  # noqa: E402
    from sorting_cv import box as _box  # noqa: E402
    from sorting_cv import camera as _camera  # noqa: E402
except ImportError:
    # Bundle loader already put sorting_cv on the path — this branch is for
    # direct invocations where the sibling is not available via sys.path[0].
    raise

logger = logging.getLogger(__name__)


class Output(TypedDict, total=False):
    status: str
    attempted_json: str
    target_name: str
    target_label: str
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_mask_overlay: np.ndarray
    dino_candidates_overlay: np.ndarray
    target_cloud: dict[str, Any]
    destination_obb: dict[str, Any]


_TILE = 384          # upscaled crop side for VLM pair sheets
_NMS_FRAGMENT_PX = 80


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _normalize(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def _parse_label_list(value: str) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    items = json.loads(text) if text.startswith("[") else text.split(",")
    return {_normalize(item) for item in items if _normalize(item)}


def _parse_canonical_targets(value: str) -> dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {}
    table = json.loads(text)
    if not isinstance(table, dict):
        raise ValueError("canonical_targets must be a JSON object mapping label to TARGET phrase")
    return {_normalize(label): str(target).strip().lower() for label, target in table.items()}


# ---------------------------------------------------------------------------
# Output sentinels
# ---------------------------------------------------------------------------

def _empty_result(status: str) -> Output:
    empty_obb = {
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "extent": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    return {
        "status": status,
        "attempted_json": "",
        "target_name": "",
        "target_label": "",
        "target_obb": empty_obb,
        "target_mask": np.zeros((1, 1), dtype=np.uint8),
        "target_mask_overlay": np.zeros((1, 1, 3), dtype=np.uint8),
        "dino_candidates_overlay": np.zeros((1, 1, 3), dtype=np.uint8),
        "target_cloud": {"points": np.empty((0, 3), dtype=np.float32)},
        "destination_obb": empty_obb,
    }


def _finished(attempts: list[dict[str, Any]] | None) -> Output:
    result = _empty_result("finished")
    result["attempted_json"] = json.dumps(attempts) if attempts is not None else ""
    return result


# ---------------------------------------------------------------------------
# Attempt log
# ---------------------------------------------------------------------------

def _load_attempts(attempted_json: str) -> list[dict[str, Any]]:
    try:
        loaded = json.loads(str(attempted_json or "[]"))
    except (TypeError, ValueError):
        return []
    return [e for e in loaded if isinstance(e, dict)] if isinstance(loaded, list) else []


def _matching_attempt(
    attempts: list[dict[str, Any]], px: float, py: float, rx: float, ry: float
) -> dict[str, Any] | None:
    for entry in attempts:
        try:
            if abs(float(entry["px"]) - px) <= max(rx, 8.0) and \
               abs(float(entry["py"]) - py) <= max(ry, 8.0):
                return entry
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _settled_region(obb: dict[str, Any], regions: list[dict[str, Any]], margin: float):
    try:
        cx, cy = float(obb["center"]["x"]), float(obb["center"]["y"])
    except (KeyError, TypeError, ValueError):
        return None
    for region in regions:
        b = region.get("obb") or {}
        try:
            rx, ry = float(b["center"]["x"]), float(b["center"]["y"])
            ex, ey = float(b["extent"]["x"]) + margin, float(b["extent"]["y"]) + margin
        except (KeyError, TypeError, ValueError):
            continue
        if abs(cx - rx) <= ex and abs(cy - ry) <= ey:
            return region
    return None


# ---------------------------------------------------------------------------
# Source-bin detection
# ---------------------------------------------------------------------------

def _source_box(
    ctx: NodeContext,
    image: Any,
    query: str,
    destination_bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    detections = (
        ctx.tool("grounding-dino.detect", image=image, query=query, box_threshold=0.10)
        .get("detections") or []
    )

    def _overlap_fraction(b: dict, d: dict) -> float:
        ix1 = max(float(b["x1"]), float(d["x1"]))
        iy1 = max(float(b["y1"]), float(d["y1"]))
        ix2 = min(float(b["x2"]), float(d["x2"]))
        iy2 = min(float(b["y2"]), float(d["y2"]))
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(1.0, (float(b["x2"]) - float(b["x1"])) * (float(b["y2"]) - float(b["y1"])))
        return inter / area

    def _adjacency_gap(b: dict, d: dict) -> float:
        """Minimum pixel distance between any edge of b and any edge of d.
        A shared boundary scores near zero; an unrelated box scores large."""
        return min(
            abs(float(b["x2"]) - float(d["x1"])),
            abs(float(b["x1"]) - float(d["x2"])),
            abs(float(b["y2"]) - float(d["y1"])),
            abs(float(b["y1"]) - float(d["y2"])),
        )

    plausible = []
    for det in detections:
        box = _box(det)
        x1, y1, x2, y2 = _bounds(box)
        label = str(det.get("label", "")).lower()
        if not (x2 - x1 >= 100 and y2 - y1 >= 100):
            continue
        if not any(t in label for t in ("source", "tool", "bin", "pile")):
            continue
        if destination_bounds and _overlap_fraction(box, destination_bounds) > 0.5:
            continue
        gap = _adjacency_gap(box, destination_bounds) if destination_bounds else 0.0
        score = float(det.get("score", 0.0))
        plausible.append((gap, -score, box))

    if not plausible:
        raise ValueError(f"could not localize {query!r}")
    # Primary: smallest edge gap (adjacent to destination). Secondary: highest confidence.
    return min(plausible, key=lambda item: (item[0], item[1]))[2]


# ---------------------------------------------------------------------------
# Detection filtering
# ---------------------------------------------------------------------------

def _det_center(det: dict[str, Any]) -> tuple[float, float]:
    b = _box(det)
    return (float(b["x1"]) + float(b["x2"])) / 2.0, (float(b["y1"]) + float(b["y2"])) / 2.0


def _drop_contained(
    detections: list[dict[str, Any]],
    thresh: float = 0.7,
    max_small: int = _NMS_FRAGMENT_PX,
) -> list[dict[str, Any]]:
    """Size-bounded containment NMS: drop small sub-region boxes (logos/labels)."""
    def _area(b: dict) -> float:
        return max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    def _inter(a: dict, b: dict) -> float:
        return max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])) * \
               max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))

    kept = []
    for i, di in enumerate(detections):
        bi, ai = _box(di), _area(_box(di))
        if ai <= 0:
            continue
        contained = any(
            j != i and _area(_box(dj)) > ai and
            _inter(bi, _box(dj)) / ai >= thresh and
            max(bi["x2"] - bi["x1"], bi["y2"] - bi["y1"]) < max_small
            for j, dj in enumerate(detections)
        )
        if not contained:
            kept.append(di)
    return kept


def _drop_near_duplicates(
    detections: list[dict[str, Any]], iou_thresh: float = 0.5
) -> list[dict[str, Any]]:
    """IoU NMS: among boxes that overlap substantially, keep the highest-confidence one."""
    def _area(b: dict) -> float:
        return max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    def _iou(a: dict, b: dict) -> float:
        inter = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])) * \
                max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
        if inter == 0.0:
            return 0.0
        union = _area(a) + _area(b) - inter
        return inter / union if union > 0 else 0.0

    scored = sorted(detections, key=lambda d: -float(d.get("score", 0.0)))
    suppressed: set[int] = set()
    kept = []
    for i, di in enumerate(scored):
        if i in suppressed:
            continue
        kept.append(di)
        bi = _box(di)
        for j in range(i + 1, len(scored)):
            if j not in suppressed and _iou(bi, _box(scored[j])) >= iou_thresh:
                suppressed.add(j)
    return kept


def _drop_containers(
    detections: list[dict[str, Any]], min_contained: int = 2
) -> list[dict[str, Any]]:
    """Drop any box whose interior contains the centres of >= min_contained other boxes.

    This removes the bin box (which wraps all the tools) while leaving the
    individual tool detections untouched.
    """
    centers = [_det_center(d) for d in detections]
    kept = []
    for i, di in enumerate(detections):
        bi = _box(di)
        n_inside = sum(
            1 for j, (cx, cy) in enumerate(centers)
            if j != i and bi["x1"] < cx < bi["x2"] and bi["y1"] < cy < bi["y2"]
        )
        if n_inside < min_contained:
            kept.append(di)
        else:
            logger.debug(
                "select_pair: dropping container box (contains %d centres): %s", n_inside, bi
            )
    return kept


def _filter_spent(
    detections: list[dict[str, Any]],
    attempts: list[dict[str, Any]] | None,
    crop_x: float,
    crop_y: float,
    max_attempts: int = 3,
) -> list[dict[str, Any]]:
    if not attempts:
        return detections
    kept = []
    for det in detections:
        cx, cy = _det_center(det)
        b = _box(det)
        rx = (float(b["x2"]) - float(b["x1"])) / 2.0
        ry = (float(b["y2"]) - float(b["y1"])) / 2.0
        entry = _matching_attempt(attempts, cx + crop_x, cy + crop_y, rx, ry)
        if entry is None or int(entry.get("attempts", 0)) < max_attempts:
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# VLM helpers
# ---------------------------------------------------------------------------

def _upscale_letterbox(crop: np.ndarray, size: int = _TILE) -> np.ndarray:
    from PIL import Image as PILImage
    h, w = crop.shape[:2]
    s = size / max(h, w, 1)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    resized = np.asarray(
        PILImage.fromarray(crop).resize(
            (nw, nh), PILImage.BICUBIC if s > 1 else PILImage.LANCZOS
        )
    )
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _crop_region(image: np.ndarray, box: dict[str, Any], pad: float = 0.30) -> np.ndarray | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _bounds(box)
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    cx1 = max(0, int(x1) - int(pad * bw))
    cy1 = max(0, int(y1) - int(pad * bh))
    cx2 = min(w, int(x2) + int(pad * bw))
    cy2 = min(h, int(y2) + int(pad * bh))
    return image[cy1:cy2, cx1:cx2].copy() if cx2 > cx1 and cy2 > cy1 else None


def _make_pair_sheet(crop_a: np.ndarray, crop_b: np.ndarray, size: int = _TILE) -> np.ndarray:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    pad = 6
    cell = size + 2 * pad
    sheet = np.full((cell, 2 * cell, 3), 255, dtype=np.uint8)
    for i, (crop, color, lbl) in enumerate(zip(
        (crop_a, crop_b), ((220, 50, 50), (50, 180, 50)), ("A", "B")
    )):
        tile = _upscale_letterbox(crop, size)
        img = PILImage.fromarray(tile)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
            )
        except Exception:
            font = ImageFont.load_default()
        draw.rectangle([0, 0, size - 1, 36], fill=color)
        draw.text((8, 4), lbl, fill=(255, 255, 255), font=font)
        x = i * cell + pad
        sheet[pad:pad + size, x:x + size] = np.asarray(img)
    return sheet


def _draw_candidates(image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Draw all DINO candidate boxes on the image, numbered by detection score rank."""
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont
        img = PILImage.fromarray(image.copy())
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
            )
        except Exception:
            font = ImageFont.load_default()
        for i, det in enumerate(detections):
            b = _box(det)
            x1, y1, x2, y2 = _bounds(b)
            score = float(det.get("score", 0.0))
            draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 255), width=2)
            label = det.get("label", "")
            draw.text((x1 + 2, y1 + 2), f"{i+1} {label} {score:.2f}", fill=(0, 200, 255), font=font)
        return np.asarray(img)
    except Exception:
        return image.copy()


def _tournament(
    ctx: Any,
    detections: list[dict[str, Any]],
    source_crop: np.ndarray,
    remaining_labels: list[str],
) -> int:
    """Single-elimination pairwise tournament. Returns winner index into detections, or -1."""
    crops, keep = [], []
    for i, det in enumerate(detections[:8]):
        cr = _crop_region(source_crop, _box(det))
        if cr is not None:
            crops.append(cr)
            keep.append(i)
    if not crops:
        return -1
    if len(crops) == 1:
        return keep[0]

    label_str = ", ".join(remaining_labels)
    bracket = list(range(len(crops)))
    while len(bracket) > 1:
        nxt: list[int] = []
        for j in range(0, len(bracket), 2):
            if j + 1 >= len(bracket):
                nxt.append(bracket[j])
                continue
            a, b = bracket[j], bracket[j + 1]
            sheet = _make_pair_sheet(crops[a], crops[b])
            resp = ctx.tool(
                "vlm.query",
                image=sheet,
                prompt=(
                    f"Two candidate objects are shown side by side: A (left, red header) "
                    f"and B (right, green header). "
                    f"Destination labels to sort into are: {label_str}. "
                    "Which object should be sorted next? "
                    "Prefer the most accessible: topmost, clearly visible, and graspable "
                    "by a parallel-jaw gripper. "
                    "Reply with exactly A or B."
                ),
            )["text"].strip().upper()
            nxt.append(a if resp.startswith("A") else b)
        bracket = nxt
        if not bracket:
            return -1
    return keep[bracket[0]] if bracket else -1


def _verify(ctx: Any, crop: np.ndarray, remaining_labels: list[str]) -> bool:
    """Yes/no: is this a graspable tool (not the robot) matching one of the labels?"""
    label_str = ", ".join(remaining_labels)
    img = _upscale_letterbox(crop)
    try:
        resp = ctx.tool(
            "vlm.query_yes_no",
            image=img,
            prompt=(
                f"Is the main object in this close-up a graspable tool "
                f"(not a robot arm, gripper, or robot part) "
                f"matching one of these labels: {label_str}?"
            ),
        )
        return bool(resp["answer"])
    except Exception:
        resp = ctx.tool(
            "vlm.query",
            image=img,
            prompt=(
                f"Is the main object in this close-up a graspable tool "
                f"(not a robot arm, gripper, or robot part) "
                f"matching one of these labels: {label_str}? Answer YES or NO."
            ),
        )["text"]
        return str(resp).strip().upper().startswith("Y")


def _assign_label(
    ctx: Any,
    crop: np.ndarray,
    remaining_labels: list[str],
    normalized: dict[str, Any],
    excluded: set[str],
) -> tuple[str, str] | None:
    """Map the tournament winner to a destination label. Returns (norm_key, raw_label) or None."""
    label_str = ", ".join(remaining_labels)
    img = _upscale_letterbox(crop)
    resp = ctx.tool(
        "vlm.query",
        image=img,
        prompt=(
            f"What is the object shown? Pick the closest matching label from this list: "
            f"{label_str}. "
            "You must pick one — reply exactly: LABEL: <label>."
        ),
    )["text"]
    m = re.search(r"LABEL\s*:\s*(.+)", str(resp), re.I)
    if not m:
        return None
    raw = m.group(1).strip().rstrip(".")
    key = _normalize(raw)
    if key in normalized and key not in excluded:
        return key, raw
    # Fuzzy match against remaining labels
    for lbl in remaining_labels:
        nk = _normalize(lbl)
        if nk == key or key in nk or nk in key:
            return nk, lbl
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    instruction: str,
    layout_json: str,
    source_description: str,
    camera_name: str = "overhead",
    destination_bounds_json: str = "",
    source_query: str = "",
    exclude_label_1: str = "",
    exclude_label_2: str = "",
    exclude_label_3: str = "",
    identity_hints: str = "",
    canonical_targets: str = "",
    allowed_labels: str = "",
    exclusion_mode: str = "label",
    attempted_json: str = "[]",
    max_attempts_per_instance: int = 3,
    max_total_attempts: int = 40,
    settled_margin_m: float = 0.0,
    min_box_px: float = 8.0,
    min_box_fraction: float = 0.0,
) -> Output:
    regions = json.loads(layout_json)
    if not regions:
        raise ValueError("sorting layout has no labelled regions")
    camera = _camera(observation, camera_name)
    image = camera["rgb"]
    allowed = _parse_label_list(allowed_labels)
    if allowed:
        regions = [r for r in regions if _normalize(r["label"]) in allowed]
        if not regions:
            raise ValueError(f"none of the layout labels is in allowed_labels {sorted(allowed)!r}")
    labels = [str(r["label"]) for r in regions]
    excluded = {_normalize(v) for v in (exclude_label_1, exclude_label_2, exclude_label_3) if v}
    remaining_labels = [l for l in labels if _normalize(l) not in excluded]
    if not remaining_labels:
        return _empty_result("finished")
    canonical = _parse_canonical_targets(canonical_targets)
    by_instance = str(exclusion_mode).strip().lower() == "instance"
    attempts = _load_attempts(attempted_json) if by_instance else None
    all_regions = json.loads(layout_json)
    normalized = {_normalize(l): r for l, r in zip(labels, regions, strict=True)}
    if by_instance and len(attempts) >= max(1, int(max_total_attempts)):
        return _finished(attempts)

    # --- source bin crop ---
    # Derive the source crop as the region of the image that is NOT the
    # destination container. discover_regions already localised the destination
    # container via DINO; its 2D image-space bounds are passed here so we can
    # crop to the complementary half of the image without a second DINO call.
    image_array = np.asarray(image)
    img_h, img_w = image_array.shape[:2]
    # Hard crop to left half and middle two-thirds vertically — the source bin
    # is on the left, and trimming the top/bottom sixths removes the robot arm
    # and table edges from the DINO input.
    crop_x, crop_y = 0, img_h // 6
    crop_x2, crop_y2 = img_w // 2, (img_h * 5) // 6
    source_crop = image_array[crop_y:crop_y2, crop_x:crop_x2]
    crop_h, crop_w = source_crop.shape[:2]

    # --- DINO item detection ---
    # Use a broad generic prompt so DINO finds all objects in the bin; the VLM
    # tournament handles semantic discrimination. Label-specific queries fail for
    # unusual names (e.g. "socket head screw m6x60") that DINO was not trained on.
    dino_query = source_query.strip() or "object . tool . item"
    item_detections: list[dict[str, Any]] = (
        ctx.tool(
            "grounding-dino.detect", image=source_crop, query=dino_query, box_threshold=0.10
        ).get("detections") or []
    )

    # Containment NMS disabled: tools inside a bin would be incorrectly dropped
    # as sub-region fragments of the bin box. Re-enable if DINO starts emitting
    # logo/label sub-regions for individual tools.
    # item_detections = _drop_contained(item_detections)

    # Collapse near-duplicate bin boxes (DINO fires on both the inner and outer
    # bin walls as separate detections; IoU NMS keeps the higher-confidence one).
    item_detections = _drop_near_duplicates(item_detections, iou_thresh=0.5)
    # Drop bin-sized boxes: any detection covering more than 30% of the source
    # crop must be the bin itself, since every object fits inside the bin.
    crop_area = float(crop_w * crop_h)
    item_detections = [
        d for d in item_detections
        if (lambda b: max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"]))(_box(d)) < 0.30 * crop_area
    ]
    # Drop whichever bin box survives via containment — identified as any box
    # whose interior contains the centres of >= 3 other detections.
    item_detections = _drop_containers(item_detections, min_contained=3)

    # Returned as an output, not written here: the runtime saves every
    # `*_overlay` into the episode's own trace. A `Path.cwd()/outputs/debug`
    # write is one path shared by every process, so a sweep running thirty
    # documents at once has thirty selections overwriting one file.
    dino_candidates_overlay = _draw_candidates(source_crop, item_detections)

    # Bin-bounds filter: reject detections whose centre falls outside the actual
    # source-bin box (not the padded crop margin). Disabled for now — in
    # practice the crop already constrains the detection area and this filter
    # was eliminating real objects when the DINO source-bin box underfit.
    # Re-enable and tune once the core pipeline is validated.
    # bin_cx1, bin_cy1 = float(sx1) - crop_x, float(sy1) - crop_y
    # bin_cx2, bin_cy2 = float(sx2) - crop_x, float(sy2) - crop_y

    # Depth floor filter: reject detections significantly above the bin floor
    # (e.g. the gripper). Disabled for now — the 90th-percentile estimate can
    # be thrown off by bin walls and the 30 cm threshold was too tight for
    # stacked industrial tools. Re-enable once the basic detection pipeline works.
    # depth_array = np.asarray(camera["depth"])
    # ...
    # item_detections = [d for d in item_detections if _keep(d)]

    if not item_detections:
        logger.warning("select_pair: no DINO detections passed filters")
        result = _finished(attempts)
        result["dino_candidates_overlay"] = dino_candidates_overlay
        return result

    # Remove spent positions from the candidate pool
    candidates = _filter_spent(item_detections, attempts, crop_x, crop_y, max_attempts_per_instance)
    if not candidates:
        result = _finished(attempts)
        result["dino_candidates_overlay"] = dino_candidates_overlay
        return result

    # --- tournament → label-assign → SAM3 loop ---
    # Candidates are retried up to max_attempts_per_instance times before being
    # dropped. Exhausting all candidates returns "retry" (not "finished") so the
    # outer loop re-observes rather than terminating.
    candidate_attempts: dict[int, int] = {}
    key = requested_label = wb = win_crop = full_box = full_px = full_py = None
    mask = cloud = obb = None

    while candidates:
        win_idx = _tournament(ctx, candidates, source_crop, remaining_labels)
        if win_idx < 0:
            logger.warning("select_pair: tournament returned no winner")
            break

        winner = candidates[win_idx]
        cid = id(winner)
        candidate_attempts[cid] = candidate_attempts.get(cid, 0) + 1

        wb = _box(winner)
        win_crop = _crop_region(source_crop, wb)

        # Label assignment
        label_result = _assign_label(ctx, win_crop, remaining_labels, normalized, excluded) \
            if win_crop is not None else None
        if label_result is None:
            logger.info("select_pair: no label matched (attempt %d)", candidate_attempts[cid])
            if candidate_attempts[cid] >= max_attempts_per_instance:
                candidates.pop(win_idx)
            continue

        key, requested_label = label_result
        if canonical.get(key) and _normalize(requested_label) != _normalize(canonical[key]):
            logger.info("select_pair: canonical mismatch for %r (attempt %d)", key, candidate_attempts[cid])
            if candidate_attempts[cid] >= max_attempts_per_instance:
                candidates.pop(win_idx)
            continue

        # Full-image box and center point for SAM3
        full_box = {
            "x1": float(wb["x1"]) + crop_x,
            "y1": float(wb["y1"]) + crop_y,
            "x2": float(wb["x2"]) + crop_x,
            "y2": float(wb["y2"]) + crop_y,
        }
        full_px = (full_box["x1"] + full_box["x2"]) / 2.0
        full_py = (full_box["y1"] + full_box["y2"]) / 2.0

        segmented = ctx.tool(
            "sam3.segment_box",
            image=image,
            box=full_box,
            pixel_x=full_px,
            pixel_y=full_py,
            use_point=True,
        )
        if not segmented.get("masks"):
            logger.warning("select_pair: SAM3 returned no mask (attempt %d)", candidate_attempts[cid])
            if candidate_attempts[cid] >= max_attempts_per_instance:
                candidates.pop(win_idx)
            continue

        sam_cands = [
            (float(segmented["scores"][i]), i)
            for i, c in enumerate(segmented["masks"])
            if int(np.count_nonzero(np.asarray(c))) >= 100
        ]
        if not sam_cands:
            logger.warning("select_pair: SAM3 masks all too small (attempt %d)", candidate_attempts[cid])
            if candidate_attempts[cid] >= max_attempts_per_instance:
                candidates.pop(win_idx)
            continue

        mask = np.asarray(segmented["masks"][max(sam_cands)[1]], dtype=np.uint8)
        cloud = ctx.tool(
            "geometry.mask_to_world_points",
            mask=mask,
            depth=camera["depth"],
            intrinsics=camera["intrinsics"],
            camera_pose=camera["pose"],
        )["points"]

        try:
            obs = ctx.tool("robot.get_observation")
            joints = obs["arms"][0]["joint_state"]
            before = len(cloud["points"])
            cloud = ctx.tool(
                "geometry.exclude_robot_points",
                points=cloud,
                joint_positions=joints,
            )["points"]
            removed = before - len(cloud["points"])
            if removed:
                logger.info("select_pair: excluded %d robot-near points (%d remain)",
                            removed, len(cloud["points"]))
        except Exception as exc:
            logger.warning("select_pair: robot-point exclusion skipped: %s", exc)

        obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]

        if by_instance:
            half_x = (full_box["x2"] - full_box["x1"]) / 2.0
            half_y = (full_box["y2"] - full_box["y1"]) / 2.0
            previous = _matching_attempt(attempts, full_px, full_py, half_x, half_y)
            if previous is None:
                attempts.append({"label": requested_label, "px": full_px, "py": full_py,
                                 "rx": half_x, "ry": half_y, "attempts": 1, "reason": "tried"})
            else:
                previous["attempts"] = int(previous.get("attempts", 0)) + 1
                previous.update({"px": full_px, "py": full_py})
        break  # success — exit the loop

    # All candidates exhausted or no winner found
    if mask is None:
        logger.warning("select_pair: no candidate produced a valid grasp — retry")
        result = _empty_result("retry")
        result["attempted_json"] = json.dumps(attempts) if by_instance else ""
        result["dino_candidates_overlay"] = dino_candidates_overlay
        return result

    # Visualization: SAM mask + DINO box + point prompt
    overlay = np.asarray(camera["rgb"], dtype=np.uint8).copy()
    mask_bool = mask.astype(bool)
    color = np.array([255, 80, 0], dtype=np.uint8)
    overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * color).astype(np.uint8)
    try:
        from PIL import Image as PILImage, ImageDraw
        _img = PILImage.fromarray(overlay)
        _draw = ImageDraw.Draw(_img)
        _draw.rectangle(
            [full_box["x1"], full_box["y1"], full_box["x2"], full_box["y2"]],
            outline=(0, 255, 0), width=2,
        )
        r = 5
        _draw.ellipse([full_px - r, full_py - r, full_px + r, full_py + r], fill=(0, 255, 0))
        overlay = np.asarray(_img)
    except Exception:
        pass

    found: Output = {
        "status": "found",
        "target_name": requested_label,
        "target_label": str(normalized[key].get("label", requested_label)),
        "target_obb": obb,
        "target_mask": mask,
        "target_mask_overlay": overlay,
        "dino_candidates_overlay": dino_candidates_overlay,
        "target_cloud": cloud,
        "destination_obb": normalized[key]["obb"],
    }
    found["attempted_json"] = json.dumps(attempts) if by_instance else ""
    return found
