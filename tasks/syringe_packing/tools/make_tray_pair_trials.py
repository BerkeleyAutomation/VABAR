# SPDX-License-Identifier: Apache-2.0
"""Write the ``tray_pair`` layout sweep as documents: one variant per syringe location.

The task language samples nothing per episode -- its only sampler is ``fixed``
-- so a sweep over where the syringe starts is a set of variants, each a
document of its own with its own ``start_poses`` layout, the way every other
variant in the corpus is stated. This writes them from ``tray_pair``'s
document and layout, with three independent draws per trial:

* the whole tray arrangement (goal syringe, spare, tray) shifted together in
  the bench plane (``--range``), so the goal stays in its tray and the spare
  keeps its spacing;
* the same arrangement yawed about the goal syringe (``--yaw-range``), so
  the barrel's heading -- and with it the grasp, the wrist turn and the
  carry -- differs from trial to trial;
* the sharps box shifted on the bench (``--box-range``), which the loader
  applies by rewriting the box geoms of the robot-stripped workcell, region
  following, exactly as the ``SHARPS_BOX_SHIFT`` sweep did.

Usage:
    uv run python tools/make_tray_pair_trials.py --count 30 --seed 0 --range 0.06 --yaw-range 25 --box-range 0.03
    uv run python tools/make_tray_pair_trials.py --clean     # remove generated trials

Re-running with the same seed rewrites identical files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TASK = ROOT / "tasks" / "sharps_disposal"
BASE_STEM = "dispose_the_syringe_from_the_tray"
TRIAL_STEM = BASE_STEM + "_t{index:02d}"
VARIANT_PREFIX = "tray_pair"
"""Defaults; ``--base`` and ``--prefix`` swap in another document family
(``dispose_both_syringes_from_the_tray`` / ``tray_pair_both``), same layouts."""
MIN_SPACING_M = 0.010
"""Least distance between any two trials' offsets [m], so the sweep is thirty
different places and not thirty draws that happen to coincide."""


def _quat_yaw(q_xyzw, yaw_rad: float):
    """``Rz(yaw) * q`` for an xyzw quaternion."""
    x, y, z, w = q_xyzw
    c, s = np.cos(yaw_rad / 2.0), np.sin(yaw_rad / 2.0)
    # (0, 0, s, c) * (x, y, z, w)
    return [
        round(float(c * x - s * y), 6),
        round(float(c * y + s * x), 6),
        round(float(c * z + s * w), 6),
        round(float(c * w - s * z), 6),
    ]


def _transform(layout: dict, dx: float, dy: float, yaw_rad: float) -> dict:
    """The arrangement yawed about the goal syringe, then shifted."""
    out = json.loads(json.dumps(layout))
    gx, gy, _ = out["objects"]["syringe_0"]["pose"]
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    for group in ("objects", "fixtures"):
        for entry in out.get(group, {}).values():
            x, y, z = entry["pose"]
            rx, ry = gx + c * (x - gx) - s * (y - gy), gy + s * (x - gx) + c * (y - gy)
            entry["pose"] = [round(rx + dx, 6), round(ry + dy, 6), z]
            entry["quat"] = _quat_yaw(entry["quat"], yaw_rad)
    return out


def write_trials(
    count: int, seed: int, extent: float, yaw_extent_deg: float = 0.0, box_extent: float = 0.0
) -> list[str]:
    base_doc = (TASK / f"{BASE_STEM}.task").read_text()
    base_layout = json.loads((TASK / "start_poses" / f"{VARIANT_PREFIX}.json").read_text())
    rng = np.random.default_rng(seed)
    # Every trial a different place: rejection-sample so no two offsets sit
    # within MIN_SPACING_M of each other, which a plain draw would allow.
    offsets: list[np.ndarray] = []
    while len(offsets) < count:
        candidate = rng.uniform(-extent, extent, size=2)
        if all(np.linalg.norm(candidate - o) >= MIN_SPACING_M for o in offsets):
            offsets.append(candidate)
    yaws = rng.uniform(-yaw_extent_deg, yaw_extent_deg, size=count) if yaw_extent_deg > 0 else np.zeros(count)
    boxes = rng.uniform(-box_extent, box_extent, size=(count, 2)) if box_extent > 0 else np.zeros((count, 2))
    names = []
    for index, (dx, dy) in enumerate(offsets):
        yaw_deg, (bx, by) = float(yaws[index]), (float(boxes[index][0]), float(boxes[index][1]))
        stem = TRIAL_STEM.format(index=index)
        variant = f"{VARIANT_PREFIX}_t{index:02d}"
        header = (
            f"; TRIAL {index:02d} of {count} in the {VARIANT_PREFIX} layout sweep: the whole tray\n"
            f"; arrangement yawed {yaw_deg:+.1f} deg about the goal syringe and shifted by\n"
            f"; ({dx:+.4f}, {dy:+.4f}) m in the bench plane; the sharps box shifted by\n"
            f"; ({bx:+.4f}, {by:+.4f}) m. All drawn from numpy default_rng(seed={seed}):\n"
            f"; shift uniform(+-{extent}) m, yaw uniform(+-{yaw_extent_deg}) deg, box uniform(+-{box_extent}) m.\n"
            f"; Written by tools/make_tray_pair_trials.py from {BASE_STEM}.task; the\n"
            f"; layout is start_poses/{variant}.json. Everything else is the base document's.\n"
        )
        doc = base_doc.replace(
            f"(define (problem {BASE_STEM.upper()})", f"(define (problem {BASE_STEM.upper()}_T{index:02d})", 1
        )
        doc = re.sub(r"^; -{20,}\n", header, doc, count=1, flags=re.M)
        (TASK / f"{stem}.task").write_text(doc)
        layout = _transform(base_layout, float(dx), float(dy), np.deg2rad(yaw_deg))
        layout["variant"] = variant
        if box_extent > 0:
            layout["box_shift"] = [round(bx, 6), round(by, 6)]
        layout["trial"] = {
            "index": index,
            "of": count,
            "seed": seed,
            "offset_m": [round(float(dx), 6), round(float(dy), 6)],
            "yaw_deg": round(yaw_deg, 3),
            "box_shift_m": [round(bx, 6), round(by, 6)],
        }
        (TASK / "start_poses" / f"{variant}.json").write_text(json.dumps(layout, indent=2) + "\n")
        names.append(variant)
    return names


def clean() -> int:
    removed = 0
    for path in list(TASK.glob(f"{BASE_STEM}_t*.task")) + list((TASK / "start_poses").glob(f"{VARIANT_PREFIX}_t*.json")):
        path.unlink()
        removed += 1
    return removed


def main() -> int:
    global BASE_STEM, TRIAL_STEM, VARIANT_PREFIX
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=BASE_STEM, help="document stem to derive the trials from")
    parser.add_argument("--prefix", default=VARIANT_PREFIX, help="variant prefix (start_poses/<prefix>.json is the base layout)")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--range", dest="extent", type=float, default=0.05, help="half-width of the uniform xy shift [m]"
    )
    parser.add_argument(
        "--yaw-range", type=float, default=0.0, help="half-width of the arrangement yaw about the goal [deg]"
    )
    parser.add_argument("--box-range", type=float, default=0.0, help="half-width of the sharps box shift [m]")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    BASE_STEM, VARIANT_PREFIX = args.base, args.prefix
    TRIAL_STEM = BASE_STEM + "_t{index:02d}"
    if args.clean:
        print(f"removed {clean()} files")
        return 0
    names = write_trials(args.count, args.seed, args.extent, args.yaw_range, args.box_range)
    print(f"wrote {len(names)} trial documents: {names[0]} .. {names[-1]}")
    for name in names:
        trial = json.loads((TASK / "start_poses" / f"{name}.json").read_text())["trial"]
        print(
            f"  {name}: shift ({trial['offset_m'][0]:+.4f}, {trial['offset_m'][1]:+.4f}) m, "
            f"yaw {trial['yaw_deg']:+6.1f} deg, box ({trial['box_shift_m'][0]:+.4f}, {trial['box_shift_m'][1]:+.4f}) m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
