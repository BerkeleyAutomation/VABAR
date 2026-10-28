# SPDX-License-Identifier: Apache-2.0
"""Write the ``tray_clutter`` layout sweep: 5-8 syringes packed into one tray.

``tray_pair``'s 30-trial sweep (``make_tray_pair_trials.py``) moves one rigid
two-syringe template around the bench. This document's sweep is a different
kind of variation -- the *population* of the tray, not its placement -- so
each trial is a fresh packing rather than a transform of a shared template.

**Placement rule.** The tray's cavity is 200x400 mm (``task.py``'s
``TRAY_OUTER_HALF``/``TRAY_WALL_THICKNESS``); a syringe is a 150x23.6 mm
capsule (``SYRINGE_HALF_LENGTH``/``SYRINGE_HALF_WIDTH``) that only fits the
200 mm axis lying dead straight across it, so every syringe's long axis runs
along that short axis and the tray's 400 mm axis is the one row has to
fit them in. ``tray_pair`` used a fixed 90 mm pitch there (barrel width
23.6 mm, so mostly clear air, by design -- see its own header). This sweep
instead:

1. spaces N syringes on an even grid across the row (pitch shrinks as N
   grows, 5 -> ~94 mm, 8 -> ~53 mm -- more syringes reads as more cluttered
   on its own before any jitter is added);
2. jitters each one's row position, cross-axis position and heading
   independently, generously enough (``Y_JITTER_FRAC`` of its own pitch,
   ``X_JITTER_M``, ``YAW_JITTER_DEG``) that neighbours' jitter ranges
   overlap -- barrels actually crossing is the cluttered look asked for,
   not a printed grid nudged off-true;
3. checks every pair with a 2-D oriented-box separating-axis test and, only
   for a pair overlapping by more than ``ALLOWED_PENETRATION_M`` (deep
   enough that Newton's contact solver would eject one of them on the first
   settle step rather than just letting them rest across each other),
   halves both syringes' jitter and rechecks -- a bounded local repair
   rather than resampling the whole trial.

The tray itself stays at ``tray_pair``'s own fixed pose for all 30 trials
(only the sharps box gets a small independent shift, as ``tray_pair`` also
draws one) -- the sweep here is what's *in* the tray, not where the tray
sits; see the header on ``dispose_the_syringe_from_the_cluttered_tray.task``.

Usage:
    uv run python tools/make_tray_clutter_trials.py --count 30 --seed 5
    uv run python tools/make_tray_clutter_trials.py --clean     # remove generated trials
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TASK = ROOT / "tasks" / "sharps_disposal"
BASE_STEM = "dispose_the_syringe_from_the_cluttered_tray"
TRIAL_STEM = BASE_STEM + "_t{index:02d}"
VARIANT_PREFIX = "tray_clutter"
REFERENCE_LAYOUT = "tray_pair"
"""Where the tray's fixed pose and the shared "parallel" orientation come
from -- this sweep never writes its own base layout, only trial ones."""

MIN_SYRINGES = 5
MAX_SYRINGES = 8

# Tray geometry, mirroring task.py's TRAY_OUTER_HALF/TRAY_WALL_THICKNESS/
# SYRINGE_HALF_LENGTH/SYRINGE_HALF_WIDTH -- duplicated rather than imported
# since task.py is a package module and this is a standalone CLI tool.
TRAY_OUTER_HALF_M = (0.100, 0.200)
TRAY_WALL_THICKNESS_M = 0.003
SYRINGE_HALF_LENGTH_M = 0.075
SYRINGE_HALF_WIDTH_M = 0.011783

ROW_MARGIN_M = 0.010
"""Inset from the cavity's long-axis wall the row itself stays inside of."""
CROSS_MARGIN_M = 0.008
"""Inset from the cavity's short-axis wall a barrel's tip must clear."""
CAV_X_M = TRAY_OUTER_HALF_M[0] - TRAY_WALL_THICKNESS_M - CROSS_MARGIN_M
"""Half-width left for a barrel's own length along the tray's short axis."""
CAV_Y_M = TRAY_OUTER_HALF_M[1] - TRAY_WALL_THICKNESS_M - ROW_MARGIN_M
"""Half-length left for the row along the tray's long axis."""

Y_JITTER_FRAC = 0.55
"""Row-position jitter, as a fraction of that trial's own pitch -- large
enough that neighbours' jitter ranges overlap, which is the point: a pile
reads as cluttered when barrels actually cross, not just crowd."""
X_JITTER_M = 0.018
"""Cross-axis jitter. 18 mm plus a 35 deg-yawed barrel's own footprint
(~68 mm half-length along this axis) still clears CAV_X_M's ~89 mm."""
YAW_JITTER_DEG = 35.0
ALLOWED_PENETRATION_M = 0.016
"""How deep two syringes may actually overlap before the repair loop eases
them apart -- most of a 23.6 mm body's width, so barrels crossing is the
normal look and only a near-coincident pair (a placement physics would
throw across the tray on the first settle step) gets pulled back. Passed
to :func:`_obb_overlap` as a *negative* clearance, which shrinks the test
rectangles rather than growing them."""
MAX_REPAIR_ROUNDS = 6


def _quat_yaw(q_xyzw, yaw_rad: float):
    """``Rz(yaw) * q`` for an xyzw quaternion."""
    x, y, z, w = q_xyzw
    c, s = np.cos(yaw_rad / 2.0), np.sin(yaw_rad / 2.0)
    return [
        round(float(c * x - s * y), 6),
        round(float(c * y + s * x), 6),
        round(float(c * z + s * w), 6),
        round(float(c * w - s * z), 6),
    ]


def _obb_axes(angle: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c, s]), np.array([-s, c])


def _obb_overlap(a: tuple[float, float, float], b: tuple[float, float, float], clearance: float) -> bool:
    """Two syringes, as ``(x, y, yaw_rad)`` about their shared baseline
    heading, with half-extents (length, width) inflated by ``clearance``.
    Separating-axis test over both rectangles' own two axes."""
    half = (SYRINGE_HALF_LENGTH_M + clearance / 2.0, SYRINGE_HALF_WIDTH_M + clearance / 2.0)
    ca = np.array(a[:2])
    cb = np.array(b[:2])
    axes_a, axes_b = _obb_axes(a[2]), _obb_axes(b[2])
    d = cb - ca
    for ux, uy in (axes_a, axes_b):
        for axis in (ux, uy):
            proj_d = abs(np.dot(d, axis))
            extent_a = abs(np.dot(axes_a[0], axis)) * half[0] + abs(np.dot(axes_a[1], axis)) * half[1]
            extent_b = abs(np.dot(axes_b[0], axis)) * half[0] + abs(np.dot(axes_b[1], axis)) * half[1]
            if proj_d > extent_a + extent_b:
                return False  # a separating axis exists: no overlap
    return True


def _pack_row(n: int, rng: np.random.Generator) -> list[tuple[float, float, float]]:
    """``n`` syringes as ``(local_x, local_y, yaw_rad)`` in the tray's own
    frame (x = short/cross axis, y = long/row axis, baseline heading = 0,
    i.e. barrel along x -- see module docstring).

    Every jittered component is tracked as an offset from a safe nominal
    (nominal y on the even grid, x and yaw nominally 0) so a repair round
    can ease a colliding pair back *toward that nominal* -- which is
    collision-free by construction, since the grid's own pitch already
    clears an unjittered barrel's width -- rather than adjusting components
    it cannot recover (the previous version left ``y`` untouched, so a
    pair jittered close in the row direction alone could never repair).
    """
    # The end syringes sit at the row's own extremes, so their jitter can only
    # push them further out; shrinking the nominal half-span (rather than
    # capping jitter at the ends specially) keeps every syringe the same
    # jitter distribution while guaranteeing an end syringe's worst-case draw
    # still lands at or inside CAV_Y_M.
    usable_half = 0.999 * CAV_Y_M / (1.0 + 2.0 * Y_JITTER_FRAC / (n - 1))
    pitch = 2.0 * usable_half / (n - 1)
    nominal_y = -usable_half + pitch * np.arange(n)
    jitter = [
        (
            float(rng.uniform(-X_JITTER_M, X_JITTER_M)),
            float(rng.uniform(-Y_JITTER_FRAC * pitch, Y_JITTER_FRAC * pitch)),
            float(np.deg2rad(rng.uniform(-YAW_JITTER_DEG, YAW_JITTER_DEG))),
        )
        for _ in range(n)
    ]

    def placement(scale: list[float]) -> list[tuple[float, float, float]]:
        return [
            (jx * s, float(ny) + jy * s, jyaw * s)
            for (jx, jy, jyaw), ny, s in zip(jitter, nominal_y, scale)
        ]

    scale = [1.0] * n
    placed = placement(scale)
    for _round in range(MAX_REPAIR_ROUNDS):
        bad = {
            i
            for i in range(n)
            for j in range(n)
            if i != j and _obb_overlap(placed[i], placed[j], -ALLOWED_PENETRATION_M)
        }
        if not bad:
            break
        for i in bad:
            scale[i] *= 0.5
        placed = placement(scale)
    else:
        # MAX_REPAIR_ROUNDS exhausted: snap whatever is still colliding all
        # the way to the nominal grid, which is collision-free by construction.
        for i in bad:
            scale[i] = 0.0
        placed = placement(scale)
    return placed


def _clutter_layout(reference: dict, index: int, count: int, rng: np.random.Generator) -> dict:
    tray_pose = reference["fixtures"]["syringe_tray"]["pose"]
    tray_quat = reference["fixtures"]["syringe_tray"]["quat"]
    quat0 = reference["objects"]["syringe_pack_0"]["quat"]  # the shared "parallel" heading
    s0 = np.array(reference["objects"]["syringe_0"]["pose"])
    s1 = np.array(reference["objects"]["syringe_pack_0"]["pose"])
    z = float(s0[2])
    u_row = (s1[:2] - s0[:2]) / np.linalg.norm(s1[:2] - s0[:2])  # tray's long axis
    u_cross = np.array([-u_row[1], u_row[0]])  # tray's short axis, i.e. barrel heading
    tray_xy = np.array(tray_pose[:2])

    n = int(rng.integers(MIN_SYRINGES, MAX_SYRINGES + 1))
    placed = _pack_row(n, rng)

    objects = {}
    for k, (lx, ly, dyaw) in enumerate(placed):
        name = "syringe_0" if k == 0 else f"syringe_pack_{k - 1}"
        xy = tray_xy + lx * u_cross + ly * u_row
        objects[name] = {"pose": [round(float(xy[0]), 6), round(float(xy[1]), 6), z], "quat": _quat_yaw(quat0, dyaw)}

    box_shift = rng.uniform(-0.03, 0.03, size=2)
    return {
        "variant": f"{VARIANT_PREFIX}_t{index:02d}",
        "settled_from": (
            f"tools/make_tray_clutter_trials.py: {n} syringes packed into {REFERENCE_LAYOUT}'s own tray pose "
            "(itself the export's settled syringe_0 pose, lifted onto the tray floor)."
        ),
        "objects": objects,
        "fixtures": {"syringe_tray": {"pose": list(tray_pose), "quat": list(tray_quat)}},
        "box_shift": [round(float(box_shift[0]), 6), round(float(box_shift[1]), 6)],
        "trial": {
            "index": index,
            "of": count,
            "n_syringes": n,
            "box_shift_m": [round(float(box_shift[0]), 6), round(float(box_shift[1]), 6)],
        },
    }


def objects_clause(n: int) -> str:
    names = ["syringe_0"] + [f"syringe_pack_{k}" for k in range(n - 1)]
    return " ".join(names) + " - syringe"


OBJ_LINE_RE = re.compile(r"^(\s*)syringe_0 syringe_pack_0 - syringe\s*$", re.M)


def write_trials(count: int, seed: int) -> list[str]:
    base_doc = (TASK / f"{BASE_STEM}.task").read_text()
    reference = json.loads((TASK / "start_poses" / f"{REFERENCE_LAYOUT}.json").read_text())
    rng = np.random.default_rng(seed)

    names = []
    for index in range(count):
        stem = TRIAL_STEM.format(index=index)
        variant = f"{VARIANT_PREFIX}_t{index:02d}"
        layout = _clutter_layout(reference, index, count, rng)
        n = layout["trial"]["n_syringes"]
        bx, by = layout["trial"]["box_shift_m"]

        header = (
            f"; TRIAL {index:02d} of {count} in the {VARIANT_PREFIX} layout sweep: {n} syringes\n"
            f"; packed into the tray (see tools/make_tray_clutter_trials.py for the placement\n"
            f"; rule), the sharps box shifted by ({bx:+.4f}, {by:+.4f}) m. Drawn from numpy\n"
            f"; default_rng(seed={seed}), one draw per trial in sequence -- box shift\n"
            f"; uniform(+-0.03) m.\n"
            f"; Written by tools/make_tray_clutter_trials.py from {BASE_STEM}.task; the\n"
            f"; layout is start_poses/{variant}.json. Everything else is the base document's.\n"
        )
        doc = base_doc.replace(
            f"(define (problem {BASE_STEM.upper()})", f"(define (problem {BASE_STEM.upper()}_T{index:02d})", 1
        )
        doc = re.sub(r"^; -{20,}\n", header, doc, count=1, flags=re.M)
        doc, n_sub = OBJ_LINE_RE.subn(lambda m, n=n: f"{m.group(1)}{objects_clause(n)}", doc, count=1)
        assert n_sub == 1, f"{stem}.task: (:objects ...) clause not found"
        (TASK / f"{stem}.task").write_text(doc)

        pose_path = TASK / "start_poses" / f"{variant}.json"
        pose_path.write_text(json.dumps(layout, indent=2) + "\n")
        names.append(variant)
    return names


def clean() -> int:
    removed = 0
    for path in list(TASK.glob(f"{BASE_STEM}_t*.task")) + list((TASK / "start_poses").glob(f"{VARIANT_PREFIX}_t*.json")):
        path.unlink()
        removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean:
        print(f"removed {clean()} files")
        return 0
    names = write_trials(args.count, args.seed)
    print(f"wrote {len(names)} trial documents: {names[0]} .. {names[-1]}")
    for name in names:
        trial = json.loads((TASK / "start_poses" / f"{name}.json").read_text())["trial"]
        print(f"  {name}: {trial['n_syringes']} syringes, box shift ({trial['box_shift_m'][0]:+.4f}, {trial['box_shift_m'][1]:+.4f}) m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
