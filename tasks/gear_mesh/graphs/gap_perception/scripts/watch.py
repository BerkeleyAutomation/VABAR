"""Stand still while the train runs; the camera says whether the pinion is turning."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sight  # noqa: E402


def _pinion_patch(ctx):
    rgb, depth, K, T = sight.frame(ctx)
    view = sight.View(rgb, depth, K, T)
    band = np.isfinite(view.z) & (view.z > view.bench_z + 0.012) & (view.z < view.bench_z + 0.028)
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    orange = band & (r > 170) & (g > 100) & (b < 110) & (r - b > 80) & (g - b > 40)
    if orange.sum() < 50:
        return None
    ys, xs = np.nonzero(orange)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return rgb[y0:y1, x0:x1].astype(float)


def run(ctx, rounds: int = 8, steps: int = 30):
    before = _pinion_patch(ctx)
    for _ in range(int(rounds)):
        ctx.tool("robot.wait_steps", steps=int(steps))
        after = _pinion_patch(ctx)
        if before is not None and after is not None and before.shape == after.shape:
            moved = float(np.abs(after - before).mean())
            if moved > 3.0:
                return {"route": "running", "pixel_change": round(moved, 2)}
        before = after
    return {"route": "stalled", "pixel_change": 0.0}
