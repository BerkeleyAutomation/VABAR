# Syringe packing: perception GAP graph

This graph follows the tool-hanging pipeline with existing open skills:

1. `perceiving-objects` locates the syringe and its physical insertion tip from
   the overhead RGB-D camera. The red cap resolves which endpoint is directed.
2. `perceiving-object-parts` separately detects the box and aperture, fits the
   observed lid plane, and intersects the aperture-mask camera ray with that
   plane. The target is therefore the perceived hole, not the box OBB center.
3. `grasping-direct-ik` registers the known syringe mesh station to the cloud
   and grasps the 45--60 mm plunger-flange band across its short axis.
4. `perceiving-object-parts` checks every calibrated Franka wrist view, chooses
   the strongest observation, and expresses the physical tip in TCP coordinates.
5. `planning-held-object-motion` chooses a clearance-first sequence and the
   smallest reachable wrist rotation: lift, rotate in clear space, then carry
   above the aperture with orientation fixed. `executing-held-object-motion`
   executes that plan with attached-object collision geometry.
6. `planning-held-object-motion` plans the local approach and linear crossing;
   `executing-feature-mating` reobserves the held syringe, updates its TCP-to-tip
   offset, and executes an orientation-locked linear insertion.
7. A camera-only terminal check rejects a syringe still visible above or away
   from the aperture; disappearance inside the opaque box is accepted only
   after insertion and release complete.

The graph uses no `sim.*` ground-truth tools. Its reusable held-object motion
planner lives in the open skill library rather than in a syringe-only package.

## Run

```bash
uv run python -m vabar run \
  --task sharps_disposal/syringe --solve graph --graph gap_perception \
  --renderer tiled --motion curobo --episodes 1 --max-steps 8500 --record \
  --out outputs/sharps_gap_curobo
```

The earlier recorded episode succeeded physically but its insertion node ended
with an error after the evaluator had already terminated. This refactor adds an
explicit graph-side verification stage. A fresh end-to-end cuRobo recording is
still required after GPU access is restored.
