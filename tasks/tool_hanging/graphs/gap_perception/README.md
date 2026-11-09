# Zero-shot-style GAP: wrench hanging

This graph is the minimal skill composition GAP can plausibly generate from
the installed Open Robot Skills library:

```text
RGB-D wrench/ring perception -> RGB-D hook/axis perception -> grasp
-> in-hand feature registration -> compute loop-over-shaft mate pose
-> generic support disengagement -> one collision-aware attached-object plan
to the pre-insertion pose -> re-register -> relation-derived engagement
-> release
```

The post-grasp graph contains no wrench-specific waypoint coordinates.
`computing-feature-mating-poses` resolves loop-over-shaft geometry and free symmetry;
the same `planning-held-object-motion` skill used by syringe packing converts
that typed approach pose into one backend-owned attached-object trajectory.
The generic `motion.plan_support_disengagement` tool plans how to clear initial contact first;
`executing-held-object-motion` and
`executing-feature-mating` executes constrained hook/support engagement. For the
screwdriver box, the graph instead selects `transporting-objects`: it keeps the
upright reorientation, descends until the screwdriver tip reaches the perceived
opening, releases, and lets gravity place it inside the container.

## Current result

The disengagement tool derives its distance from attachment thickness and a
general clearance margin. CuRobo or PyRoKi then jointly chooses reorientation
and transit to the geometry-derived hook approach; the graph supplies no lift
pose, lift distance, or intermediate carry pose.

## Validate

```bash
GAP_SKILLS_PATH=robot_skills:../open-robot-skills \
uv run gap run tasks/tool_hanging/graphs/gap_perception --validate-only
```

## Run

```bash
uv run python -m vabar run \
  --task tool_hanging/wrenches_level1 \
  --solve graph --graph gap_perception \
  --renderer tiled --motion curobo --episodes 1 --max-steps 8500
```
