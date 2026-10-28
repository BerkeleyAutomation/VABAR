# `cable_route/gt` — the ground-truth graph

This workflow serves `cable_route/wrap` using privileged simulator state.
The Easy and Medium routing tasks use the perception workflow in
[`graphs/weave`](../weave).

The graph plans a route against the live cable, closes the gripper and engages
the simulator's cable latch, checks the hold, executes the carry span, releases,
and reads the winding state. Stages are addressed by boundary names such as
`stop_before: "close_hold"` and `start_after: "close_hold"`.

| Stage | Privileged tools |
| --- | --- |
| Plan | `sim.plan_route` |
| Grip and release | `sim.set_gripper`, `sim.hold_cable`, `sim.release_cable` |
| Check hold and winding | `sim.get_cable_state` |

The plan gate accounts for both position and rotation. Plans seed from the home
configuration, and the carry span streams as a merged motion. The simulator
latch is specific to this diagnostic graph; the delivered Easy and Medium
vision policy uses contact-only grasping.

## Run an individual episode

From the repository root:

```bash
uv run python -m vabar run --task cable_route/wrap --solve graph --graph gt --episodes 0:1
```

This graph requires its privileged tools. The Easy/Medium vision protocol is in
[`docs/running_graphs.md`](../../../../docs/running_graphs.md).
