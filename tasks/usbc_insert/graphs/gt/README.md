# `usbc_insert/gt` — the ground-truth graph

This workflow mates the USB-C plug into the panel-mount bulkhead using
privileged simulator state and both arms. The delivered Easy and Medium
vision policy is in [`graphs/mate`](../mate).

The free receptacle rests in a cradle without an axial stop. The support arm
braces the receptacle before the working arm grasps and inserts the plug.
Both mating halves use hydroelastic contact through pressure fields.

| Stage | Privileged tools and purpose |
| --- | --- |
| Brace | `sim.plan_brace` positions the support arm on the barrel's smooth band. |
| Plan | `sim.plan_insert` solves the insertion schedule against part poses. |
| Grip and release | `sim.set_gripper` applies calibrated, ramped partial widths. |
| Check grasp | `sim.get_insertion_state` supplies the grasp signal. |
| Correct | `sim.get_insertion_state` and `sim.plan_insert` update the stroke for the held plug pose. |
| Check seat | `sim.get_insertion_state` selects the seated, retry, or lost route. |

The correction accounts for lateral and axial displacement after grasping.
A held but short insertion can take one statically unrolled retry with a fresh
correction and a back-off before another push. The graph's exit is distinct
from the task evaluator's physical goal after release and settling.

## Run an individual episode

From the repository root:

```bash
uv run python -m vabar run --task usbc_insert/bulkhead --solve graph --graph gt --episodes 0:1
```

This graph requires its privileged tools. The Easy/Medium vision protocol is in
[`docs/running_graphs.md`](../../../../docs/running_graphs.md).
