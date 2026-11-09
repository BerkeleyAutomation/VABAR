# tool_hanging on upstream RoboSimStudio

Validated against `robosimstudio/robosimstudio@d9b2e44` (unmodified), graph-as-policy
`KeplerC@09e45f8`, open-robot-skills `main@752dfbd`, one RTX PRO 6000. Every run is a
single episode unless stated, through `python -m vabar run` (upstream `bench_run.py`).

## What changed, and why

| File | Change | Why |
| --- | --- | --- |
| `tasks/tool_hanging/records.py` (new) | `from .assets import *` | Upstream's directory loader executes a suite's `staging.py`, `scenes.py` and `records.py`; the compatibility runtime also executed `assets/`. Without it the asset records were never registered and every goal failed to bind (`encloses ... has no implementation for operand kinds ['', '']`). |
| `vabar/ext/tool_hanging.py` (new) | registers the `yam_tool_hanging` embodiment | The suite's robot is not in upstream. Transcribed from the compatibility runtime field for field (VABAR's own `assets/embodiments/yam/yam_tool_hanging.xml`, the cuRobo model, no `urdf`, so IK is Newton's own solver as when the documents were measured). |
| `tasks/tool_hanging/staging.py` | `import vabar.ext.tool_hanging` | Registers the robot when the suite is scanned, in any process. |
| `tasks/tool_hanging/layouts.py` | `sample_level{1,2}_layout(*, task, episode_id, rng)` → `(task, episode_id, rng)` | Upstream calls an episode sampler positionally; the keyword-only signature raised `TypeError` on every Level-1/2 reset. Body unchanged. |

No goal, number, mesh or graph was edited.

## Probe

`python -m vabar probe --suite tool_hanging --envs 13 --steps 5`: the suite loads, **13/13 variants
lower, compile, reset and step**.

## Scripted groundtruth (`--solve routine`)

Against the parity table in the source repository's README ("re-run after the merge" column).

| Task | Here | Recorded | |
| --- | --- | --- | --- |
| `wrenches_level1` | success, 1.00, 1,165 | success, 1.000, 1,165 | identical |
| `pliers_level1` | success, 1.00, 1,090 | success, 1.000, 1,090 | identical |
| `scissors_level1` | success, 1.00, 915 | success, 1.000, 915 | identical |
| `screwdrivers_level1` | fail, 0.67, 1,095 | fail, 0.667, 1,095 | identical |
| `pliers` | fail, 0.00, 1 | fail, 0.000, 1 | identical |
| `wrenches` | **success**, 1.00, 2,330 | fail, 1.000, 2,330 | same reward and steps; upstream's evaluator calls it a success |

`wrenches` once raised `left-arm IK failed at correct_pre_hook: position 1907 mm` instead, in a
run sharing the GPU with four other simulations; alone it succeeded twice (once through the runner,
once driven directly). Recorded as nondeterminism under contention, not reproduced.

## Reference graph (`--solve graph --graph gt`, privileged)

| Task | Result | Note |
| --- | --- | --- |
| `scissors_level1` | success, 1.00, 2,116 | identical with `--motion curobo` (the graph plans through the suite's `sim.plan_*` tools, so `--motion` does not apply) |
| `pliers_level1` | success, 1.00, 2,053 | |
| `wrenches` | success, 1.00, 2,400 (budget) | finishes on the last step of its budget |
| `scissors` | fail, 0.50, 2,400 (budget) ×4 | **step budget**: with `--max-steps 6000`, success 2/2 at 4,133 and 4,116 steps. Upstream's connector runs each `robot.execute_trajectory` to tolerance (≤ 15 steps per waypoint), costing more steps per leg than the graph README's budget assumes. The document's `(:max-steps 2400)` is unchanged. |
| `screwdrivers_level1` | fail, 0.33, 346 | the job cursor (`tools.py:jobs`) skips the goal's `containment screwdriver_1.tip ...` clause because it compares the feature-qualified operand to object names; the graph verifies with no work. Same code path in the source runtime (`about` is `str(ref)` in both), so this is content, not the port. |
| `wrenches_level1` | fail, 0.33, 80 | `wrench_adjustable_260` (the Level-1 enlarged wrench) has no entry in `tools.py:TOOL_STATIONS`. The source repository ran Level 1/2 through `gap_perception` by default; `gt` was never given this tool class. |

Run with `--bus-timeout 180`, as the source repository's commands do: the default 30 s watchdog
can expire on the first `sim.plan_pick` when several simulations share a machine (observed once,
and the graph then gives that tool up).

## Perception graph (`gap_perception`)

**Since resolved:** on the current pins the graph runs, and is the cells' reference graph (see
the 2026-11-09 section below for full-cell results). What follows is the 2026-10-28 record.

**Not runnable under the pinned graph-as-policy (`KeplerC@09e45f8`).** The documented Level-1
command

    python -m vabar run --task tool_hanging/wrenches_level1 --solve graph --graph gap_perception \
      --motion curobo --no-privileged --perception vision --max-steps 8500 --bus-timeout 180

is refused at graph validation before any node runs (fail, 0.33, 228 settle steps):

    Graph validation failed (6 issues): [error] subgraphs.target_sg: subgraph names skill
    'perceiving-functional-features' but no script node runs one of its canonical scripts (...) (SK2)

Rule SK2 requires a subgraph that names a skill bundle to run one of that bundle's canonical
scripts by file name. This graph names bundles (`perceiving-functional-features`,
`planning-held-object-motion`, `executing-held-object-motion`, ...) but runs its own graph-local
scripts (`perceive_target.py`, `perceive_fixture.py`, `execute_reorientation.py`,
`execute_approach_only.py`, and `disengage_sg` runs none). The six errors are in `target_sg`,
`fixture_sg`, `disengage_sg`, `carry_sg`, `preapproach_sg` and one more.

This is the known runtime-revision conflict, not the port. The source repository pinned
graph-as-policy `lgulich@febee27` for these graphs and documented that later revisions (`3a1bda6`
onward) fail them on exactly this canonical-script contract. Bridging it means either rewiring the
graph (canonical script names, SK3 signatures) or running this suite on the older runtime. Both
change something outside a conservative port, so neither was done. SAM 3, GroundingDINO, an HF
token and Vertex VLM credentials were all present, so the models are not the blocker.
`gap_perception_level2` was not attempted (same runtime, same graph family).

## Release, termination and the landmark beads (2026-11-09)

Three problems, reported against `scissors_level1` but true of the whole suite:

- **No goal required the tool to be let go.** `contact-force` against the holder was meant to be the
  release test ("a ring held around the arm by the gripper reads ~0 N"). It is not: lowered onto its
  hook while still in the jaws, a tool reads well above the 0.2 N floor. Driving each routine and
  reading the goal every step, `wrenches_level1`, `pliers_level1` and `scissors_level1` scored success
  on 57, 21 and 6 steps with the gripper still shut on the tool, the first of them with the nearest
  hand 9.7, 8.2 and 2.3 cm from the tool's origin.
- **Success did not end the episode** (`(:terminate-on-success false)` in the suite), because of the
  first problem: ending on a goal a held tool satisfies would have scored hangs the policy went on
  to drop. A policy that never asks to stop therefore ran to its full step budget (8,500 at Level 1,
  22,000 at Level 2) and was scored wherever it left things, a tool still in hand included.
- **The landmark beads rendered in every camera**: red ring and bow centres, green bead circles on
  the wrenches and scissors, red tip dots on the screwdrivers. They do not collide, but a perception
  graph or a learned policy sees them.

What changed:

- Every tool's clause now ends in `(> (hands-clear TOOL) R)` and is held for a second,
  `(hold-for 1.0 ...)` (`wrenches.task` already held its whole goal for two). `hands-clear` is the
  distance from the tool's origin to the nearest TCP of either arm. R is the tool's radius about its
  origin, measured off its collision hulls, plus 15 mm, rounded up to the centimetre: a hand nearer
  than that could still be holding the tool, and one further away cannot.

  | Tool record | Radius | R |
  | --- | ---: | ---: |
  | `scissor_pair_1` / `scissor_pair_2` | 79.6 / 84.1 mm | 0.10 m |
  | `plier_combination` / `plier_cutter` | 87.8 / 111.4 mm | 0.11 / 0.13 m |
  | `screwdriver_slim` / `screwdriver_grip` | 175.5 / 100.9 mm | 0.20 / 0.12 m |
  | `wrench_adjustable_150` / `_200` / `_260` | 123.7 / 148.8 / 193.4 mm | 0.14 / 0.17 / 0.21 m |
  | `hanging_hammer_1` / `_2` | 334.9 mm (origin at a corner of the mesh) | 0.35 m |

- With the release in every goal the suite no longer states `(:terminate-on-success false)`, so
  success ends the episode as it does elsewhere.
- The 200 frozen copies under `benchmark/tool_hanging/` restate their goals, and carry the same clause.
- The beads are off unless `VABAR_LANDMARK_MARKERS=1`, which restores exactly what was drawn before.

Scripted routines, episode 0, before (`main@c421a21`) and after:

| Task | Before | After | Success steps with the jaws shut |
| --- | --- | --- | --- |
| `wrenches_level1` | success, 1,165 (policy stop) | success, 1,042 (success) | 57 → 0 |
| `pliers_level1` | success, 1,090 (policy stop) | success, 924 (success) | 21 → 0 |
| `scissors_level1` | success, 915 (policy stop) | success, 773 (success) | 6 → 0 |
| `screwdrivers_level1` | fail, 1,095 | fail, 1,095 | 0 → 0 |
| `wrenches` | success, 2,330 (policy stop) | success, 2,270 (success) | 0 → 0 |
| `screwdrivers` | success, 2,190 (policy stop) | success, 2,114 (success) | 0 → 0 |
| `scissors` | fail, 1,830 | fail, 1,830 | 0 → 0 |
| `pliers` | fail, 1 | fail, 1 | 0 → 0 |

Every outcome is unchanged. After the change each first success comes with the jaws open and the
nearest hand 13.8–30.9 cm from the tool's origin. The hammers and the Level-2 documents ship no routine;
`hammers` ships no graph either, so its 0.35 m is unexercised by any shipped solution.

The benchmark cells with their reference graph (`gap_perception`, cell flags, `--config fast --limit 30`,
all 30 instances), before and after, on the same machine at the same time:

| Cell | Before: solved, per-tool | After: solved, per-tool | How episodes ended, after |
| --- | --- | --- | --- |
| tool hanging L1 | 28/30, 95.6% | 30/30, 100.0% | success 30 |
| tool hanging L2 | 23/30, 86.7% | 26/30, 91.7% | success 26, graph stopped 4 |

Both are inside the spread [`docs/benchmark.md`](../benchmark.md) records for repeated runs (29-30/30 and 21-26/30). In each of
the four Level-2 failures after the change every hand was 15-42 cm clear of every tool; the tool that
failed was not on its holder (no second of hold), so the release clause decided none of them.

## Requested shared changes

- **Decision needed: tool_hanging's perception graphs vs the single graph-as-policy pin** (see
  above). Either adapt `graphs/gap_perception*/workflow.json` to SK2, or accept that these reference
  graphs do not run on the release stack.

- **`assets/embodiments/yam/yam_bimanual_arms.xml` shadows upstream's file of the same name.** The
  collection's `assets/` is searched before the packaged one, and the two files differ, so every
  task on upstream's `yam_bimanual` (not only VABAR's) builds VABAR's copy while this root is
  registered. tool_hanging does not use it. Consider removing VABAR's copy if no VABAR suite needs
  its differences, or renaming it.
- The gt graph README's "Measured" section and its claim of Level-1 coverage are stale (see the
  two Level-1 rows above); worth a docs pass.
