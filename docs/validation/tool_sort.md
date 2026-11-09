# `tool_sort` on upstream RoboSimStudio

Measured 2026-10-28 on one RTX PRO 6000 (`CUDA_VISIBLE_DEVICES=1`), with
RoboSimStudio pinned at `d9b2e44` and installed unmodified, graph-as-policy at
`09e45f8`, and open-robot-skills at `main@752dfbd`.

## What changed, and why

The port only fixes imports. Task logic, documents, numbers and assets are as
delivered.

| File | Change | Why |
| --- | --- | --- |
| `tasks/tool_sort/staging.py` | `import vabar` | Installs the scripted-routine module that upstream leaves out, so the documents' `(:routine ...)` lower. The same line is in every suite. |
| `tasks/tool_sort/tools.py` | `robosimstudio_gap.sim_tools` → `robosimstudio_policy_agent.sim_tools` | Upstream renamed the package. `register_goal_cursor` is otherwise unchanged. |
| `tasks/tool_sort/routine.py` | Adds `import newton`, `dataclasses.replace`, and `grasp_quat` / `osc_action` from `robosimstudio.solvers` | These names were used but never imported. The old compatibility runtime made them resolve anyway; upstream does not. Every one exists upstream with the same signature. Without the fix, `RoutineSortFaces` (the Level-1 routine) raised `NameError`. |
| `tasks/tool_sort/graphs/gt/README.md` | Two path references updated | The old ones named `robosimstudio_gap.*`. |
| `tasks/_shared/scripts/*.py` (new, shared) | The six GaP node scripts from the old runtime's `nodes/`, copied verbatim | `clamp_grasp_depth`, `compute_align_pose`, `container_obb`, `give_up`, `next_goal`, `target_obb`. Upstream's `robosimstudio_policy_agent.graphs` resolves a graph's missing scripts from `<task root>/_shared/scripts/`. The `tool_sort`, `tool_hanging` and `syringe_packing` graphs use them. |

No `vabar/ext/tool_sort.py` was needed. The native pairings are `panda` and
`panda_robotiq`, both upstream. `ur5` and `linear_4310_attached` are only
undeclared alternative hosts, and nothing builds them.

## Probe

`python -m vabar probe --suite tool_sort --envs 1`: **ok, 41 variants**. Every
document lowers, and `tool_sort/sort_all` compiles and resets (4.2 s).

`Env` compile, `reset(0)` and 5 idle steps also pass on `hammer`, `syringe`,
`sort_all` and `level1_1`.

## Scripted groundtruth vs. the recorded parity table

Each row is one episode: `python -m vabar run --task tool_sort/<v> --solve routine`.

| Variant | Recorded (old runtime) | Upstream RoboSimStudio |
| --- | --- | --- |
| `hammer` | success, 1.000, 786 | **success, 1.00, 786** |
| `syringe` | success, 1.000, 1,958 | **success, 1.00, 1,958** |
| `sort_all` | fail, 0.250, 4,801 | **fail, 0.25, 4,801** |
| `sort_all_neat` | fail, 0.250, 4,801 | **fail, 0.25, 4,801** |
| `level1_1` | (not in the table) | success, 1.00, 2,183 |
| `level1_2` | (not in the table) | success, 1.00, 2,053 |

Every recorded row agrees to the step.

## Not validated

- **`gt` graph** (`--solve graph --graph gt`). The workflow names
  open-robot-skills bundles (`grasping-direct-ik`, …), so open-robot-skills must
  be on the registry path. gap's bundle boot also scans the graph scripts' text
  and starts the out-of-process `curobo` tool bundle. `waypoint_move.py` names
  `curobo.plan_to_pose`, though only in its docstring.
  - That bundle needs its own environment, built with
    `cd open-robot-skills/tools/curobo && CUDA_HOME=/usr/local/cuda uv sync`.
    It compiles cuRobo against the bundle's torch.
  - That build was not run here because it needs permission. Without it, the
    episode ends at step 74 with
    `tool bundle 'curobo': failed to start its server`.
  - The bundle also resolves `graph-as-policy-core` from
    `../../../graph-as-policy/gap-core`, so a `graph-as-policy` checkout at
    `09e45f8` has to sit beside `open-robot-skills/` (present here).
- **`gap_perception`, `gap_perception_loop` and `gap_perception_syringe`.** On
  top of the curobo bundle, `perceiving-sorting-pairs` calls a hosted VLM and the
  perception bundles use SAM 3 / GroundingDINO weights. Not attempted.
- `sort_new` and the five long industrial piles were not re-run.

## Requested shared changes

- **Environment / README.** Document the workspace layout the graphs need:
  `graph-as-policy/` (at the pinned rev) and `open-robot-skills/` beside the
  checkout. Also document the one-time build of the curobo tool bundle
  (`CUDA_HOME` set, `uv sync` in `open-robot-skills/tools/curobo`). Graph
  validation for every suite depends on it.
- The bundle shadow warnings (`executing-held-object-motion`,
  `perceiving-deformable-linear-objects`, `perceiving-routing-fixtures` exist in
  both `robot_skills/` and open-robot-skills) are expected: `robot_skills/` comes
  first on purpose.

## Follow-up (main session, after the port)

- **`ur5` and `linear_4310_attached` are registered** (`vabar/ext/tool_sort.py`,
  imported by `staging.py`). Runs never build them. But the Studio's Task panel
  resolves every pairing a suite declares, and the two `(:host ...)` rows naming
  them made `StudioSession("tool_sort/...")` raise `KeyError: unknown embodiment
  'ur5'`. The definitions are transcribed from the RoboSimStudio revision the
  suite was written against, over the assets in `assets/embodiments/`.
- **`gt` graph:** once the cuRobo tool bundle was built, `tool_sort/hammer`
  succeeds with reward 1.00 in 604 steps.
- **`gap_perception` and `gap_perception_syringe`** failed graph-as-policy's SK2
  check (subgraphs running graph-local scripts under a bundle's name). Those
  subgraphs are relabelled `generic`, with the original names in `meta.composes`;
  both now validate clean. `gap_perception_loop`, the Level-1 reference graph,
  was already clean.
