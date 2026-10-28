# gear_mesh on upstream RoboSimStudio

Measured on 2026-10-28, on one RTX PRO 6000 Blackwell, against RoboSimStudio `d9b2e44`, which pins newton
1.7.0 and mujoco 3.15.0. The release environment installs upstream without modification. The baseline is the
pre-release collaboration checkout (newton 1.5.0.dev0, mujoco 3.11), run on the same machine.

## What changed, and why

| File | Change | Why |
| --- | --- | --- |
| `records.py` (new) | Imports `assets.records`. | Upstream's generic loader runs a suite's `staging.py`, `scenes.py` and `records.py`. It no longer runs the `assets/` package. Without this file the board's asset record was never registered, and every document failed to lower with `BindError: gears-seated(board) has no implementation`. |
| `assets/gear.py` | `EntityGear.after_build` saves the friction of every shape the gear built, then restores it after `EntityRigid.after_build`. | Upstream's `EntityRigid.find_in_scene` now writes `spec.friction` onto every shape of the body. That erased the bore's 0.15 friction and replaced it with the teeth's 1.4. The hub then braked on its peg, and the gate slipped to a ratio of 0.75. This was the only property that differed: every other shape material, body mass and joint parameter in the compiled model is identical between the two stacks. |
| `staging.py` | `import vabar` | Installs the scripted-routine module (`vabar/__init__.py`). |
| `gear_mesh.suite`, 200 drawn `*.task` | Comments and `(:note ...)` text that pointed to `docs/gear_mesh_levels.md` now point to `tasks/gear_mesh/difficulty.py` or say "round-1/2 evaluation numbers". | That doc was never in this tree. `difficulty.py` is the in-tree statement of what makes a level. These are comments and notes only; no clause or number changed, and `assets/variants.json` does not hash document text. |

## Results

| Check | Result |
| --- | --- |
| `python -m vabar probe --suite gear_mesh --envs 1 --steps 20` | All **204 variants load and lower**. `gear_mesh/pair` compiles, resets and steps. |
| Compile + reset + 20 idle steps | OK on `gearmesh-easy-single-01`, `gearmesh-easy-pair-01`, `gearmesh-medium-train-001`, `gearmesh-medium-train-100`, `pair`, `train`, `quad`, `quad_seeded`. |
| Gate: `rigs/gear_bench.py spin` | See below. |
| Gate: `rigs/gear_bench.py broken` (negative control) | With gear_b dropped, gear_c reads 0.000. With gear_c dropped, gears a/b read 0.970. This is the expected population. |
| `--solve routine` | `LookupError: gear_mesh routine: cannot find two finger bodies to measure the jaw axis`. This is the known state, identical upstream and in the collaboration checkout: the suite stands the Menagerie 2F-85, whose pad bodies the routine cannot find. Not fixed. |
| Graph `gap_perception`, `gearmesh-easy-single-01`, `--no-privileged --perception vision --motion curobo --renderer rtx` | **SUCCESS, reward 1.00, 2,511 steps.** 8.4 min wall time. Recorded upstream: success, 1.000, 4,392 steps. Collaboration checkout: success, 1.000, 4,363 steps. The episode ends on success while `press_sg.push` is still holding the button, so gap logs `the simulation bus closed: the episode ended`. That is expected. |

**Gate: `rigs/gear_bench.py spin`**

| Stack | Pinion (rad/s) | Ratio a / b / c | Verdict |
| --- | --- | --- | --- |
| Before the fix | 3.85 | 0.750 / 0.750 / 0.750 | SLIP, the train does NOT drive |
| After the fix | 3.87 | **0.967 / 0.967 / 0.967** | **the train drives** |
| Collaboration checkout, same machine | 3.87 | 0.968 / 0.968 / 0.968 | the train drives |
| Recorded upstream (`aa42d099`) | 3.87 | 0.968 / 0.968 / 0.969 | the train drives |

## How to run

```bash
python -m vabar probe --suite gear_mesh
python -m vabar run --task gear_mesh/gearmesh-easy-single-01 --solve graph --graph gap_perception \
    --no-privileged --perception vision --motion curobo --renderer rtx
```

The graph reads the board through the `overhead` camera, so **`--renderer rtx` (or `tiled`) is required**.
Upstream's `bench_run` renders nothing by default. Without a renderer, the first node fails with
`no camera 'overhead' in the observation`. SAM 3 is the `sam3` tool bundle from open-robot-skills and needs
the gated `facebook/sam3` weights (an HF token).

The rigs reach the suite by name through `task_module`, so they need the task root registered:

```bash
python -c "import vabar, runpy, sys; vabar.install(); sys.argv=['gear_bench','spin']; \
  runpy.run_path('tasks/gear_mesh/rigs/gear_bench.py', run_name='__main__')"
```

## Not validated here

- The other 203 variants under the graph. Only one episode was run; it is one sample, and the graph's thread timing
  is not pinned.
- The `panda` host row (the Franka hand). It compiles, but per the suite's own note it is unmeasured at the hand's
  own gain.
- Comments in the suite still cite `tools/build_gear_scenes.py`, `tests/test_gear_mesh.py` and
  `docs/task_layout.md`. These are RoboSimStudio-side paths that aren't in this tree, and are left as written.
