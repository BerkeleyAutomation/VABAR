# Cable suites on upstream RoboSimStudio: `usbc_insert` and `cable_route`

Both suites ran before on a vendored fork of RoboSimStudio (`cable_runtime/`,
KeplerC/RoboSimStudio@eeb2aa88, Newton 1.5.0.dev0 @ `10402ec`) launched by a separate
`run_cable.py`. They now load as two directories of the VABAR task root on the
**unmodified** pinned upstream (robosimstudio/robosimstudio@`d9b2e44`, Newton 1.7.0,
mujoco 3.15). Nothing under `.venv` was edited.

## What changed, and why

| File | Change | Why |
| --- | --- | --- |
| `tasks/*/staging.py` | `import vabar` (done for all suites) | installs the scripted-routine module upstream leaves out |
| `tasks/usbc_insert/records.py`, `tasks/cable_route/records.py` (new) | import every module under `assets/` (`pkgutil.walk_packages`) | the old loader walked a suite's `assets/` package itself; upstream's generic loader runs only `staging.py`/`scenes.py`/`records.py`. Without it `usb_c_bulkhead`'s `bore` cavity was never registered and `insertion(plug, bulkhead.bore)` failed to bind |
| `tasks/cable_route/assets/{layout,cable,rod,terminals}.py`, `tools.py` | `robosimstudio.domains.rod[.declare/.measures]` → `robosimstudio_materials.rod[...]` | package moved upstream; same names |
| `tasks/cable_route/routine.py` | `robosimstudio_bench.policies.base.policy_param` → `robosimstudio_bench.solutions.base.solution_param as policy_param` | policy → solution rename upstream; same function |
| `tasks/usbc_insert/staging.py` | `D405_WRIST_XFORM` stated in the suite (value unchanged) | upstream's `yam_linear_arena` hand is otherwise identical but no longer carries the constant |
| `tasks/usbc_insert/assets/entity.py` | a cable chain rooted on a free body is labelled with that body's articulation (`_host_articulation`) | Newton >= 1.6 rejects a joint whose parent body is in another articulation (`ModelBuilder._validate_joints`). The old Newton accepted it and the MuJoCo conversion made the chain a subtree of the plug/bulkhead; labelling the joints with the host's articulation gives the conversion that same tree. Each chain still gets its own `add_articulation` range (its own FK pass, as before). MuJoCo warns "joint order is not in depth-first topological order" on the two-chain scenes (insert_medium, bulkhead); the medium insertion below succeeds under it |
| `tasks/cable_route/tools.py` | `register_cable_tools` also registers `curve.fit_centerline` / `curve.track_centerline` | the old connector registered them for every task; upstream has no curve tools. Tagged `geometry`, so they survive `--no-privileged` |
| `vabar/cable/curve_tools.py`, `trackdlo_skeleton.py`, `LICENSE.trackdlo`, `__init__.py` (new) | vendored from KeplerC/RoboSimStudio@eeb2aa88 `robosimstudio_gap/` | only the import path and the docstring license path edited; TrackDLO stays MIT with its notice |

Not needed after all: `servo.py` (the cable graphs run `--motion osc`), the old
palette segmenter (upstream carries it byte-identical; see below), `autolab_robot`.

## Load and build (`python -m vabar probe` + compile/reset/5 idle steps per variant)

| Suite | Load | Variants | Env compile + reset |
| --- | --- | --- | --- |
| `usbc_insert` | ok | 4 | 4/4 ok (bulkhead, insert_easy, insert_medium, port) |
| `cable_route` | ok | 16 | 14/16 ok; `panel_l2`, `port3` raise `KeyError` in `assets/layout.py:terminated_spawn_y` |

`panel_l2` and `port3` are a **source defect, not a port one**: both are in
`WEAVE_TERMINALS` but missing from `TERMINATED_SPAN` (which names only `port_easy`,
`port_medium`); the code is byte-identical in the source tree. Neither is a
benchmark cell. Left as is.

## The frozen dataset (`benchmark/cable_tasks/dataset`, 400 files)

Each file is its base document (`usbc_insert/insert_{easy,medium}`,
`cable_route/port_{easy,medium}`) restated at `(:episodes (:count 1) (:seed S))`,
S = 10000..10099; the old `tools/fixed_scenes.py` checked that nothing else differs.
Upstream defines exactly this equivalence (`robosimstudio_bench.taskbuild.episodes.episode_seed`:
an instance at seed *k* is episode `k - base_seed` of its base document), and every
base document has seed 0, so **instance seed S is `--episodes S` on the base task** -- no
extra loader or task root. Checked for all 400: `make_episode(base, S)` reproduces the
manifest's recorded initialization exactly (400/400).

The old runner's protocol per case (from the manifest + `tools/run_scene_instance.py`):

| Cell | Base task | Graph | Settle steps |
| --- | --- | --- | --- |
| USB-C Easy | `usbc_insert/insert_easy` | `mate` | 600 |
| USB-C Medium | `usbc_insert/insert_medium` | `mate` | 600 |
| Routing Easy | `cable_route/port_easy` | `gt` (= `graphs/weave`) | 60 |
| Routing Medium | `cable_route/port_medium` | `gt` (= `graphs/weave`) | 60 |

all with `--perception vision --no-privileged --renderer rtx --bus-timeout 600` and
**`ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette`** (the old `RSS_SEGMENTATION_BACKEND`;
upstream's palette segmenter is byte-identical). The variable matters: with
open-robot-skills resolvable, `auto` silently switches to the gated SAM3 bundle. The
30 evaluation scenes per cell are `manifest.json`'s `cases[*].selected_tests`.

```bash
ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette python -m vabar run \
  --task usbc_insert/insert_easy --episodes 10009 \
  --solve graph --graph mate --perception vision --no-privileged \
  --settle-steps 600 --bus-timeout 600 --renderer rtx
```

## Runs (one RTX PRO 6000; single episodes unless stated)

| Run | This port | Old cable runtime, same scene |
| --- | --- | --- |
| `cable_route/weave3` routine, ep 0 | **success**, 1.00, 2,732 steps | recorded in the source README: failure, 0.67, 2,955 |
| `cable_route/port_easy` routine, ep 0 | failure, 0.25, 2,114 (seat0 only; "weave script complete") | -- |
| `usbc_insert/insert_easy` `gt` graph (privileged), ep 0 | **success**, 1.00, 3,711 (depth 12.5 mm, lat 0.2 mm) | -- |
| USB-C Easy `mate`, protocol, instance -10 (seed 10009) | failure, 0.00, 3,564 (depth -0.24 mm, lat 16.4 mm, mate gap 13.24 mm) | failure, 0.00, 3,659 (depth -0.26 mm, lat 17.2 mm, mate gap 13.26 mm) |
| USB-C Easy `mate`, protocol, 6 more selected (10024, 10026, 10031, 10032, 10043, 10084) | 0/6, each lat 14-17 mm at the mouth | not run |
| USB-C Medium `mate`, protocol, instance -86 (seed 10085) | **success**, 1.00, 4,393 (depth 13.0 mm, lat 0.5 mm) | not run |
| Routing Easy `gt`, protocol, instance -10 (seed 10009) | failure, 0.50, 4,271: both seats woven, terminal not detected (palette masks <= 2,124 px vs `BLOCK_MIN_PX = 3400`), cable end outside the port box | **success**, 1.00, 6,473: terminal likewise not detected (masks <= 2,549 px), seats woven, and the cable end came to rest inside the port box |
| Routing Easy `gt` with default `--perception oracle` | aborts at step 113: `no camera 'cable' in the observation` | -- |

How to read this:

* **The scene the port builds is the source's scene.** For `cable_route/port_easy` at
  seed 10009, both stacks build the same cable (50 links, 588.2 mm, 11.228 g, identical
  end poses) and the same 70 bodies / 70 joints; every suite-defined joint, shape and
  material matches. What differs is upstream's, not the suite's: Newton 1.7's MJCF
  import gives the YAM arm and finger joints limit gains ke 10000 / kd 10 where Newton
  1.5 gave 2500 / 100, and the YAM fingertip decomposition has three more convex pieces
  per finger. Those are the robot and engine the release is pinned to, so they are left
  as upstream has them.
* **On USB-C the port reproduces the source.** On the one USB-C Easy scene run on both
  stacks, the graph takes the same stage sequence with the same printed geometry (plug
  at (439.1, -67.7) vs (439.1, -67.6) mm, grasp width 10.91 mm on both) and ends in the
  same place to within a millimetre.
* **On routing the two stacks diverge on the one shared scene.** Both miss the terminal
  the same way (the orange block under the 3,400 px floor, so the graph never attempts
  the endpoint insertion), so neither run is the graph working as designed; the old run
  scored because the free cable end settled inside the port box anyway. The port
  weaves both stations on the first pass (the old run needed its restore pass). One
  scene each cannot separate this from contact nondeterminism plus the robot
  differences above; see the four-scene comparison below.

### Routing Easy, five selected scenes on both stacks (protocol, `gt`/weave graph)

| Scene (seed) | Port: reward / seat0 / seat1 / port box | Old runtime: reward / seat0 / seat1 / port box |
| --- | --- | --- |
| -10 (10009) | 0.50 / yes / yes / no | **1.00** / yes / yes / yes |
| -63 (10062) | 0.50 / yes / yes / no | 0.50 / yes / yes / no |
| -85 (10084) | 0.50 / yes / yes / no | 0.50 / yes / yes / no |
| -44 (10043) | 0.50 / yes / yes / no | 0.75 / yes / yes / yes (not a success) |
| -54 (10053) | 0.25 / yes / no / no | 0.50 / yes / yes / no |
| **Success** | **0/5** | **1/5** |

**Since resolved:** the 3,400 px floor was measured
on the suite's 1280x960 `cable` view, and `yam_station_cable@v1` renders the same pose
at 1280x720. The floor is now scaled by the frame's `fx * fy`.

**The terminal is never detected on either stack (10/10 runs).** The weave graph's
`plug_port.py` needs an orange-block mask of at least `BLOCK_MIN_PX = 3400` px; the
palette masks measure 2,513-2,562 px on the old runtime and 2,081-2,185 px on the port
(the port's RTX render of the block is ~15 % smaller in the `cable` view), so the
endpoint insertion is skipped on every run and the port box is reached, when it is,
only by the free end settling there. Weaving itself works on both: 5/5 (old) and 4/5
(port) runs seat both stations. So the delivered routing protocol's terminal stage
does not function as shipped, independently of this port; tuning the floor (or the
`cable` camera) is a content change outside a conservative port and is left for the
suite owners.
* **USB-C Easy under the palette protocol fails 7/7 here, on both stacks for the scene
  checked.** The plug is grasped, re-observed and stroked, then released at the mouth
  ~15 mm off the bore. The reseat step reports `no supported 'orange plug' read`, so
  the final correction never runs. The privileged `gt` graph inserts on the same task,
  and the medium `mate` run inserts, so the physics, cable chain and mate contact are
  sound; this is the vision `mate` graph's easy-tier behaviour in the delivered source,
  not something the port introduced. Not fixed (out of a conservative port's scope).
  **Since found and fixed.** The
  graph's bench read was written for `usbc_perception@v1`'s straight-down camera. On the
  documents' `yam_station_usbc@v1` head camera, the port's side wall biased it ~9 mm and
  ~7°, and `inspect_vision` then rejected its own correct wrist read. Segmentation was
  not the cause: SAM3 fails the same way.
* **The routine has no terminal step.** `RoutineWeave` seats stations; on the port
  tiers the terminal is the weave graph's `terminating-the-run` skill. A routine number
  on `port_easy`/`port_medium` is therefore not a reference.
* **`graphs/weave` is a vision graph.** It needs `--perception vision` (camera frames);
  the default `oracle` gives it no `cable` camera.

## Not validated

* Routing Medium and any batch beyond the runs above (no success-rate claim is made).
  **Since measured and fixed.** The crossing's open
  jaws came down on the next spool on L2's 80 mm pitch, and the Medium-only terminal branches
  let the end spring out of the port.
* `weave_vision`, `usbc_insert` `gt` on medium/bulkhead/port, the routine on the other
  weave tiers.
* SAM3 segmentation (`--segmentation sam3` in the old runner): needs gated weights.
* Fleet runs (`--num-worlds > 1`) of these suites.

## Requested shared changes (outside this fork's paths)

1. **`vabar bench` / the README must set `ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette`**
   for the four cable cells, and pass `--perception vision --no-privileged
   --settle-steps {600|60} --bus-timeout 600 --renderer rtx` with graph `mate`
   (USB-C) or `gt` (routing). Instance -> episode id: the manifest's recorded seed.
2. **Other suites likely have the same `assets/` gap.** Upstream's loader no longer walks
   `assets/`; any suite whose records/atoms live only under `assets/` and are not imported
   by `staging.py`/`scenes.py` needs a `records.py` like the two here (e.g. `gear_mesh`'s
   `gears_seated` atom is registered in `assets/records.py`).
3. `pyproject` `cable` extra is sufficient as is (opencv-python, scikit-image, pillow
   cover TrackDLO); no change needed.
4. **Do not quote the paper's cable numbers as reproducible from this tree.** Under the
   shipped palette protocol USB-C Easy fails 7/7 here (the one scene also run on the old
   runtime fails there identically) and Routing Easy's terminal is never detected on
   either stack (10/10 runs); the paper's 86.7 % / 46.7 % were
   evidently measured under a different configuration (e.g. SAM3 segmentation or
   different graph constants). The suite owners should say which. (USB-C Easy:
   since fixed.)
   *Routing: it was the camera.* The delivery's `cable` view was 1280x960, and the
   terminal floor is in its pixels.

## Reproducing the A/B

The old runtime was rebuilt from the source repository's own frozen lock
(`UV_PROJECT_ENVIRONMENT=<venv> uv sync --frozen --extra cable --python 3.12` in the
source checkout) and run through its own `run_cable.py run-scene`. Scratch logs for every
run in this file: `scratchpad/runs/cable/` (not part of the release).
