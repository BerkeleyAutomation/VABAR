# Arena GaP adapter

`isaaclab_arena_gap` connects the RSS version's GaP manipulation graphs to industrial Isaac Lab Arena
environments. Arena owns simulation, task scoring, recording, and episode JSONL. The graph runs in
AUTOLab's own environment through RoboSimStudio's policy agent (the RSS adapter, `rss_adapter/`),
receives calibrated observations and returns robot targets through a synchronous msgpack bridge.

## Runtime interface

```text
Arena environment                              GaP process (scripts/run_rss_graph.sh)
-----------------                              --------------------------------------
GapRemotePolicy                                rss_adapter.runner: AUTOLab's graph,
  RGB-D, joints, gripper, calibration  ------>   RoboSimStudio's runtime and skills
  arm and gripper targets              <------
```

One GaP process serves one Arena environment. Managed jobs start GaP lazily on the first policy
step and reap its complete process group at reset or shutdown. Until GaP marks a command channel
valid, the bridge holds the observed robot state. Arena's episode JSONL remains authoritative even
when GaP reports graph completion.

The maintained robot profiles are the ones `rss_adapter/runner.py` `ROBOTS` binds:

- `fr3`: seven FR3 joints and the Robotiq closedness action.
- `yam_bimanual`, `yam_bimanual_wide`, `yam_bimanual_linear`: two six-joint YAM arms, each with
  its continuous gripper action, on the crank, wide-crank or linear hand.

Each sends named RGB-D, live intrinsics, and live OpenCV-optical camera poses in the live
robot-base frame. Kinematics, IK and motion planning are RoboSimStudio's, in the GaP process.

## Setup

From the repository root:

```bash
./setup.sh
```

The root `.venv` contains Arena, Isaac Lab, Isaac Sim, `isaaclab_arena_vabar`, and `isaaclab_arena_gap`.
The RSS adapter runs in the RSS version's own environment (`AUTOLAB_VABAR_VENV`, synced from AUTOLab's
lock; `docs/SETUP.md` section 3 says where it lives); the five skill bundles keep separate Python 3.11 environments. VLM credentials belong in
`$HOME/.config/gap/vlm.env`; the repository supplies overrideable provider, endpoint, and model
defaults. A repository-owned registry override adapts the pinned VLM bundle's GPT-5 request shape
for provider-qualified model IDs while reusing its exact environment. See `docs/SETUP.md`.

## Run jobs

`experiment_configs/<task-family>/*.yaml` contains typed Arena Experiment
Definitions. Baselines and sensitivity Experiments use the same interface
locally and through OSMO.

The default managed job is the easy wrench-hanging pair:

```bash
uv run vabar-arena run
```

Run the managed syringe graph after setup; the policy starts and reaps GaP automatically:

```bash
uv run vabar-arena run \
  isaaclab_arena_gap/experiment_configs/syringe_disposal/vabar_syringe_disposal_single_gap_baseline.yaml \
  --skip-setup --no-headless --viz kit
```

Use `vabar_syringe_disposal_cluttered_gap_baseline.yaml` for the six-syringe cluttered variant.

The three randomized tool-sorting tasks share AUTOLab's `tool_sort/gap_perception_loop` graph:

```bash
uv run vabar-arena run \
  isaaclab_arena_gap/experiment_configs/tool_sorting/vabar_tool_sorting_easy_1_gap_baseline.yaml \
  --skip-setup --no-headless --viz newton
```

Substitute `2` or `3` for the other inventories. Arena episode JSONL remains authoritative;
graph completion alone is not task success.

## OSMO evaluator image

`docker/` builds the exact-source worker used by Arena-owned OSMO
Experiments. It supports clean release builds and explicit, fully recorded dirty
development builds. See [docker/README.md](../docker/README.md) for the image
interface, provenance contract, build command, and one-episode OSMO smoke. The
`osmo/` package contains submission code only; reusable Experiment Definitions
live under `experiment_configs/`.

The catalog contains 22 single-environment baseline Experiments, all managed GaP
remote-policy Experiments that run the RSS version's graph through the RSS adapter
(`rss_adapter`, launcher `scripts/run_rss_graph.sh`). Five additional
Experiments (cable easy and medium; gear easy, easy pair and medium train) evaluate the published
layout seeds. Syringe has four GaP
baselines: single, both, designated and cluttered, all using
`syringe_packing/gap_perception`.

Every GaP job uses the managed auto-spawn lifecycle. Supported pairs name a graph bound in
`rss_adapter/runner.py` `BINDINGS`, the AUTOLab document where the level or variant needs one
(`gap_inputs.autolab_task`) and, only where a recorded reason requires it, a package-owned `rss_*`
graph adaptation (`graph_adaptations/README.md`).

## Inspect

`show` loads the same benchmark registrations without starting GaP:

```bash
uv run vabar-arena show --env vabar_tool_sorting__easy_1
uv run vabar-arena show --env vabar_tool_hanging__wrench_medium --enable_cameras
```

Inspection holds the robot still and disables scoring and recorders.

## Tests

The adapter suite contains four maintained public contracts:

1. GaP/Arena transport, observation/action adapters, lifecycle, and result sidecars.
2. Static validation of every baseline Arena Experiment YAML.
3. One opt-in fresh-process build/reset/step smoke per baseline Experiment YAML.
4. Idempotent registration plus launcher entrypoints.

Baseline-policy behavior tests are intentionally deferred. The simulator-free suite is part of the
standard checks in [docs/SETUP.md section 7](../docs/SETUP.md#7-run-the-non-gpu-tests-and-preflight).

Run the 22 fresh-process baseline Experiment smokes with a visible CUDA device:

```bash
ISAAC_CAP_RUN_LIVE_TESTS=1 \
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  uv run pytest -q isaaclab_arena_gap/tests/test_arena_experiments.py -m live
```

## Ownership

`isaaclab_arena_gap` owns the bridge, robot adapters, the RSS adapter, Experiment Definitions, the
`rss_*` graph adaptations, launchers, and bridge tests. `isaaclab_arena_vabar` owns policy-independent scenes,
assets, embodiments, tasks, scoring, and registrations. Arena owns the generic runner and
environment framework. The pinned GaP and skill repositories remain unmodified.
