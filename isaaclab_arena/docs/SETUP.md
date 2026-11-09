# Setup: GaP in Isaac Lab Arena

This guide takes a clean Linux machine to a scored Arena run of AUTOLab's GaP graphs. The code is
the `isaaclab_arena/` folder of the VABAR repository
(`BerkeleyAutomation/AUTOLab-GaP-NVIDIA-Collaboration`, branch `vabar-robosimstudio`), next to the
RSS (RoboSimStudio) version of the same tasks at the repository root (section 4). For a result you
want to reproduce later, record the commit you ran together with the Experiment YAML and the result
JSONL (section 9).

You will need, before running `./setup.sh`:

- an NVIDIA GPU with 24 GB of memory and the host tools in section 2 (`git`, `uv`
  0.9.x, CMake older than 4, a C++ compiler);
- read access to the VABAR repository;
- a VLM API key (`GAP_VLM_API_KEY`) and a Hugging Face token (`HF_TOKEN`) with access to the gated
  `facebook/sam3` model, in `$HOME/.config/gap/vlm.env` (section 5). Setup's final preflight fails
  without the VLM key.

Every runnable pairing is a typed Arena Experiment under `isaaclab_arena_gap/experiment_configs/`, grouped by
task family. Every one runs AUTOLab v1.0's graph through the RSS adapter (`isaaclab_arena_gap.rss_adapter`,
section 3). The supported GaP pairs are the four syringe tasks with
`syringe_packing/gap_perception`; the Gear Mesh single, pair, and train families with
`gear_mesh/gap_perception`; the three easy tool-sorting tasks with `tool_sort/gap_perception_loop`; the easy
wrench, screwdriver, plier, and scissors tasks and their medium variants with `tool_hanging/gap_perception`
(the medium `*_level2` documents resolve it to AUTOLab's level-2 workflow); USB-C easy/medium with
`usbc_insert/mate`; and cable easy/medium with `cable_route/gt` (AUTOLab's `graphs/weave`). Camera rigs, embodiments, lighting, and physics
ownership are described in [DESIGN.md](DESIGN.md).

## 1. What Runs

One `isaaclab_arena_gap.arena_runner` process registers this folder's environments, robots and
policy with Arena, delegates to Arena's experiment runner, and creates and scores the environment.
`GapRemotePolicy` exchanges one observation/action per simulation step with a separate GaP process.
Jobs with `auto_spawn=true` start that GaP process themselves and stop it at reset or shutdown; every
shipped GaP job does this, so you never start GaP by hand.

The policy uses Grounding DINO, a configured VLM, SAM3, geometry, and cuRobo. It is not an end-to-end
VLA. The VLM selects and verifies visual detections inside a static graph.

## 2. Validated Reference Machine

This is the configuration used for acceptance, not a requirement that every field match exactly.

- Ubuntu 24.04
- NVIDIA RTX 4090, 24 GB VRAM
- NVIDIA driver 580.159.04
- CUDA toolkit 12.8 at `/usr/local/cuda`
- `git`
- network access to Hugging Face: setup downloads the scene assets (about 5 GB) from
  `huggingface.co/keplerccc/VABAR-arena-assets`
- read access to the VABAR repository; the Arena, GaP runtime and skills sources are public
- network access to Isaac Lab's asset root: every graph scene's warehouse HDR dome light is resolved
  from Isaac Lab's asset URL (`ISAACLAB_NUCLEUS_DIR`) rather than copied into this repository

These host tools are requirements, not just the reference configuration:

- `uv` 0.9.x first on `PATH`. Setup checks the exact `uv pip check` output of uv 0.9 (step 7 of
  section 6), and later releases change it (uv 0.12 adds a deprecation warning), so `./setup.sh`
  and preflight refuse any other series. `uv self update 0.9.25` pins a standalone install;
  otherwise `uv tool install uv==0.9.25` and put uv's tool directory first on `PATH`.
- CMake 3.x (older than 4) and a C++ compiler (`build-essential`) on `PATH`. The root sync builds
  `egl-probe` from source, and its `CMakeLists.txt` declares a minimum that CMake 4 no longer
  accepts. On a host whose system CMake is 4.x, `uv tool install 'cmake<4'` provides one.

The native environment uses Python 3.12 and Arena 0.3.1's frozen
lock: Isaac Lab 17.0.2 from the pinned Isaac Lab 3.0 EA source commit
`91a0906f93802eb859192dcb0bca879f891e92d6`, Isaac Sim 6.1.0.0, and Newton
1.5.2. GaP uses its own Python 3.11 environment and per-skill environments.

## 3. Repositories and Pins

| In-tree checkout | Remote | Immutable revision | Required |
|---|---|---|---|
| `.` | the `isaaclab_arena/` folder of the VABAR repository (`BerkeleyAutomation/AUTOLab-GaP-NVIDIA-Collaboration`, `vabar-robosimstudio`); the RSS version (AUTOLab v1.0) is the repository root | The commit you check out | Yes |
| `external/gap/graph-as-policy` | `graph-robots/graph-as-policy` | `71549e1584839337eb355c280e7b1e50c805b8c7` | Yes |
| `external/gap/open-robot-skills` | `graph-robots/open-robot-skills` | `752dfbd96968d4d08379f8111765efbf6457800b` | Yes |
| `external/IsaacLab-Arena` | `isaac-sim/IsaacLab-Arena` (`release/0.3.1`) | `c7edf3a9d5d8276565208f72422f8f8a0806d6cf` | Yes, release 0.3.1 + native venv |

The three gitlinks and the checked-out commit of the VABAR repository are the executable source
pins. `open-robot-skills` is upstream `main` and supplies the
`curobo`, `geometry`, `grounding-dino`, `sam3` and `vlm` tool bundles with scalar public tool schemas;
AUTOLab's runtime boots each out-of-process from its own environment under
`external/gap/open-robot-skills/tools/<bundle>/.venv`. `graph-as-policy` is upstream `main`: its
`gap skills install` builds those bundle environments, and each of them installs its `gap-core`
(the bundle RPC server) editable from this checkout.

The RSS version at the repository root is AUTOLab's v1.0 release, used by
`isaaclab_arena_gap.rss_adapter`: every shipped
experiment config sets `gap_launcher: isaaclab_arena_gap/scripts/run_rss_graph.sh` and runs AUTOLab's graph
through RoboSimStudio's own policy agent, in AUTOLab's locked environment
(`AUTOLAB_VABAR_VENV`, here `.venv-autolab` in this folder, plus `usd-core` and
`newton-usd-schemas` to load Arena's robot USD).
The binding per graph (`rss_adapter/runner.py`) carries the v1.0 benchmark cell's flags and environment;
an experiment whose level or variant uses a different AUTOLab document names it in
`gap_inputs.autolab_task` (gear pair/train, syringe variants, USB-C, cable, and each tool-hanging tool
and level), and that document also resolves the graph name. Only a few `rss_*` graph adaptations remain,
for scene differences Arena keeps on purpose (`isaaclab_arena_gap/graph_adaptations/README.md`). The
adapter is the only GaP runtime: a config that omits `gap_launcher` runs `run_rss_graph.sh` too.
Cable routing (`cable_route/gt`, which the port documents resolve to `graphs/weave`) also names the
document's static fixtures in `gap_scene_statics` (`peg0: peg_0`, ..., `anchor`, `port`): the session
measures those kinematic Arena assets on the first step and the adapter hands them to AUTOLab's tools as
`handles.statics`, as AUTOLab's compiled scene does under `--no-privileged`. The cable itself is never sent.

The Arena pin is the `release/0.3.1` branch: upstream `main` with exact episode limits plus the
0.3.1 backports, including the `OnTreeInitializer` placement default. It pins Isaac
Lab `91a0906f93` (3.0 EA with the multiccd configuration fix). Arena's placement-candidate seeds
overlap across neighbouring placement seeds in 0.3.1; `isaaclab_arena_vabar.placement_seed_compat`,
installed by `register_components()`, spreads them. Arena owns Python
3.12, Isaac Sim/Lab/PyTorch dependencies, and the authoritative frozen lock. Setup syncs that lock
into the repository root `.venv` and installs Arena, `isaaclab_arena_gap`, and `isaaclab_arena_vabar` there
as editable overlays. Arena itself contains none of these benchmark environments; this folder
registers them with Arena at startup.

## 4. Clone the Exact Sources

```bash
git clone https://github.com/BerkeleyAutomation/VABAR.git
cd VABAR/isaaclab_arena
```

All later commands run from that `isaaclab_arena/` directory. To reproduce a recorded evaluation,
check out its commit before setup.

Do not use `--recurse-submodules`. `./setup.sh` initializes the top-level gitlinks,
Arena's pinned Isaac Lab child, and only GaP's required VAB and robosuite children. Arena's GR00T
submodule and GaP's unrelated LIBERO-PRO/real-robot trees remain uninitialized. The GaP
repositories remain siblings under `external/gap/`.

The scripts resolve the `external/` submodule layout by default. AUTOLab's tasks, graphs and lock
resolve to the repository root, the directory enclosing this folder
(`isaaclab_arena_gap/scripts/autolab_layout.sh`, and `isaaclab_arena_gap.task_sources.autolab_root()`
for tests and tools); the RSS adapter's environment is `.venv-autolab` in this folder, which leaves
the repository root's own `.venv` alone. A development machine with separately cloned checkouts
can point every script at them via environment variables instead:
`GAP_ROOT`, `SKILLS_ROOT`, `AUTOLAB_VABAR_ROOT` (with `AUTOLAB_VABAR_VENV`, default
`$AUTOLAB_VABAR_ROOT/.venv`), and `ARENA_SOURCE_ROOT`; use
`ARENA_PYTHON` to run an authoring tool with an explicitly provisioned interpreter. `./setup.sh`
rejects an alternate `ARENA_ENV_ROOT`: the supported simulation environment is always the root
`.venv`, which keeps unqualified `uv run` commands consistent. Explicit source-root values always
win over the `external/` defaults.

## 5. Configure Credentials

Never put credentials in a repository. Create a private environment file:

```bash
mkdir -p "$HOME/.config/gap"
${EDITOR:-vi} "$HOME/.config/gap/vlm.env"
chmod 600 "$HOME/.config/gap/vlm.env"
```

The private file needs only credentials:

```bash
export GAP_VLM_API_KEY=your-key
export GH_TOKEN=your-github-token
export HF_TOKEN=your-hugging-face-token
```

`GH_TOKEN` is read only by the Docker image builder (`docker/build.sh`); a native setup does not
need it. Without the file, preflight and the GaP launcher also accept `GAP_VLM_API_KEY` exported in
the process environment, which is how OSMO injects it into the evaluator image.

`GAP_VLM_PROVIDER`, `GAP_VLM_BASE_URL` and `GAP_VLM_MODEL` select the VLM route. The repository
defaults (`isaaclab_arena_gap/scripts/vlm_env.sh`) are `GAP_VLM_PROVIDER=openrouter`, which names
the bundle's generic OpenAI-compatible HTTP adapter, an NVIDIA-hosted OpenAI-compatible endpoint,
and `GAP_VLM_MODEL=openai/openai/gpt-5.6-sol`. To use your own provider, set the three in the
process environment or the private file, for example
`GAP_VLM_BASE_URL=https://<your-openai-compatible-endpoint>/v1`; they override the defaults.

`isaaclab_arena_gap/skill_overrides/tools/vlm` shadows only the pinned VLM bundle, through GaP's
multi-registry precedence, so that GPT-5 is recognized from the final component of a
provider-qualified model ID (otherwise the bundle sends the `temperature` field that GPT-5.6
rejects). The override executes the otherwise unchanged pinned source and reuses its locked
environment; `./setup.sh` prepares that environment link.

Anthropic can use `ANTHROPIC_API_KEY`. Vertex uses `GAP_VLM_PROVIDER=vertex`, project/region/model
variables, and Application Default Credentials. `HF_TOKEN` must belong to a Hugging Face account
that has been granted access to the gated `facebook/sam3` weights. It is needed only for their first
download; a token stored by `hf auth login` works in its place. The Docker builder also falls
back to tokens from `gh auth token` and `hf auth token`; it passes both through BuildKit secrets and
does not persist them in an image layer.

## 6. Set Up the Locked Environments

```bash
./setup.sh
```

The shell entrypoint exists before Arena's gitlink or any Python environment is available. It
selects Python 3.12 through `.python-version`, prepares the root simulation environment, and
delegates GaP, AUTOLab v1.0 and the skill bundles to their independent locks. Setup uses an existing `CUDA_HOME`
or auto-detects `/usr/local/cuda`; on a module-based system, load the site CUDA module or set
`CUDA_HOME` first. After the first setup, `uv run vabar-arena setup` remains an equivalent convenience
command.

Behind the launcher, the idempotent setup performs, in order:

1. exact selective submodule initialization;
2. GaP `uv sync --frozen` with Python 3.11;
3. AUTOLab v1.0 `uv sync --frozen --no-default-groups` into `AUTOLAB_VABAR_VENV`, the RSS adapter's
   environment, plus `usd-core` and `newton-usd-schemas` to load Arena's robot USD;
4. locked Python-3.11 environments for `curobo`, `geometry`, `grounding-dino`, `sam3`, and `vlm`;
5. Arena-owned `uv sync --frozen`, targeted at the repository root `.venv`, with Python 3.12 and
   the lock-selected Isaac Lab/Isaac Sim/Newton/PyTorch stack;
6. explicit editable Arena, `isaaclab_arena_vabar/`, and `isaaclab_arena_gap/` overlays with their declared
   runtime dependencies, constrained to the frozen lock's `idna==3.10`; this does not activate
   Arena's old simulator dependency groups; and
7. one captured `uv pip check` plus the runtime preflight. A zero exit is accepted. A nonzero exit
   is accepted only when its complete diagnostics contain the normal UV check summary and exactly
   Arena main's five known conflicts: openpi-client/NumPy, isaacsim-core/Newton 1.5.0,
   isaacsim-kernel/WebSockets, the absent onnxruntime-gpu wheel, and ovphysx/packaging. Any
   other command error, diagnostic, duplicate, missing conflict, or additional conflict is fatal.

The GaP lock is intentionally frozen rather than re-resolved. Its
`open-robot-skills` source is the editable sibling gitlink, so the code comes
from the checked `752dfbd` checkout. Its source-only skill changes add no dependencies, so
GaP's older lock metadata may fail `uv lock --check` without changing the environment. A future
ORS pin that adds a dependency must be paired with a compatible upstream GaP lock instead of
extending this exception.

The selected source environment deliberately replaces Isaac Sim's older package requirements;
`uv pip check` therefore cannot be a zero-only gate. Setup applies the exact fail-closed profile
above rather than treating every nonzero dependency check as acceptable.

Pass `--offline` with a populated UV/model cache. Use `--skip-preflight` when preparing packages on
a login/build node before GPU visibility or credentials are available. `--skip-skills` is intended
for environment checks only; `--source-only` verifies the complete source tree without creating any
environment. A later exact Isaac Lab sync removes the editable overlays; rerun `./setup.sh` or
reinstall them afterward with:

```bash
uv pip install --python .venv/bin/python \
  --editable external/IsaacLab-Arena \
  --editable isaaclab_arena_vabar \
  --editable isaaclab_arena_gap
```

### MJWarp runtime

The simulator uses the MuJoCo-Warp release pinned by Isaac Lab's lock; `check_setup.sh` checks the
installed version against it. Deterministic physics stays off: requesting it breaks Newton's CUDA
graph capture, and Isaac Lab's MuJoCo-Warp 3.11 loses convex contacts in that mode.

### Shared cache materialization

Package locks do not include model weights, the host CUDA driver/toolkit, VLM credentials, or
staging USD assets. On an internet-connected build node, set shared cache locations before setup,
then prefetch the gated models after loading the private environment:

```bash
export CAP_CACHE_ROOT="${CAP_CACHE_ROOT:-$HOME/.cache/vabar}"
export UV_CACHE_DIR="$CAP_CACHE_ROOT/uv"
export HF_HOME="$CAP_CACHE_ROOT/huggingface"
set -a
. "$HOME/.config/gap/vlm.env"
set +a
external/gap/graph-as-policy/.venv/bin/gap skills check \
  --skills "$PWD/external/gap/open-robot-skills" --download
```

Do not run concurrent setup commands against the same checkout or bundle environments. Build once,
then expose the resulting checkout/cache read-only to jobs or use it as the input to a container
image. Keep credentials outside the repository and inject them per job.

An upstream `numba`/`coverage==7.4.4` incompatibility affects unrelated Isaac Lab features that
import `numba`; do not alter the lock to work around unrelated imports.

## 7. Run the Non-GPU Tests and Preflight

This is the standard check before handing work over:

```bash
GAP_ROOT="${GAP_ROOT:-$PWD/external/gap/graph-as-policy}"
PYTHONPATH=isaaclab_arena_gap/src \
  "$GAP_ROOT/.venv/bin/python" -m pytest -q isaaclab_arena_gap/tests

OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  uv run pytest -q

./isaaclab_arena_gap/scripts/check_setup.sh
```

The first test run covers what needs a GaP runtime: the Arena bridge transport the RSS adapter
drives, the graph-result sidecars, the launcher, and the artist-wrench CAD replay through
AUTOLab v1.0's graph scripts. The root test run covers the Arena policy, transport, robot adapters,
the RSS adapter's bindings and adaptations, and simulator-free industrial contracts. It collects only
`isaaclab_arena_gap/tests` and `isaaclab_arena_vabar/tests`; live tests are skipped by default, so this
command does not initialize Isaac Sim. Preflight verifies repository ancestry, Python environments,
the actual Arena-policy import, VLM/Hugging Face configuration, GPU visibility, and the canonical
managed lifecycle. A pin, missing environment, or failed policy import is fatal; a dirty worktree is
a warning. Set `CAP_CHECK_EXTERNAL_PORT=1` when using the accepted fixed-port external GaP flow.

Live tests are opt-in and need a visible CUDA device. The industrial suite starts one fresh Isaac
process per movable-asset drop and per environment YAML; asset verdicts are test outcomes:

```bash
ISAAC_CAP_RUN_LIVE_TESTS=1 \
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
uv run pytest -q \
  isaaclab_arena_vabar/tests -m live
```

Each environment smoke constructs and resets one YAML in a fresh process, applies ten finite hold
steps, and checks finite observations and reward; it does not run a policy or claim task success.
To run only one family, for example tool hanging:

```bash
ARENA_SOURCE_ROOT="${ARENA_SOURCE_ROOT:-$PWD/external/IsaacLab-Arena}"
ISAAC_CAP_RUN_LIVE_TESTS=1 \
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
uv run pytest -q \
  isaaclab_arena_vabar/tests/environments/test_environment_smoke.py \
  -k tool_hanging
```

The Arena job catalog adds one fresh-process smoke per baseline Experiment YAML (22): each
constructs the selected environment, resets, executes ten finite hold steps, and closes:

```bash
ISAAC_CAP_RUN_LIVE_TESTS=1 \
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  uv run pytest -q isaaclab_arena_gap/tests/test_arena_experiments.py -m live
```

The Cable Easy/Medium parity test against AUTOLab's own cable runtime is also opt-in. It runs on CPU against AUTOLab's pre-v1.0
cable runtime (`run_cable`), which v1.0 no longer ships: `CABLE_NATIVE_ROOT` names a checkout of it
(its commits are in the VABAR repository's history) and `CABLE_NATIVE_PYTHON` an interpreter that can
import it:

```bash
git -C .. worktree add /tmp/autolab-cable 82e414c
CABLE_NATIVE_ROOT=/tmp/autolab-cable \
CABLE_NATIVE_PYTHON=/path/to/autolab/cable/python \
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  uv run pytest -q isaaclab_arena_vabar/tests/environments/test_cable_native_declarations.py
```

## 8. Run the Supported Pairs

`uv run vabar-arena run <experiment.yaml>` prepares the environments (skip with `--skip-setup` once
they exist) and launches Arena. `--no-headless` opens the Isaac Sim viewer and `--viz` selects
visualizer backends (`kit` needs a display; `newton`, `rerun`, and `viser` also work headless).
`GapRemotePolicy` starts and reaps GaP itself.

With no argument, `run` selects the default easy wrench-hanging pair:

```bash
uv run vabar-arena run \
  isaaclab_arena_gap/experiment_configs/tool_hanging/vabar_tool_hanging_wrench_easy_gap_baseline.yaml
```

It selects `vabar_tool_hanging__wrench_easy` and runs AUTOLab's `tool_hanging/gap_perception`
through the RSS adapter with the `rss_tool_hanging_artist_wrench` adaptation. Substitute `scissors`,
`screwdriver`, or `plier` for the other easy families (each names its AUTOLab `*_level1` document), and
use the `*_medium_gap_baseline.yaml` files for the two-tool medium variants, whose `*_level2` documents
resolve `gap_perception` to AUTOLab's level-2 workflow:

```bash
uv run vabar-arena run \
  isaaclab_arena_gap/experiment_configs/tool_hanging/vabar_tool_hanging_screwdriver_medium_gap_baseline.yaml \
  --skip-setup
```

Syringe disposal (substitute `both`, `designated`, or `cluttered` for the other variants):

```bash
uv run vabar-arena run \
  isaaclab_arena_gap/experiment_configs/syringe_disposal/vabar_syringe_disposal_single_gap_baseline.yaml \
  --skip-setup --no-headless --viz kit
```

Four-object tool sorting (substitute `2` or `3` for the other inventories):

```bash
uv run vabar-arena run \
  isaaclab_arena_gap/experiment_configs/tool_sorting/vabar_tool_sorting_easy_1_gap_baseline.yaml \
  --skip-setup
```

Gear Mesh, USB-C, and cable routing follow the same pattern with the files under
`isaaclab_arena_gap/experiment_configs/gear_insertion/`, `isaaclab_arena_gap/experiment_configs/usbc_insertion/` and `isaaclab_arena_gap/experiment_configs/cable_routing/`.
For example, Cable Medium:

```bash
uv run vabar-arena run \
  isaaclab_arena_gap/experiment_configs/cable_routing/vabar_cable_routing_medium_gap_baseline.yaml \
  --skip-setup
```

### Parallel Gear Mesh layouts

For `vabar_gear_insertion__easy_pair` and
`vabar_gear_insertion__medium_train`, set `environment.layout_names` to
one distinct RSS layout name per `environment_builder.num_envs` slot. Names
come from `isaaclab_arena_vabar/src/isaaclab_arena_vabar/tasks/gear_insertion/variants.json`;
sample names without replacement across all jobs before splitting them into slot lists. Keep
`one_episode_per_env: true`: resets reuse the assigned source start. Each episode records its
layout as `gear_layout`. Omitting names retains the existing scalar variation behavior for
`num_envs: 1`.

### Inspect

`show` loads the same registrations without starting GaP:

```bash
uv run vabar-arena show --env vabar_tool_sorting__easy_1
uv run vabar-arena show --env vabar_tool_hanging__wrench_easy --viz kit --enable_cameras
```

Add `--skip-setup` when the environments are already prepared. Do not add `--presets` with `--env`;
the registered factory owns the Newton profile and a preset replaces it. Inspection holds the robot still and disables scoring
and recorders.

### Evaluate your own policy

The environments, scoring and recording are Arena's and do not depend on GaP, so any Arena policy
can run them and is scored the same way, in the episode JSONL (section 9). Copy one of the baseline
YAMLs above and replace its whole `policy:` block. `type` takes either a registered Arena policy name
(for example `zero_action`) or the dotted import path of a policy class; the other keys under
`policy:` are that class's config fields. Drop every GaP key (`gap_*`, `robot_profile`, `auto_spawn`,
`policy_device`, ...): they are fields of `GapRemotePolicy`'s config, and Arena rejects keys the
selected policy's config does not declare. A minimal check of the plumbing:

```yaml
runs:
  gear_easy_zero_action:
    environment:
      type: vabar_gear_insertion__easy
      enable_cameras: true
    environment_builder:
      placement_seed: 42
    policy:
      type: zero_action
    rollout_limit:
      num_episodes: 1
    num_rebuilds: 1
```

```bash
uv run vabar-arena run path/to/my_experiment.yaml --skip-setup
```

`zero_action` sends all-zero joint targets, so it only shows that the environment runs and is
scored; it is not a baseline. Keep `enable_cameras: true`: the run records camera videos, which
needs the cameras.

A policy of your own subclasses `isaaclab_arena.policy.policy_base.PolicyBase[YourCfg]`, where
`YourCfg` is a dataclass subclassing `PolicyCfg`, gives itself a `name`, implements
`get_action(env, observation)`, and is decorated with `isaaclab_arena.assets.register.register_policy`.
Arena imports the module named by the dotted `type`, which registers the class. Make that module
importable from the root `.venv`, either by installing your package into it
(`uv pip install --python .venv/bin/python --editable path/to/your_package`) or by exporting
`PYTHONPATH`, which the launcher keeps.

What the policy sees and sends depends on the robot:

| Robot | Tasks | Observations | Actions |
|---|---|---|---|
| FR3 with Robotiq 2F-85 | gear insertion, syringe disposal, tool sorting | `isaaclab_arena_vabar/src/isaaclab_arena_vabar/embodiments/fr3_robotiq_2f85/config.py` (`IndustrialFr3RobotiqObservationsCfg`) | `.../fr3_robotiq_2f85/actions.py`: 7 absolute arm joint targets, then gripper closedness from 0 to 1 |
| Bimanual YAM | tool hanging, USB-C insertion, cable routing | `.../bimanual_yam/config.py` (`IndustrialBimanualYamObservationsCfg`) | `.../bimanual_yam/actions.py`: left arm (6 joint targets), left gripper, right arm, right gripper |

Proprioception is in the `policy` observation group. With `enable_cameras: true`, RGB and depth
images are in the `camera_obs` group as `<camera>_rgb` and `<camera>_distance_to_image_plane`; the
cameras of each robot and task are in the same embodiment folder's `cameras.py`.

Arena's documentation covers the policy interface, Experiments, and remote policies that run
behind a policy server in another process, such as GR00T or pi0.5:

- [Policy concepts](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/pages/concepts/policy/index.html)
- [Arena Experiments](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/pages/concepts/concept_arena_experiments.html)
- [Your first Experiment](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/pages/quickstart/arena_experiment.html)
- [Running a real policy](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/pages/quickstart/running_a_real_policy/index.html) (GR00T and pi0.5 through a policy server)

## 9. Outputs

The launcher prints an output directory below `outputs/arena/<timestamp>/` in this folder. Watch for the
policy connection, graph stages, and Arena success. A successful graph exit alone is not enough;
the Arena episode JSONL must contain `"success": true`. Find results and recordings with:

```bash
find outputs/arena -type f \
  \( -name 'episode_results*.jsonl' -o -name '*.mp4' \) -print
```

Keep the Experiment YAML, seed, source pins, and result JSONL together for every reported run.

## 10. OSMO Evaluation (optional)

This section applies only to NVIDIA's OSMO cluster; local runs (section 8) do not need it. The
exact-source evaluator image and its build/run contract are documented in
[docker/README.md](../docker/README.md). Published results require an
immutable base and final image digest. Local development may use `--allow-dirty`; the image then
contains and records tracked plus non-ignored untracked runtime-source changes while still rejecting
dirty dependency gitlinks. Build and publish the lightweight collector separately with
`docker/build_collector.sh`; submit both by immutable digest, and rebuild both from the
revision being evaluated. Render a one-episode workflow before submitting it:

```bash
uv run python -m isaaclab_arena_gap.osmo.submit_gap_experiment --dry_run \
  --experiment_cfg isaaclab_arena_gap/experiment_configs/tool_sorting/vabar_tool_sorting_easy_1_gap_baseline.yaml \
  --collector_image <registry>/arena-gap-collector@sha256:<digest> \
  experiment_runner.image=<registry>/arena-gap-evaluator@sha256:<digest> \
  osmo.workflow_name=tool-sort-smoke \
  experiment_cfg.runs.vabar_tool_sorting_easy_1_gap.rollout_limit.num_episodes=1
```

`--serial_runs` batches a multi-Run Experiment into one evaluator task, still one Isaac process per Run. Persistent Isaac/Kit cache
reuse is opt-in through `--cache_mode auto`, `--cache_artifact_root_url`, and an explicit
`--cache_driver_branch`; see the Docker README for the cache contract. OSMO injects the `cap-vlm`
credential into evaluator tasks; never put its API key in the image or Experiment YAML.

OSMO collection downloads Arena results and video indexes plus evaluator-image provenance and per-Run
status. Each evaluator publishes its full output under `<output_url>/runner-outputs/<task-name>/`;
keep that directory available alongside the combined report, which links to the original videos.
Evaluator and cache-warmup jobs leave camera-video recording off. Policy RGB-D cameras remain
enabled.

## 11. Documentation and Authoring Tools

- `uv run tools/generate_asset_docs.py` runs each dynamic prop in a fresh Isaac Sim process and
  writes per-prop thumbnails, collision images, drop GIFs, plus the catalog at
  `isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets/README.md`. Acceptance thresholds and
  artifact layout are in [ASSET_DOCS.md](ASSET_DOCS.md).
- `uv run tools/generate_environment_docs.py` captures the first five simulated seconds of every
  environment YAML and writes the table at
  `isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/README.md`. `--check` validates the
  committed README and animations without starting Isaac Sim; `--resume <outputs/environment-docs/run-...>`
  continues an interrupted campaign.
- Curated environments and assets are maintained by hand.

## 12. Common Failures

| Symptom | Check |
|---|---|
| GaP cannot find skill bundles | Rerun `./setup.sh` and confirm both pinned checkouts are present under `external/gap/`. Setup passes `--skills` explicitly and does not trust an ambient registry. |
| Approach IK fails immediately | Inspect the selected YAML object pose and the task's workspace bounds before changing IK. |
| Industrial sharps target is offset or yawed incorrectly | Confirm the skills gitlink is at `752dfbd` and inspect the FR3 `ArmDescription` TCP offset and the RSS robot-model calibration. |
| GaP finishes but Arena reports false | Read the JSONL and final poses/video. Check contact with the bin, gripper state, and episode timeout; graph completion is not task success. |
| Broken pipe prevents result output | Confirm the bridge includes graph-end disconnect handling. The policy holds the robot until Arena terminates. |
| A multi-gigabyte Isaac Sim wheel times out | Rerun `./setup.sh` against the same UV cache. Completed artifacts are retained, and setup defaults to an 1800-second HTTP read timeout. |
| First run is slow | Model loading, VLM latency, cuRobo CUDA compilation, and cold RTX shader compilation dominate. Do not infer a deadlock from startup alone; inspect logs and GPU/process activity. |
| Renderer crash or startup stall with several simulators | Run local simulator jobs serially on one machine. |

## 13. Source Ownership

- Application entry point, pair configuration, bridge/adapters, the RSS adapter, `rss_*` graph
  adaptations, and launchers: `isaaclab_arena_gap`.
- Industrial scenes, tasks, assets, embodiments, scoring, and registration:
  `isaaclab_arena_vabar`.
- Generic simulation, experiment construction, policy lifecycle hooks, scoring, and recording
  framework: Isaac Lab Arena.
- Robot model, IK, motion, task graphs and graph runtime: AUTOLab v1.0 (the RSS version at the repository root, with
  RoboSimStudio and its own locked GaP); generic skill bundles: pinned upstream `open-robot-skills`;
  bundle RPC server (`gap-core`) and bundle installer: upstream `graph-as-policy`.

Update the revision table above whenever one of these revisions changes.
