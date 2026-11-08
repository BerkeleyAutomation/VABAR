# VABAR on Isaac Lab Arena

This folder runs the VABAR tasks and AUTOLab's reference GaP (Graph as Policy) graphs in
[Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena) on Isaac Sim. The RSS version
(RoboSimStudio, at the root of this repository) defines the tasks and graphs; this folder rebuilds
the scenes in Arena and runs those graphs unmodified through the RSS adapter. Arena's episode JSONL
is the score: a graph that finishes is not a success unless Arena's task check passes.

| Path | Purpose |
|---|---|
| `isaaclab_arena_gap/` | RSS adapter, Arena policy bridge, Experiment YAMLs, launchers, `vabar-arena` CLI |
| `isaaclab_arena_vabar/` | Arena scenes, tasks, assets, robots (embodiments), and scoring |
| `tools/` | Benchmark-authoring and conversion tools |
| `external/` | Pinned sources: Isaac Lab Arena, graph-as-policy, open-robot-skills |
| `docs/` | Setup, design, and evidence records |

## Before you start

- **Linux**, validated on Ubuntu 24.04.
- **An NVIDIA GPU with 24 GB of memory**, validated on an RTX 4090. Driver and CUDA versions are in
  [docs/SETUP.md section 2](docs/SETUP.md#2-validated-reference-machine).
- **`git` and Git LFS** (`git lfs install` run once). Some scene assets are stored in LFS.
- **`uv` 0.9.x**, first on `PATH`. `./setup.sh` refuses other series; `uv self update 0.9.25`
  pins a standalone install.
- **CMake older than 4 and a C++ compiler** (`build-essential`). Setup builds `egl-probe` from
  source, which CMake 4 rejects; `uv tool install 'cmake<4'` provides one.
- **About 100 GB of free disk.** The Python environments take about 75 GB (the Isaac Sim
  environment alone is about 40 GB), the pinned sources about 9 GB, and the downloaded models
  about 4 GB. uv's package cache comes on top unless it is on the same filesystem.
- **Read access to the VABAR repository** (branch `vabar-robosimstudio`). The other pinned sources
  are public.
- **Credentials**, needed before `./setup.sh` because its final check looks for them:
  - a VLM API key, as `GAP_VLM_API_KEY`;
  - a Hugging Face token, as `HF_TOKEN`, for an account that has been granted access to the
    gated [`facebook/sam3`](https://huggingface.co/facebook/sam3) model.

  Put both in a private file outside the repository; the launchers and the setup check read it:

  ```bash
  mkdir -p "$HOME/.config/gap"
  cat > "$HOME/.config/gap/vlm.env" <<'EOF'
  export GAP_VLM_API_KEY=your-key
  export HF_TOKEN=your-hugging-face-token
  EOF
  chmod 600 "$HOME/.config/gap/vlm.env"
  ```

  The VLM route defaults to the NVIDIA-hosted OpenAI-compatible endpoint and model set in
  `isaaclab_arena_gap/scripts/vlm_env.sh`. If your key is for another provider, also set
  `GAP_VLM_PROVIDER`, `GAP_VLM_BASE_URL` and `GAP_VLM_MODEL` in that file; see
  [docs/SETUP.md section 5](docs/SETUP.md#5-configure-credentials).

## Quickstart

Clone without `--recurse-submodules`; setup initializes only the submodules it needs.

```bash
git clone --branch vabar-robosimstudio \
  git@github.com:BerkeleyAutomation/AUTOLab-GaP-NVIDIA-Collaboration.git
cd AUTOLab-GaP-NVIDIA-Collaboration/isaaclab_arena
git lfs pull
./setup.sh
```

`./setup.sh` checks out the pinned sources and builds the Isaac Sim environment (`.venv`), the RSS
version's environment (`.venv-autolab`), and the GaP skill environments. It ends with a preflight
check of the environments, credentials and GPU. The first run downloads tens of gigabytes; rerunning
it is safe.

Then run one episode of easy gear insertion with AUTOLab's `gear_mesh/gap_perception` graph:

```bash
uv run vabar-arena run \
  isaaclab_arena_gap/experiment_configs/gear_insertion/vabar_gear_insertion_easy_gap_baseline.yaml \
  --skip-setup
```

Without `--skip-setup`, `run` first reruns `./setup.sh`, preflight included, to make sure the
environments are current. Once setup has succeeded you can skip that. Add `--no-headless` to watch
in the Isaac Sim viewer. The first episode is slow while models load and shaders compile; an easy
gear episode is about 3000 simulation steps.

The launcher prints its output directory, `outputs/arena/<timestamp>/`. The score is the episode
JSONL in the run's subfolder; a successful episode has `"success": true`. Camera videos sit next to
it, and the GaP graph trace is under `gap-trace/`:

```bash
find outputs/arena -name 'episode_results*.jsonl' -o -name '*.mp4'
```

Every supported task has a YAML under `isaaclab_arena_gap/experiment_configs/<task family>/`;
[docs/SETUP.md section 8](docs/SETUP.md#8-run-the-supported-pairs) walks through them.

## Inspect an environment

```bash
uv run vabar-arena show --env vabar_tool_sorting__easy_1 --skip-setup
```

`show` builds the environment in the Isaac Sim viewer without a policy and holds the robot still
until you stop it.

## Evaluate your own policy

The environments do not depend on GaP. Any Arena policy can run them, and Arena scores it the same
way, in the episode JSONL. Copy a baseline YAML and replace its whole `policy:` block: set `type` to
a registered Arena policy name or the dotted import path of your own policy class, and drop the GaP
policy's keys (`gap_*`, `robot_profile`, `auto_spawn`, ...), which Arena rejects for any other
policy. For example, Arena's built-in `zero_action` policy on easy gear insertion:

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

[docs/SETUP.md, "Evaluate your own policy"](docs/SETUP.md#evaluate-your-own-policy) explains how to
make your own policy class importable and where each robot's observations and actions are defined.

## Test

The non-GPU tests and the setup preflight are in
[docs/SETUP.md section 7](docs/SETUP.md#7-run-the-non-gpu-tests-and-preflight). Simulator tests are
opt-in: set `ISAAC_CAP_RUN_LIVE_TESTS=1` and pass `-m live`.

## Further reading

- [docs/SETUP.md](docs/SETUP.md): full setup, every runnable task, outputs, and common failures.
- [docs/DESIGN.md](docs/DESIGN.md): architecture and which component owns what.
- [isaaclab_arena_vabar/README.md](isaaclab_arena_vabar/README.md): the benchmark scenes, tasks and
  assets.
- [isaaclab_arena_gap/README.md](isaaclab_arena_gap/README.md): the RSS adapter and the Experiment
  YAMLs.
