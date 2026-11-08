# VABAR on Isaac Lab Arena

This folder runs the VABAR tasks and the reference GaP (Graph as Policy) graphs of the RSS version in
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

From the root of the VABAR repository:

```bash
cd isaaclab_arena
git lfs pull
./setup.sh
```

`./setup.sh` checks out the pinned sources and builds the Isaac Sim environment (`.venv`), the RSS
version's environment (`.venv-autolab`), and the GaP skill environments. It ends with a preflight
check of the environments, credentials and GPU. The first run downloads tens of gigabytes; rerunning
it is safe.

Then run one episode of easy gear insertion with the RSS graph `gear_mesh/gap_perception`:

```bash
uv run vabar-arena run \
  isaaclab_arena_gap/experiment_configs/gear_insertion/vabar_gear_insertion_easy_gap_baseline.yaml \
  --skip-setup
```

Without `--skip-setup`, `run` first reruns `./setup.sh`, preflight included, to make sure the
environments are current. Once setup has succeeded you can skip that. Add `--no-headless` to watch
in the Isaac Sim viewer. The first episode is slow while models load and shaders compile; an easy
gear episode is about 3000 simulation steps.

The full list of tasks and their experiment configs is in
[docs/SETUP.md section 8](docs/SETUP.md#8-run-the-supported-pairs).

## Inspect an environment

```bash
uv run vabar-arena show --env vabar_tool_sorting__easy_1 --skip-setup
```

`show` builds the environment in the Isaac Sim viewer without a policy and holds the robot still
until you stop it.

## Run VABAR with your own policy

You can also run your own policy on VABAR. The tasks are regular Isaac Lab Arena environments, so
any Arena policy can run them and is scored the same way; see Arena's
[policy guide](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/pages/concepts/policy/index.html)
and [experiment guide](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/pages/concepts/concept_arena_experiments.html)
for how to write a policy and point an experiment at it.

## Further reading

- [docs/SETUP.md](docs/SETUP.md): full setup, every runnable task, outputs, and common failures.
- [docs/DESIGN.md](docs/DESIGN.md): architecture and which component owns what.
- [isaaclab_arena_vabar/README.md](isaaclab_arena_vabar/README.md): the benchmark scenes, tasks and
  assets.
- [isaaclab_arena_gap/README.md](isaaclab_arena_gap/README.md): the RSS adapter and the Experiment
  YAMLs.
