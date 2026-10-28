# VABAR: A Variational Automation Benchmark for Agentic Robotics

<!-- TODO(authors): add affiliations and confirm the meaning of the asterisk before release. -->
Shuangyu Xie<sup>*</sup>, Kaiyuan Chen, Uksang Yoo, Xinyue Zhu, Meadow Shen, Jaimyn Drake, Eugen Solowjow,
Lionel Gulich, Alexander Millane, Rafael Cathomen, Tuur Stuyck, Sangeeta Subramanian, Yichao Pan,
Bruce McGowan, Michael Benedict, Linxi Fan, Yuke Zhu, Spencer Huang, Ken Goldberg

[Project page](https://vabar-benchmark.github.io/) | [RoboSimStudio](https://github.com/robosimstudio/robosimstudio) | [Skill library](https://github.com/graph-robots/open-robot-skills)

<p align="center">
  <img alt="The six VABAR task classes: tool sorting, tool hanging, gear meshing, syringe packing, cable insertion and cable routing" src="media/teaser.jpg" width="100%" />
</p>

VABAR is a benchmark for robots that repeat one job across a known distribution of
variation. It has six manipulation task classes taken from industrial, medical and
infrastructure applications. Each class has two difficulty levels and a fixed set of
seeded instances. VABAR runs on the [Newton](https://github.com/newton-physics/newton)
physics engine through [RoboSimStudio](https://github.com/robosimstudio/robosimstudio).

| Task classes | Difficulty levels | Asset library | Skill library | Instances per cell | Embodiments |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 6 | 2 | 55+ | 33 | N = 30 | Franka (single arm), YAM (bimanual) |

## Abstract

Robots in commercial and industrial settings must keep doing specific tasks while the
shape and pose of the objects vary in known ways. We call this setting **Variational
Automation (VA)**, to distinguish it from fixed automation. Recent work on agentic
robotics combines programs written by pretrained LLM and VLM coding agents, which are
easy to inspect, with model-free policies, which adapt to open-world conditions.

We propose design criteria for agentic-robotics benchmarks:
- accessibility;
- repeatability;
- diversity;
- relevance to agentic robotics and to practical applications;
- physical realizability;
- resistance to benchmark saturation;
- computational efficiency.

We apply these criteria to six manipulation task classes: tool sorting, tool hanging,
gear meshing, syringe packing, cable insertion and cable routing.

VABAR, the **Variational Automation Benchmark for Agentic Robotics**, is open,
simulation-based and easy to replicate. It is built entirely on open-source tools. It
provides:
- task implementations on the Newton physics engine;
- standardized task variations;
- difficulty tiers;
- success metrics;
- a library of 33 skills that generalize across tasks, with programmatic reference
  solutions written by people.

We report initial baselines from an open agentic manipulation harness, Graph-as-Policy
(GaP), and from a second implementation on Isaac Lab Arena.

### What a VA task is

A VA task class is a tuple $T = (E, R, A, O, p_O, S)$:

| Symbol | Meaning |
| --- | --- |
| $E$ | environment coordinate frames |
| $R$ | robot and gripper |
| $A$ | admissible assets, each with physical properties and a collision mesh |
| $O$ | space of admissible object configurations |
| $p_O$ | distribution over that space |
| $S$ | measurable success condition |

An instance $\tau_i = \lbrace (a_j, x_j) \rbrace \in O$ pairs each asset with its
initial pose in $SE(3)$, and is drawn $\tau_i \sim p_O$ under a controlled seed.

Admissible variation is constrained in five ways:
- **task invariance:** every instance is still the same task;
- **physical feasibility:** objects are supported and not interpenetrating;
- **robot feasibility:** everything is within the robot's reach;
- **controlled difficulty:** levels widen the admissible region;
- **skill preservation:** every instance at a level can be solved with the same skills.

<p align="center"><img alt="The VABAR evaluation framework" src="media/framework.jpg" width="90%" /></p>

## The six task classes

| | Task class | Setting | Robot | Level 1 | Level 2 |
| --- | --- | --- | --- | --- | --- |
| <img src="media/poster-task-tool-sorting.jpg" width="160" /> | **1 · Tool Sorting** (`tool_sort`) | workshop | Franka | Singulated tools, no distractors | Cluttered tools that overlap and occlude |
| <img src="media/poster-task-tool-hanging.jpg" width="160" /> | **2 · Tool Hanging** (`tool_hanging`) | workshop | Franka | One tool, known fixture pose | Multiple tools and fixture types, randomized poses |
| <img src="media/poster-task-gear-meshing.jpg" width="160" /> | **3 · Gear Meshing** (`gear_mesh`) | assembly | Franka | Assemble two gears | Assemble three gears |
| <img src="media/poster-task-syringe-packing.jpg" width="160" /> | **4 · Syringe Packing** (`syringe_packing`) | healthcare | Franka | Singulated targets | Cluttered scenes, diverse syringe types |
| <img src="media/poster-task-cable-insertion.jpg" width="160" /> | **5 · Cable Insertion** (`usbc_insert`) | data centre | YAM, bimanual | Fixed, visible port; connector nearby with small misalignment | Randomized port pose; moderate connector misalignment |
| <img src="media/poster-task-cable-routing.jpg" width="160" /> | **6 · Cable Routing** (`cable_route`) | data centre | YAM, bimanual | 2–3 widely spaced pegs, short route | 4–6 moderately spaced pegs, multiple turns |

Each class fixes an operational objective and names the axes its instances vary along:

| Task class | Varies |
| --- | --- |
| Tool sorting | tool types, poses and density |
| Tool hanging | tool type, number of tools, tool pose, holder pose |
| Gear meshing | number of gears, gear size, initial poses |
| Syringe packing | number of objects, disposal box type |
| Cable insertion | connector pose, port pose |
| Cable routing | peg count, position and spacing; route; cable length, slack and curvature; board pose |

<p align="center"><img alt="Level 1 and Level 2 on tool sorting: singulated tools, then cluttered ones" src="media/difficulty-levels.jpg" width="90%" /></p>

## Installation

VABAR needs Linux, an NVIDIA GPU, Python 3.12 and [uv](https://docs.astral.sh/uv/).
RoboSimStudio, Newton, the GaP runtime and cuRobo are installed from Git at pinned
revisions (see `pyproject.toml`).

```bash
git clone https://github.com/BerkeleyAutomation/VABAR-RoboSimStudio.git
cd VABAR-RoboSimStudio
uv sync
uv run python -m vabar list            # 371 variants across the six suites
```

The environment is about 12 GB. `UV_NO_GROUP=rtx uv sync` leaves out the path-traced
renderer on a GPU without RT cores.

That is enough for the tasks, the scripted groundtruth and the Studio. The GaP graphs
also need the skill library. Its bundles resolve the graph runtime from a sibling
checkout, so all three repositories sit side by side:

```bash
cd ..   # the directory holding VABAR-RoboSimStudio/
git clone https://github.com/graph-robots/open-robot-skills.git
git -C open-robot-skills checkout 752dfbd96968d4d08379f8111765efbf6457800b
git clone https://github.com/KeplerC/graph-as-policy.git
git -C graph-as-policy checkout 09e45f8de27bc1a29629382ed7c2b9b4b384b3d6
(cd open-robot-skills/tools/curobo && CUDA_HOME=/usr/local/cuda uv sync)   # cuRobo tool bundle, once
```

The perception graphs use SAM 3, GroundingDINO and a VLM. The models are downloaded
the first time a bundle uses them. SAM 3 is a gated Hugging Face model, so set
`HF_TOKEN` first. The VLM is configured with `GAP_VLM_*`. See
[`docs/running_graphs.md`](docs/running_graphs.md).

## Quick start

```bash
uv run python -m vabar list                                                  # every suite/variant
uv run python -m vabar run --task tool_hanging/wrenches_level1 --solve routine --record
uv run python -m vabar run --task tool_hanging/scissors_level1 --solve graph --graph gt --bus-timeout 180
uv run python -m vabar run --task tool_hanging/wrenches_level1 --episodes 1 --solve graph --graph gap_perception \
    --motion curobo --no-privileged --perception vision --bus-timeout 180 --max-steps 8500
uv run python -m vabar studio --task gear_mesh/gearmesh-easy-pair-01          # the interactive Studio
```

`vabar run` is RoboSimStudio's own headless runner (`scripts/bench_run.py`) with the
VABAR suites registered. Its main options:
- `--solve`: the policy, which is one of
  - `routine`, the scripted groundtruth;
  - `graph`, a GaP computation graph the task ships, chosen with `--graph`;
  - `idle`;
  - a dotted path to your own solution class.
- `--episodes`: one instance or several, e.g. `0:30` or `3,7,11`.
- `--workers`: run episodes in parallel processes.
- `--record`: keep video and state.

`python -m vabar run --help` lists everything.

### Interactive Studio

```bash
uv run python -m vabar studio --task tool_sort/level1_1 --port 8080
```

This opens the [RoboSimStudio Studio](https://github.com/robosimstudio/robosimstudio/tree/main/examples/05_studio)
on a VABAR task. Open the printed URL, forwarding the port from a remote machine. The
Studio has these panels:
- **Task**: lists all six suites and their variants; picking one loads it.
- **Goal**: scores the scene clause by clause as you drive it.
- **Policy**: runs the scripted `routine` or any graph the task ships.

You can also drive the arms by hand, record a take and export video.

## The benchmark

The benchmark has twelve cells: six task classes × two levels, with 30 seeded
instances each. They are listed in [`benchmark/vabar_v1.json`](benchmark/vabar_v1.json)
and generated from the frozen instance documents under [`benchmark/`](benchmark/).
Each instance is a `(task, episode)` pair on the registered suites, so running a cell
is an ordinary run over an episode list:

```bash
uv run python -m vabar bench --list                                   # the cells
uv run python -m vabar bench --cell tool_hanging/L1 --solve graph     # the cell's reference graph, 30 instances
uv run python -m vabar bench --cell cable_insertion/L2 --print        # the vabar run commands it issues
uv run python -m vabar bench --cell tool_hanging/L1 --verify          # resets match the frozen transforms
```

**How success is counted.**
- **Independent attempts:** in tasks built from attempts that are independent by
  construction (sorting or hanging separately sampled objects), each object-level
  attempt is a trial. Success rate is the fraction of objects manipulated successfully.
- **Sequential, state-dependent tasks:** in tasks like gear meshing, the whole instance
  is the unit, and subgoal success is reported as a diagnostic.
- **Cycle time:** simulated time from policy start to success, failure or timeout. The
  tables report the median over successful trials.

### Baseline results (from the paper)

Success rate (%) and median cycle time (simulated seconds) over N = 30 instances per
cell (GaP Human–ILA: N = 40, on its own instances). A dash means no successful trial.

| Task | Level | GaP (Human–Newton) | GaP (Human–ILA) | Claude Code | GaP |
| --- | :---: | ---: | ---: | ---: | ---: |
| Tool Sorting | L1 | 47.1 · 421.6 | 20.0 · 555.0 | 0.0 · — | 26.7 · 134.2 |
| | L2 | 33.7 · 913.8 | — | 0.0 · — | 13.3 · 232.3 |
| Tool Hanging | L1 | 86.7 · 86.1 | 67.5 · 98.3 | 16.7 · 96.2 | 33.3 · 81.5 |
| | L2 | 81.7 · 158.4 | 50.0 · 138.3 | 10.0 · 190.8 | 10.0 · 270.9 |
| Gear Meshing | L1 | 100.0 · 165.9 | 87.5 · 198.9 | 0.0 · — | 70.0 · 156.6 |
| | L2 | 90.0 · 235.5 | 80.0 · 381.3 | 0.0 · — | 60.0 · 284.1 |
| Syringe Packing | L1 | 87.0 · 228.8 | 52.5 · 939.0 | 0.0 · — | 33.3 · 371.1 |
| | L2 | 63.0 · 472.3 | 40.0 · 1810.0 | 0.0 · — | 12.1 · 652.2 |
| Cable Insertion | L1 | 86.7 · 78.8 | 97.5 · 104.9 | 0.0 · — | 36.7 · 79.0 |
| | L2 | 30.0 · 74.2 | 17.5 · 118.8 | 0.0 · — | 23.3 · 80.5 |
| Cable Routing | L1 | 46.7 · 116.0 | 50.0 · 145.9 | 3.3 · 120.7 | 36.7 · 50.8 |
| | L2 | 46.7 · 189.8 | 5.0 · 151.5 | 0.0 · — | 13.3 · 120.3 |

The four systems:
- **GaP (Human):** a computation graph specified by hand over the skill library. This
  repository ships it as each task's reference graph.
- **Claude Code:** a general-purpose coding agent (Claude Opus 5) with no robotics
  harness or skill library.
- **GaP:** the graph generated autonomously by the same model, over the same skill library.

These are the paper's numbers. [`docs/VALIDATION.md`](docs/VALIDATION.md) records what
this release reproduces on the pinned RoboSimStudio. The cable cells in particular do
not reproduce under the shipped protocol (palette segmentation): the reference graphs
also fail on USB-C Easy and Routing Easy under the cable suites' original runtime, so
the configuration behind those rates is still being confirmed. See
[`docs/validation/cable.md`](docs/validation/cable.md).

## Skill library

<p align="center"><img alt="The VABAR skill library: 33 skills across perception, grasping, motion and verification" src="media/skills.jpg" width="90%" /></p>

VABAR extends the [Modular Open Robot Skill Library](https://github.com/graph-robots/open-robot-skills)
with reusable procedures for the following:
- functional-feature manipulation;
- constrained placement;
- deformable cable handling, through metric feature localization, source–destination
  correspondence, in-hand registration and staged feature mating.

The skill bundles live in two places:
- **open-robot-skills:** all 33 skills, in a checkout beside this one. Override the
  location with `VABAR_OPEN_ROBOT_SKILLS` or `GAP_SKILLS_PATH`.
- **[`robot_skills/`](robot_skills/):** the 20 cable bundles, shipped in this repository.

A task's reference graph is a `workflow.json` plus node scripts under
`tasks/<suite>/graphs/<label>/`, composed from those skills. See
[`docs/running_graphs.md`](docs/running_graphs.md).

## Repository layout

```
tasks/<suite>/          one directory per task class, in RoboSimStudio's document format:
  <suite>.suite           what every variant shares: cell, robot, physics, cameras
  <variant>.task          one document per variant: objects, fixtures, goal, policies
  staging.py scenes.py    the arrangements, camera rigs and cell the documents name
  assets/                 asset records, entity code, meshes and collision hulls
  routine.py tools.py     the scripted groundtruth and the suite's own graph tools
  graphs/<label>/         reference GaP graphs (workflow.json + scripts/)
tasks/_shared/scripts/  graph node scripts several suites share
assets/                 the tool-sorting workcell export and robot models the suites use
benchmark/              frozen instance documents and manifests; vabar_v1.json, the twelve cells
robot_skills/           the cable-task GaP skill bundles
vabar/                  what RoboSimStudio does not ship: the routine hook, the CLI, the benchmark runner
docs/                   running graphs, cable tasks, validation records
```

## How VABAR uses RoboSimStudio

VABAR is a task collection for RoboSimStudio, not a fork of it. The pinned upstream
release is installed unmodified. VABAR attaches to it in three ways:
- **The suites are an external task root.** `vabar.install()` calls RoboSimStudio's
  `add_task_root("tasks/")`, and the `assets/` beside that root is searched
  automatically. Any RoboSimStudio entry point therefore sees them:
  `python scripts/bench_run.py --task-root tasks` from a RoboSimStudio checkout works too.
- **Scripted groundtruth.** The public RoboSimStudio release leaves out its
  scripted-routine module. [`vabar/routines.py`](vabar/routines.py) provides it, and each
  suite's `staging.py` imports `vabar`, so the `(:routine …)` the documents declare
  resolve in every process that scans the suites: CLI, Studio and pool workers.
- **Everything else** a suite needs beyond upstream lives in that suite's directory or
  in `vabar/`. That includes embodiments, goal measures and cable perception tools.

## Asset licences

Code is Apache-2.0. Robot models keep their own licences (see [NOTICE](NOTICE)).
[ASSET_PROVENANCE.md](ASSET_PROVENANCE.md) lists every object mesh and fixture with its
source, and flags the ones whose redistribution licence is still being confirmed.

## Citation

```bibtex
@inproceedings{xie2027vabar,
  title     = {{VABAR}: A Variational Automation Benchmark for Agentic Robotics},
  author    = {Xie, Shuangyu and Chen, Kaiyuan and Yoo, Uksang and Zhu, Xinyue and Shen, Meadow and
               Drake, Jaimyn and Solowjow, Eugen and Gulich, Lionel and Millane, Alexander and
               Cathomen, Rafael and Stuyck, Tuur and Subramanian, Sangeeta and Pan, Yichao and
               McGowan, Bruce and Benedict, Michael and Fan, Linxi and Zhu, Yuke and Huang, Spencer and
               Goldberg, Ken},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2027},
  note      = {Under review}
}
```

## Acknowledgements

VABAR is built on:
- [RoboSimStudio](https://github.com/robosimstudio/robosimstudio)
- [Newton](https://github.com/newton-physics/newton)
- [Warp](https://github.com/NVIDIA/warp)
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
- [Graph-as-Policy](https://github.com/lgulich/graph-as-policy)
- [open-robot-skills](https://github.com/graph-robots/open-robot-skills)
- [cuRobo](https://github.com/NVlabs/curobo)
- [PyRoKi](https://github.com/chungmin99/pyroki)
- [viser](https://github.com/viser-project/viser)
- [Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena)

Robot models come from MuJoCo Menagerie, ROS-Industrial and [i2rt](https://github.com/i2rt-robotics/i2rt).

## License

Apache License 2.0; see [LICENSE](LICENSE).
