# VABAR: A Variational Automation Benchmark for Agentic Robotics

<!-- TODO(authors): add affiliations and confirm the meaning of the asterisk before release. -->
Shuangyu Xie<sup>*</sup>, Kaiyuan Chen, Uksang Yoo, Xinyue Zhu, Meadow Shen, Jaimyn Drake, Eugen Solowjow,
Lionel Gulich, Alexander Millane, Rafael Cathomen, Tuur Stuyck, Sangeeta Subramanian, Yichao Pan,
Bruce McGowan, Michael Benedict, Linxi Fan, Yuke Zhu, Spencer Huang, Ken Goldberg

[Project page](https://vabar-benchmark.github.io/) | [RoboSimStudio](https://github.com/robosimstudio/robosimstudio) | [Skill library](https://github.com/graph-robots/open-robot-skills)

<p align="center">
  <img alt="The six VABAR task classes: tool sorting, tool hanging, gear meshing, syringe packing, cable insertion and cable routing" src="media/teaser.jpg" width="100%" />
</p>

VABAR is a benchmark for robots that repeat one job across a known distribution of variation:
six manipulation task classes from industrial, medical and infrastructure settings, each with two
difficulty levels and 30 seeded instances per level. It runs on the [Newton](https://github.com/newton-physics/newton)
physics engine through [RoboSimStudio](https://github.com/robosimstudio/robosimstudio), and every task
ships a scripted solution and reference [Graph-as-Policy](https://github.com/graph-robots/graph-as-policy) (GaP) graphs.

## Quick start

Needs Linux, an NVIDIA GPU, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/BerkeleyAutomation/VABAR.git
cd VABAR
uv sync                                  # ~12 GB; on a GPU without RT cores: UV_NO_GROUP=rtx uv sync
uv run python -m vabar list              # 374 variants across the six suites

uv run python -m vabar run --task tool_hanging/wrenches_level1 --solve routine --record   # scripted solution, with video
uv run python -m vabar studio --task gear_mesh/gearmesh-easy-pair-01                      # interactive Studio: open the printed URL
```

The [Studio](https://github.com/robosimstudio/robosimstudio/tree/main/examples/05_studio) lists every
task, scores the goal clause by clause as you drive the arms by hand, runs a task's scripted routine
or any of its graphs, and records takes as video. On a remote machine, pick a port with `--port 8080`
and forward it.

### Running the graphs

The GaP graphs also need the skill library and the graph runtime, checked out beside `VABAR/`:

```bash
cd ..
git clone https://github.com/KeplerC/open-robot-skills.git
git -C open-robot-skills checkout b15859a244658abc6060f3aa3b6827780746a5fe
git clone https://github.com/KeplerC/graph-as-policy.git
git -C graph-as-policy checkout 06dd6193e13eab816123febef1369deda0849d91
(cd open-robot-skills/tools/curobo && CUDA_HOME=/usr/local/cuda uv sync)   # cuRobo tool bundle, once
cd VABAR

uv run python -m vabar run --task tool_hanging/scissors_level1 --solve graph --graph gt --bus-timeout 180
uv run python -m vabar run --task tool_hanging/wrenches_level1 --episodes 1 --solve graph --graph gap_perception \
    --motion curobo --no-privileged --perception vision --bus-timeout 180 --max-steps 8500
```

The perception graphs use SAM 3, GroundingDINO and a VLM, downloaded on first use. SAM 3 is gated on
Hugging Face, so set `HF_TOKEN` first; the VLM is configured with `GAP_VLM_*`. See
[`docs/running_graphs.md`](docs/running_graphs.md).

`vabar run` options:
- `--solve`: `routine` (scripted), `graph` (a graph the task ships, picked with `--graph`), `idle`,
  or a dotted path to your own policy class.
- `--episodes`: `0:30` or `3,7,11`; `--workers N` runs them in parallel.
- `--record`: keep video and state.

`python -m vabar run --help` lists everything.

## The six task classes

| | Task class | Setting | Robot | Level 1 | Level 2 |
| --- | --- | --- | --- | --- | --- |
| <img src="media/poster-task-tool-sorting.jpg" width="160" /> | **1 · Tool Sorting** (`tool_sort`) | workshop | Franka | Singulated tools, no distractors | Cluttered tools that overlap and occlude |
| <img src="media/poster-task-tool-hanging.jpg" width="160" /> | **2 · Tool Hanging** (`tool_hanging`) | workshop | Franka | One tool, known fixture pose | Multiple tools and fixture types, randomized poses |
| <img src="media/poster-task-gear-meshing.jpg" width="160" /> | **3 · Gear Meshing** (`gear_mesh`) | assembly | Franka | Assemble two gears | Assemble three gears |
| <img src="media/poster-task-syringe-packing.jpg" width="160" /> | **4 · Syringe Packing** (`syringe_packing`) | healthcare | Franka | Singulated targets | Cluttered scenes, diverse syringe types |
| <img src="media/poster-task-cable-insertion.jpg" width="160" /> | **5 · Cable Insertion** (`usbc_insert`) | data centre | YAM, bimanual | Fixed, visible port; connector nearby with small misalignment | Randomized port pose; moderate connector misalignment |
| <img src="media/poster-task-cable-routing.jpg" width="160" /> | **6 · Cable Routing** (`cable_route`) | data centre | YAM, bimanual | 2–3 widely spaced pegs, short route | 4–6 moderately spaced pegs, multiple turns |

## Running the benchmark

The benchmark has twelve cells, six task classes × two levels, with 30 frozen instances each
([`benchmark/vabar_v1.json`](benchmark/vabar_v1.json)). A cell runs its reference graph on its instances:

```bash
uv run python -m vabar warmup                                     # once per machine: compiles the renderer (5-8 min)
uv run python -m vabar bench --list                               # the twelve cells
uv run python -m vabar bench --cell tool_hanging/L1               # the reference graph on the cell's 30 instances
uv run python -m vabar bench --cell tool_hanging/L1 --config fast # quicker; the first 15 instances (--limit 30 for all)
uv run python -m vabar bench --cell tool_hanging/L1 --print       # the vabar run commands it would issue
```

| `--config` | |
| --- | --- |
| `default` | One GPU: the simulator waits while the graph thinks, observations render only when the graph asks, tool servers stay up between episodes, and several workers share the GPU. |
| `fast` | `default` plus cheaper physics, where a full cell showed no change in outcome. |
| `paper` | The paper's protocol exactly. |

**How success is counted.**
- Where a task's objects are independent (sorting or hanging separately sampled tools), each
  object is a trial, and the rate is the fraction of objects done. A hung tool counts once it has
  been let go: no hand within the tool's own radius of it, for a second.
- In sequential tasks such as gear meshing the whole instance is the unit; subgoals are reported
  as a diagnostic.
- Cycle time is simulated time from policy start to success, failure or timeout; the tables give
  the median over successful trials.

### Results

Success rate (%) and median cycle time (simulated seconds), N = 30 per cell (GaP Human–ILA: N = 40,
on its own instances); a dash means no successful trial. *This release* re-runs the shipped reference
graph on this repository (`--config fast --limit 30`, one 96 GB GPU, 5–37 min a cell), as the range
over repeated runs.

| Task | Level | GaP (Human–Newton) | This release | GaP (Human–ILA) | Claude Code | GaP |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Tool Sorting | L1 | 47.1 · 421.6 | 72–81 | 20.0 · 555.0 | 0.0 · — | 26.7 · 134.2 |
| | L2 | 33.7 · 913.8 | 57–64 | — | 0.0 · — | 13.3 · 232.3 |
| Tool Hanging | L1 | 86.7 · 86.1 | 98–100 | 67.5 · 98.3 | 16.7 · 96.2 | 33.3 · 81.5 |
| | L2 | 81.7 · 158.4 | 83–93 | 50.0 · 138.3 | 10.0 · 190.8 | 10.0 · 270.9 |
| Gear Meshing | L1 | 100.0 · 165.9 | 98–100 | 87.5 · 198.9 | 0.0 · — | 70.0 · 156.6 |
| | L2 | 90.0 · 235.5 | 89–92 | 80.0 · 381.3 | 0.0 · — | 60.0 · 284.1 |
| Syringe Packing | L1 | 87.0 · 228.8 | 87–90 | 52.5 · 939.0 | 0.0 · — | 33.3 · 371.1 |
| | L2 | 63.0 · 472.3 | 47–63 | 40.0 · 1810.0 | 0.0 · — | 12.1 · 652.2 |
| Cable Insertion | L1 | 86.7 · 78.8 | 100 | 97.5 · 104.9 | 0.0 · — | 36.7 · 79.0 |
| | L2 | 30.0 · 74.2 | 67–80 | 17.5 · 118.8 | 0.0 · — | 23.3 · 80.5 |
| Cable Routing | L1 | 46.7 · 116.0 | 50–77 | 50.0 · 145.9 | 3.3 · 120.7 | 36.7 · 50.8 |
| | L2 | 46.7 · 189.8 | 40–70 | 5.0 · 151.5 | 0.0 · — | 13.3 · 120.3 |

- **GaP (Human):** a graph specified by hand over the skill library; this repository ships it as each
  task's reference graph.
- **Claude Code:** a general-purpose coding agent (Claude Opus 5) with no robotics harness or skill library.
- **GaP:** the graph generated autonomously by the same model, over the same skill library.

The cable cells differ most from the paper: their graphs' perception has since been fixed for the
camera rig the benchmark stages. Per-cell cost and GPU memory, the full config reference, what changed
since the paper and how VABAR sits on RoboSimStudio are in [`docs/benchmark.md`](docs/benchmark.md);
[`docs/VALIDATION.md`](docs/VALIDATION.md) records the port.

## Skill library

<p align="center"><img alt="The VABAR skill library: 33 skills across perception, grasping, motion and verification" src="media/skills.jpg" width="90%" /></p>

VABAR extends the [Modular Open Robot Skill Library](https://github.com/graph-robots/open-robot-skills)
with 33 skills for functional-feature manipulation, constrained placement and deformable cable
handling. They live in the `open-robot-skills` checkout beside this one (override the location with
`VABAR_OPEN_ROBOT_SKILLS` or `GAP_SKILLS_PATH`), plus the 20 cable bundles in
[`robot_skills/`](robot_skills/). A task's reference graph is a `workflow.json` plus node scripts
under `tasks/<suite>/graphs/<label>/`.

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
docs/                   the benchmark in detail, running graphs, validation records
```

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

VABAR is built on [RoboSimStudio](https://github.com/robosimstudio/robosimstudio),
[Newton](https://github.com/newton-physics/newton), [Warp](https://github.com/NVIDIA/warp),
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp),
[Graph-as-Policy](https://github.com/lgulich/graph-as-policy),
[open-robot-skills](https://github.com/graph-robots/open-robot-skills), [cuRobo](https://github.com/NVlabs/curobo),
[PyRoKi](https://github.com/chungmin99/pyroki), [viser](https://github.com/viser-project/viser) and
[Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena). Robot models come from MuJoCo Menagerie,
ROS-Industrial and [i2rt](https://github.com/i2rt-robotics/i2rt).

## License

Code is Apache-2.0 ([LICENSE](LICENSE)). Robot models keep their own licences ([NOTICE](NOTICE)), and
[ASSET_PROVENANCE.md](ASSET_PROVENANCE.md) lists every object mesh and fixture with its source.
