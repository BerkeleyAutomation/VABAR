# Running the reference graphs

A task's `(:graph LABEL "graphs/<dir>")` is a Graph-as-Policy (GaP) workflow: a
`workflow.json` plus node scripts. `--solve graph --graph LABEL` runs it through
the graph runtime. The scripted `routine` needs nothing beyond `uv sync`. A
graph also needs the skill library and, for the perception graphs, models and a VLM.

## Workspace

The skill bundles live in [open-robot-skills](https://github.com/graph-robots/open-robot-skills).
Some of them run out of process from their own virtual environments, and those
environments resolve `graph-as-policy-core` from a sibling checkout. So all three
repositories sit side by side:

```
workspace/
  VABAR-RoboSimStudio/        this repository
  open-robot-skills/          graph-robots/open-robot-skills @ main (752dfbd)
  graph-as-policy/            KeplerC/graph-as-policy @ 09e45f8 (the rev pyproject.toml pins)
```

```bash
cd workspace
git clone https://github.com/graph-robots/open-robot-skills.git
git -C open-robot-skills checkout 752dfbd96968d4d08379f8111765efbf6457800b
git clone https://github.com/KeplerC/graph-as-policy.git
git -C graph-as-policy checkout 09e45f8de27bc1a29629382ed7c2b9b4b384b3d6
# The cuRobo tool bundle compiles CUDA kernels against its own torch, once:
(cd open-robot-skills/tools/curobo && CUDA_HOME=/usr/local/cuda uv sync)
```

`vabar` registers the skill registries itself:
1. `robot_skills/`, the cable bundles shipped here;
2. `../open-robot-skills`, or `$VABAR_OPEN_ROBOT_SKILLS`.

Setting `$GAP_SKILLS_PATH` overrides both. Three bundle names exist in both
registries; the copy in `robot_skills/` is the one used, and gap logs a warning
naming the shadowed copy:
- `executing-held-object-motion`
- `perceiving-deformable-linear-objects`
- `perceiving-routing-fixtures`

## Models and credentials

- **SAM 3** (`facebook/sam3`): a gated Hugging Face model, used by every perception
  graph. Accept its licence on the model page, then `export HF_TOKEN=hf_...`.
  Without it, the first perception node fails with `GatedRepoError: 401`. Machines
  that already have the weights cached don't need the token.
- **GroundingDINO and the other bundle models**: downloaded on first use into the
  bundles' environments.
- **A VLM**:
  - Who calls one: tool sorting reads the bin labels with it, and some perception
    skills use one for grounding.
  - OpenAI-compatible endpoint:

    ```bash
    export GAP_VLM_PROVIDER=openrouter
    export GAP_VLM_BASE_URL=https://api.openai.com/v1
    export GAP_VLM_API_KEY="$OPENAI_API_KEY"
    export GAP_VLM_MODEL=gpt-5.6
    ```

  - Vertex: set `GAP_VLM_PROVIDER`, `GAP_VLM_PROJECT_ID`, `GAP_VLM_REGION` and
    `GAP_VLM_MODEL`, with application-default credentials.

## The evaluated recipes

`python -m vabar bench --cell <cell> --print` prints the exact command for each
cell. The reference graph and its flags are recorded per cell in
`benchmark/vabar_v1.json`:

| Cell | Graph | Flags |
| --- | --- | --- |
| `tool_sorting/L1` | `gap_perception_loop` | `--no-privileged --perception vision --bus-timeout 180 --max-steps 20000` |
| `tool_hanging/L1` | `gap_perception` | `--motion curobo --no-privileged --perception vision --bus-timeout 180 --max-steps 8500` |
| `tool_hanging/L2` | `gap_perception` | same, with `--max-steps 22000` |
| `gear_meshing/L1`, `L2` | `gap_perception` | `--motion curobo --no-privileged --perception vision --bus-timeout 180` |
| `syringe_packing/L1`, `L2` | `gap_perception` | `--motion curobo --no-privileged --perception vision --bus-timeout 180` |
| `cable_insertion/L1`, `L2` | `mate` | `--no-privileged --perception vision --renderer rtx --bus-timeout 600 --settle-steps 600` |
| `cable_routing/L1`, `L2` | `gt` (`graphs/weave`) | `--no-privileged --perception vision --renderer rtx --bus-timeout 600 --settle-steps 60` |

The cable cells also set `ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette`; `vabar bench`
does this for them and `--print` shows it. Without it the runtime picks the gated SAM 3
segmenter whenever open-robot-skills is on the skill path.

For example:

```bash
uv run python -m vabar run --task tool_hanging/wrenches_level1 --episodes 1 \
  --solve graph --graph gap_perception --motion curobo --no-privileged --perception vision \
  --bus-timeout 180 --max-steps 8500 --record --record-from-motion --out outputs/wrenches_level1
```

The privileged `gt` graphs that most suites also ship read the simulator's own
answers (`sim.*` tools). They are useful for debugging a task, but they are not
the evaluated policy.

## Flags worth knowing

| Flag | Why |
| --- | --- |
| `--bus-timeout 180` | The watchdog on a graph's tool request is wall-clock time. A cuRobo plan can take tens of seconds; the default 30 s kills it mid-solve and the trace reads as a failed plan. |
| `--no-privileged` | Withholds the `sim.*` ground-truth tools. The perception graphs call none of them. |
| `--perception vision` | Takes the RGB-D + detector branch of a graph's perception router. Without a renderer named, `vabar run` adds `--renderer auto` so the cameras exist. |
| `--renderer rtx` | The cable cells' renderer. Their perception thresholds were tuned on RTX images. |
| `--motion curobo` | Collision-aware planning for `robot.go_to_pose`. It needs the cuRobo tool bundle above. |
| `--max-steps N` | Raises the episode budget above the document's own. |
| `--trace-dir DIR` | Keeps `dag_trace.json` and `node_data/` per episode. This is where a failed run is diagnosed. |
| `--record --record-from-motion` | Video and state, starting when the arm first moves. |

## Checking a graph without running it

```bash
GAP_SKILLS_PATH=robot_skills:../open-robot-skills uv run gap run tasks/tool_hanging/graphs/gap_perception --validate-only
```

Every shipped graph validates clean against the pinned runtime and the skill
library above. Some subgraphs run the graph's own scripts for a skill rather
than that bundle's canonical scripts. Those are labelled `generic`, as
graph-as-policy's skill contract (SK1/SK2) requires, and the skill each one
composes is recorded in the workflow's `meta.composes`.
