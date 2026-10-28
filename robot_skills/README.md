# Robot skills and policy graphs

This package contains 20 native GaP skill bundles and the USB-C insertion and
cable-routing policy graphs. Each bundle includes input and exit contracts,
declared tool capabilities, executable Python recipes, and subgraph examples.

| Graph | Task variants expected in the simulator |
| --- | --- |
| [USB-C insertion](../tasks/usbc_insert/graphs/mate) | `usbc_insert/insert_easy`, `usbc_insert/insert_medium` |
| [Cable routing](../tasks/cable_route/graphs/weave) | `cable_route/port_easy`, `cable_route/port_medium` |

## Run from this checkout

`vabar` adds this directory to GaP's skill registries (see
[`docs/running_graphs.md`](../docs/running_graphs.md)). For an individual episode
of a task variant, from the repository root:

```bash
ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette uv run python -m vabar run \
  --task usbc_insert/insert_medium --solve graph --graph mate \
  --perception vision --no-privileged --renderer rtx --bus-timeout 600 \
  --settle-steps 600 --episodes 0:1

ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette uv run python -m vabar run \
  --task cable_route/port_medium --solve graph --graph gt \
  --perception vision --no-privileged --renderer rtx --bus-timeout 600 \
  --settle-steps 60 --episodes 0:1
```

Use `insert_easy` or `port_easy` for Easy. The frozen files are episodes of these
variants; `python -m vabar bench --cell cable_insertion/L1` runs the 30 selected
ones. For routing, the `gt` artifact label selects `graphs/weave`, and
`--perception vision --no-privileged` selects camera-based execution.

The recipes call the RoboSimStudio connector's `usbc.*`, `cable.*`, and `curve.*`
tools (registered by the suites' `tools.py`, with `curve.*` from
`vabar/cable/`). When using these bundles in another environment, supply a compatible
connector and add `robot_skills` to its GaP registries. An explicit `skills=`,
`--skills`, or `GAP_SKILLS_PATH` overrides normal registry discovery.

## Contracts and maintenance

[catalog.json](catalog.json) records the strategies, graph ownership, and exit
meanings. Task graphs own the executable code; skill bundles contain their
exported recipes and local helpers. Keep the workflow contracts and matching
skill recipes synchronized when changing either. Generation and validation tools
belong to the authoring environment and are not needed to execute the exported
recipes.

Graph completion, visual verification, and settled physical success are distinct.
For example, a cable `done` exit means a commanded motion completed; USB-C
`inserted` is the selected perception body's verdict. The independent task
evaluator checks the physical goal after release and settling.

The vision path uses RGB-D and robot proprioception with privileged tools
withheld. Contact friction carries the grasped objects. Oracle recipes are
explicitly selected diagnostic alternatives. Each SKILL.md describes its inputs,
exits, and required capabilities.
