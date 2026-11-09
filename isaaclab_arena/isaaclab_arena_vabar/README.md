# Industrial Benchmark

`isaaclab_arena_vabar` provides policy-independent Arena benchmark content. It owns physical
assets, robot embodiments, custom task implementations, complete environment graphs, and scoring
parameters. It has no dependency on `isaaclab_arena_gap`.

## Package concepts

- `assets/` is the flat catalog of physical USD dependency closures. It includes the FR3 with
  Robotiq 2F-85 robot asset and generated VAB objects and backgrounds.
- `embodiments/` defines how Arena controls and observes a robot. The `vabar__fr3_robotiq_2f85`
  and `vabar__bimanual_yam` evaluation embodiments reference their robot assets from `assets/`.
- `tasks/` contains reusable package-owned task implementations. The VAB tool-sort environments use
  the package-owned `ObjectsInRegionsTask`, derived from Arena's; gear insertion supplies its own policy-independent task.
- `environments/` groups complete Arena graph YAML by task family: `tool_sorting/`,
  `tool_hanging/`, `syringe_disposal/`, `usbc_insertion/`, `gear_insertion/` and
  `cable_routing/`. Each graph includes its
  embodiment, scene, relations, and task.

`isaaclab_arena_vabar.register_components()` registers assets, embodiments, custom tasks, and
parameterized environment factories. Static
environments require no Python factory.

## Tests

The suite contains only durable runtime contracts: task semantics, embodiment action/observation
interfaces, public registration, one physical drop per movable asset, and one
construction/reset/step smoke per environment YAML. The 37 drops and 22 environment smokes are
marked `live` and run in fresh Isaac processes when `ISAAC_CAP_RUN_LIVE_TESTS=1` is set.

The root simulation environment contains Arena, Isaac Lab, PyTorch, OpenUSD, and Isaac Sim. After
`./setup.sh`, run the simulator-free contracts with `uv run pytest`; live cases remain opt-in.

## Gear insertion

The contact-rich insertion directory contains three AUTOLab Gear Mesh families. The single, pair, and train environments select coupled generated
layouts with one, two, and three loose gears, respectively. Success requires assembly before the
button starts the pinion and then requires every released gear to remain seated and turning at its
ratio for one second. The task owns no IK or robot-specific controls; policies own IK.

```bash
uv run vabar-arena show \
  isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/gear_insertion/medium_train.yaml
```

## Canonical tool-sort environments

Three randomized four-object tasks use AUTOLab's electrical, bench, and wiring assets and labels,
`on(bin1)` placement with a 10 mm edge margin, and fixed destination labels. Two items differ from
AUTOLab's sort scenes: easy 1 stages cutting pliers in place of the safety glasses and easy 2
combination pliers in place of the adjustable wrench, each under AUTOLab's own label for that item.
These tasks run AUTOLab's level-1 graph on the Robotiq 2F-85, and AUTOLab's level-1 suite, which
runs that pairing, stages both pliers but never the glasses or the wrench. Syringe disposal retains
its easy and medium variants. Runtime support is Newton-only.

The runtime environment paths are:

- `isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/tool_sorting/easy_1.yaml`
- `isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/tool_sorting/easy_2.yaml`
- `isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/tool_sorting/easy_3.yaml`
- `isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/syringe_disposal/single.yaml`
- `isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/syringe_disposal/both.yaml`
- `isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/syringe_disposal/designated.yaml`
- `isaaclab_arena_vabar/src/isaaclab_arena_vabar/environments/syringe_disposal/cluttered.yaml`

Generated registration modules live under `assets/registrations/`.

The four syringe tasks dispose one, both of two, the designated red-cap syringe
beside a blank spare, or every cluttered syringe (six, or five in three trials). Tray and box positions vary
by +/-5 cm in X/Y; `on` places syringes randomly with a 1 cm edge margin.
Clutter lays its syringes out in a row from one of AUTOLab's clutter trials (in three of them the
RSS version drops one, and Arena parks it off to the side), each barrel across the tray with up to 35 deg of yaw jitter; the other tasks use fixed headings.

## Inspect an environment

Select the typed Newton registration directly with `vabar-arena show`:

```bash
uv run vabar-arena show --env vabar_tool_sorting__easy_1
```

Add `--skip-setup` after an earlier setup.

Policy configurations are owned by `isaaclab_arena_gap`. The three sorting tasks share `tool_sort/gap_perception_loop`;
all four syringe tasks use `syringe_packing/gap_perception`.
Arena episode JSONL remains authoritative.

Inspection with `vabar-arena show` is documented in
[`isaaclab_arena_gap/README.md`](../isaaclab_arena_gap/README.md#inspect).

## FR3 cameras

Every FR3 task uses the camera rig its AUTOLab suite stages in the RSS version, one rig per suite
(`embodiments/fr3_robotiq_2f85/cameras.py`); see
[Shared camera rigs](../docs/DESIGN.md#shared-camera-rigs).
