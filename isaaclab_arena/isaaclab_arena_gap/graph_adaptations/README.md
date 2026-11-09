# VABAR graph adaptations

Every Experiment runs the RSS graph (v1.0) through the RSS adapter (`run_rss_graph.sh`).
Where a config names `gap_adaptation`, `rss_adapter.runner.adapted_graph` copies the pinned
graph to a scratch directory and applies `graph_adaptations/<name>.patch` there; the RSS graphs
at the repository root are never modified, and a patch that no longer applies stops the run. "Artist" assets
are this benchmark's own SimReady models (`isaaclab_arena_vabar` assets), which replace the meshes
of the RSS version.

| Patch | Graph (revision) | Experiment | Why it differs from the RSS graph |
|---|---|---|---|
| `rss_usbc_insert_easy` | `usbc_insert/mate` (v1.0 `18b974e`) | USB-C insertion easy | Arena: the artist plug's silver nose. The plug prompt names it so SAM 3 segments the whole plug |
| `rss_usbc_insert_medium` | `usbc_insert/mate` (v1.0 `18b974e`) | USB-C insertion medium | Arena: the silver plug nose and the silver metal bulkhead, which the RSS graph looks for as a "blue object" |
| `rss_tool_hanging_artist_wrench` | `tool_hanging/gap_perception` (v1.0 `18b974e`) | wrench easy | Arena: the artist ring spanner. Without its CAD, `cad_distal_loop` registers the RSS adjustable wrench onto it and takes the open jaw for the ring |
| `rss_tool_hanging_artist_wrench_level2` | `tool_hanging/gap_perception_level2` (v1.0 `18b974e`) | wrench medium | Same artist ring spanner and the same hunks, on the level-2 graph and the easy graph's scripts it runs from `../gap_perception/scripts`. Its patch paths start at the RSS `tool_hanging/graphs` directory |

These patches adapt the RSS graphs to this benchmark's own assets and stay here. A change
that fixes behaviour not specific to Arena belongs in the RSS version, not in a patch.

Each Experiment selects at most one patch, so a patch is complete for its pair rather than
a delta on another patch. Code a patch imports lives in `isaaclab_arena_gap`, which
`run_rss_graph.sh` puts on the adapter process's `PYTHONPATH`:

- `isaaclab_arena_gap.tool_hanging_asset_profile`: the artist wrench CAD profile
  (`artist_wrench_profiles`) and hook-centreline refinement (`refine_artist_hook`);
- `isaaclab_arena_gap.usbc_shell_perception`: the plug-shell fit (`fit_plug_shell`) the
  medium relook uses.

## USB-C

Both patches declare `sam3.segment_text` in the workflow and prompt for the artist plug
including its metal nose. `rss_usbc_insert_medium` additionally prompts for the medium
scene's metal bulkhead, selects it by measured surface support, and fits the plug shell
with `usbc_shell_perception` for the relook pass.

## Wrench easy and medium

`rss_tool_hanging_artist_wrench` supplies the artist USD mesh and closed-ring landmark to
the RSS graph's CAD registration, queries a wrench rather than an adjustable wrench, fits the
declared hook mesh to the camera cloud to recover the shaft centreline, and keeps the RSS graph's
vertical mating offsets (`crossing_offsets_m` 8/18 mm, `mate_settling_offset_m` 10 mm). The
loaded arm ends 3-8 mm below its Cartesian target, as it does in the RSS run; with the offsets
zeroed, the ring crossed below the shank axis against 7.7 mm of clearance (ring r = 10.2 mm,
shank r = 2.49 mm). Carry transit uses the RSS version's `transit_speed_scale=0.5` because the thin
artist shaft slipped during the full-speed wrist rotation. Unadapted, v1.0's registration
landed the ring feature 70-75 mm on the robot side of the wrench's centre (the open jaw) in
most runs. The environment keeps the artist visuals; registration geometry is exported
into a temporary directory at runtime. Arena's release and hanging predicate remain
authoritative. `rss_tool_hanging_artist_wrench_level2` applies the same changes to both wrench subgraphs of the level-2 graph (`pair_sg_1/2` query, `motion_plan_sg_1/2` transit speed) and to the shared perception scripts; `rss_adapter.runner.adapted_graph` copies the whole `graphs` directory for a patch whose paths name a sibling graph.
