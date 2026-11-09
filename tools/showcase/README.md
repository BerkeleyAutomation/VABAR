# Showcase clips

Tools for the project page's task videos. They are for presentation only and
change no task, document or benchmark number. The scenes' own looks (the wood
bins on the tool-sorting bench, and the lab and warehouse walls from
`vabar/backdrop.py`) belong to the tasks, so every run draws them. These tools
only add cameras and make the clips.

| File | What it does |
| --- | --- |
| `hero_run.py` | `python -m vabar run`, plus presentation cameras (`HERO_CAMS`); its docstring covers the other switches |
| `render_offline.py` | Re-renders a recorded run through the path tracer from recorded state, with no re-simulation. Ported from RoboSimStudio-dev `tools/render_offline.py` (`e94e85f2`) |
| `encode.sh` | Speeds a clip up to about 20 s and writes a 1280×720 master and a 760×570 crop for the page's task cards |

## Collect cheap, render after

A path tracer in the loop costs 1–12 control steps per second. A run without one
manages about 65. So run the policy once with `--record` and no presentation
cameras, then replay the recorded body state through RTX for the one view
wanted:

```bash
OUT=outputs/showcase
# 1. the policy, recording state (a perception graph still needs --obs-renderer rtx)
uv run python tools/showcase/hero_run.py --task tool_sort/level1_4 \
  --solve graph --graph gt --bus-timeout 180 --record --out $OUT --name tool_sorting
# 2. the clip
HERO_CAMS='ts:0.95,0.02,1.62:0.05,0.02,0.92:50:960x720;hero:0.90,0.02,1.62:0.05,0.02,0.88:40:1280x720' \
  uv run python tools/showcase/render_offline.py --run $OUT/tool_sorting \
  --renderer rtx --cameras ts,hero --every 4
# 3. the page's files
tools/showcase/encode.sh $OUT/tool_sorting/rerender/0000/hero.mp4 task-tool-sorting
```

Tool sorting took about 1 minute to collect and 5 to render. The same clip
rendered live took about 40.

## The six clips

Each row is the instance and policy that match the page's original clip, and
the camera the clip uses.

| Task | Instance | Policy (flags) | Camera (`HERO_CAMS`) |
| --- | --- | --- | --- |
| Tool Sorting | `tool_sort/level1_4` | `--solve graph --graph gt` | `ts:0.95,0.02,1.62:0.05,0.02,0.92:50:960x720` (4:3), `hero:0.90,0.02,1.62:0.05,0.02,0.88:40` |
| Tool Hanging | `tool_hanging/wrenches_level1` | `--solve routine` | `hero:0.50,-0.95,1.25:0.68,0,1.0:42` |
| Gear Meshing | `gear_mesh/pair` | `--solve graph --graph gap_perception --motion curobo --no-privileged --perception vision --obs-renderer rtx` | `hero:1.00,-0.45,1.15:0.55,0.0,0.80:34` |
| Syringe Packing | `syringe_packing/tray_pair_both` | `--solve graph --graph gap_perception --motion curobo --roll held --no-privileged --perception vision --obs-renderer rtx` | `front:1.55,-0.15,1.55:-0.35,-0.05,1.0:42:960x720` |
| Cable Insertion | `usbc_insert/insert_easy` | `--solve graph --graph gt` | `hero:0.82,-0.10,0.92:0.43,-0.01,0.84:32` |
| Cable Routing | `cable_route/port_easy --episodes 10010` | `--solve graph --graph gt --no-privileged --perception vision --obs-renderer rtx --settle-steps 60 --bus-timeout 600`, `ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette` | `hero:1.10,-0.40,1.22:0.50,-0.06,0.76:36`, `c43:…:40:960x720` (4:3) |

Notes:

- **Tool sorting.** The scripted routine places only the two nuts on
  `level1_4` (0.4). The `gt` graph places all five.
- **Gear meshing.** The routine needs Franka finger bodies and raises on the
  suite's 2F-85. The graph is the suite's policy, as `docs/validation/gear_mesh.md`
  records.
- **Syringe packing.** With this release's pins the Level-1 graph posts one of
  the two syringes (0.5), as `docs/VALIDATION.md` records. The page's clip
  posts both.
- **Cable routing.** The camera aims at the cable's resting centre on the table,
  read from the recorded body state. The `cable_route/weave3` routine reaches
  1.0, but its two arms cross. The `gt` graph routes with one arm.
