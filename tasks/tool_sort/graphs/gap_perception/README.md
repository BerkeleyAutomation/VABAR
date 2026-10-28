# Tool sorting: perception GAP graph

This graph sorts every visible tool from the source bin into the destination
compartment carrying the corresponding printed type label. It uses three Open
Skills:

- `perceiving-sorting-pairs` reads the four labels, reconstructs each
  compartment floor from overhead RGB-D, and repeatedly returns one perceived
  source tool with its matching destination OBB;
- `grasping-linear-feature` fits the selected part's full 3D axis, computes a
  perpendicular inclination-aware grasp, approaches along that axis-normal,
  and closes the gripper;
- `transporting-objects` lifts the held tool, moves above the selected
  compartment, descends inside its walls, releases, and retracts.

There is no simulator goal cursor, object pose, region pose, category-to-cell
table, or grasp-contact query. The label reading and object-to-label association
come from the overhead image; all manipulation coordinates come from calibrated
depth. The layout is perceived once and the source bin is reobserved after each
placement.

The task rig already provides `agentview`, `overhead`, and `eye_in_hand` RGB-D
cameras. Sorting perception intentionally uses `overhead`: unlike insertion,
this task does not need to estimate a functional feature relative to the hand.
The wrist camera remains available to grasp skills but is not required by the
current direct-IK composition.

## Run

Configure a working runtime VLM first (`OPENROUTER_API_KEY`, or the documented
`GAP_VLM_PROVIDER=vertex` settings), then run:

```bash
uv run python -m vabar run \
  --task tool_sort/sort_all --solve graph --graph gap_perception \
  --renderer tiled --episodes 1 --max-steps 20000 \
  --record --record-from-motion \
  --out outputs/tool_sort_gap_perception_sort_all
```

To use an OpenAI API key for this run without changing the shared OpenRouter
default, first export `OPENAI_API_KEY`, then add these temporary overrides to
the run command:

```bash
GAP_VLM_PROVIDER=openrouter \
GAP_VLM_BASE_URL=https://api.openai.com/v1 \
GAP_VLM_API_KEY="$OPENAI_API_KEY" \
GAP_VLM_MODEL=gpt-4o-mini \
uv run python -m vabar run \
  --task tool_sort/sort_all --solve graph --graph gap_perception \
  --renderer tiled --episodes 1 --max-steps 20000 \
  --record --record-from-motion \
  --out outputs/tool_sort_openai
```

Here `openrouter` names the library's OpenAI-compatible HTTP adapter; the base
URL routes the request directly to OpenAI rather than to OpenRouter.

Graph validation is credential-free:

```bash
GAP_SKILLS_PATH=robot_skills:../open-robot-skills \
uv run gap run tasks/tool_sort/graphs/gap_perception --validate-only
```
