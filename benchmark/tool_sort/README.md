# Frozen Tool Sort task specifications

This directory contains the finalized, deterministic Tool Sort benchmark
instances intended for source control and evaluation sharing.

## Layout

- `level1/tasks/`: 30 fixed Level-1 task files.
- `level2/tasks/`: 30 fixed Level-2 task files.

## Task naming

Tasks keep their original names from the `tool_sort` suite:

```text
level1_1.task  …  level1_30.task
level2_1.task  …  level2_30.task
```

Each is a machine-generated scene: a fixed set of industrial tools in `bin1`, each
assigned to a labelled compartment in `bin2`. Level 1 scatters four or five separated tools.
Level 2 drops 8 to 12 of them, of four types, into a pile with one distractor no
compartment names, which stays in `bin1`. The scene geometry, object poses and
compartment assignments are fully determined by the seed embedded in the task's
sidecar — no resampling occurs at evaluation time.

## Running

These task files are self-contained and load through the standard `tool_sort`
suite infrastructure. The `gap_perception_loop` graph is the intended policy:

```bash
python -m vabar run --task tool_sort/level1_1 --solve graph --graph gap_perception_loop \
  --no-privileged --perception vision
python -m vabar bench --cell tool_sorting/L1 --solve graph   # all thirty
python -m vabar bench --cell tool_sorting/L2 --solve graph   # all thirty of the heap tier
```
