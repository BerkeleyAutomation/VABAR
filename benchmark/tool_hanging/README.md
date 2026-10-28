# Frozen Tool Hanging task specifications

This directory contains the finalized, deterministic Tool Hanging benchmark
instances intended for source control and evaluation sharing.

## Layout

- `easy/tasks/`: 100 fixed Level-1 task files.
- `medium/tasks/`: 100 fixed Level-2 task files.
- `easy/images/` and `medium/images/`: overhead previews for the first 30
  instances at each difficulty.
- `easy/contact_sheet_5x6.png` and `medium/contact_sheet_5x6.png`: compact
  visual reviews of those previews.
- `easy/manifest.jsonl` and `medium/manifest.jsonl`: one provenance record per
  task, including its source family, source episode, seed, and frozen layout
  overrides.

Every task follows the naming convention
`suite-level-name-number`, for example:

```text
toolhanging-easy-hangwrench-001.task
toolhanging-medium-insertscrewdriver-025.task
```

The `(:instance-layout ...)` section in each file records the sampled object
and fixture transforms. Its episode sampler is `fixed`, so evaluation does not
resample the scene.

These files are benchmark artifacts rather than an additional registered task
suite. Each manifest row names the canonical variant (`source_task`) and the
episode (`source_episode`) it was drawn as: the Level-1/2 samplers seed from
`seed + episode` with the suite's seed 0, so instance *k* is that variant's
episode *k*, and `overrides` records the exact transforms of the draw.
`benchmark/vabar_v1.json` lists the first thirty of each level as the evaluated
cells, and `python -m vabar bench --cell tool_hanging/L1 --verify` checks a reset
against the recorded transforms.

```bash
python -m vabar bench --cell tool_hanging/L1 --solve graph   # Level 1, thirty instances
python -m vabar bench --cell tool_hanging/L2 --print         # the underlying vabar run commands
```
