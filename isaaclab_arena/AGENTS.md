# Agent Entry Point

This repository is a UV workspace for the GaP-to-Arena adapter, industrial benchmark content, and
benchmark-authoring tools.

Read these files before changing code:

1. `docs/SETUP.md` - exact repositories, revisions, installation, preflight, and run commands.
2. `docs/DESIGN.md` - protocol and implementation boundaries.

Working rules:

- Put bridge, adapter, and integration-specific graph changes in `isaaclab_arena_gap/`.
- Put policy-independent benchmark scenes, tasks, assets, scoring, and definitions in
  `isaaclab_arena_vabar/`.
- Put independently movable authoring and conversion utilities in `tools/`.
- Put reusable scene-building, embodiment, placement, camera, recording, and scoring framework
  changes upstream in Isaac Lab Arena; keep benchmark-specific content in `isaaclab_arena_vabar/`.
- Treat the `external/` gitlinks as immutable source pins. Use `./setup.sh` on a fresh clone
  and `uv run vabar-arena run` for local evaluation; do not run
  `git submodule update --remote` or initialize every nested submodule recursively.
- Preserve scalar `ctx.tool()` calls and public tool schemas.
- Keep industrial embodiment profiles and task-pair adaptations in this repository.
- Treat Arena's episode JSONL as authoritative; GaP graph completion is not the benchmark score.
- Preserve `num_envs=1` equivalence. Vector jobs use one isolated GaP graph session per Arena
  environment, each starting its own tools; graph sessions never run in lockstep.

Before handing work to another agent, run the non-GPU tests and preflight in
[docs/SETUP.md section 7](docs/SETUP.md#7-run-the-non-gpu-tests-and-preflight). They cover the
bridge transport and RSS adapter, the Arena policy, adapters, and industrial benchmark contracts; live simulator
cases remain skipped unless explicitly enabled. If any dependency revision changed, update the
revision table in `docs/SETUP.md`.
