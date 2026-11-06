# `syringe_packing`: the working scene

`speed/default-fast` gave this suite the multi-syringe graph, but kept the scene the v1.0 export shipped. Its notes
named the colliders as the likely reason the cells stayed below the paper, untested. This change ports the scene the
graph was written on.

**Source.** RoboSimStudio-dev `vabar/tasks/sharps_disposal` at `1cd9f76a`, chiefly `18ab35f6` and `5c561ad9`.

## What changed

| Area | Before | Now |
| --- | --- | --- |
| Syringe collider | One authored box, 23.6 mm across its 150 mm, on a 10.7 mm barrel | 61 CoACD hulls of the mesh it draws |
| Tray collision | An invisible floor slab, 6 mm high and 4 mm thin, and no walls | Four walls on the drawn walls, and a corrected 20 mm floor |
| Layouts | Syringes crossing walls that did not exist (`tray_clutter_t13` holds 7) | 22 out-of-tray syringes removed, documents and budgets re-derived (`t13` holds 5) |
| Robot | Panda with the Franka hand | `default.toml` runs the Panda with a Robotiq 2F-85, and the host declares `gap_perception` |
| Variants | 94, including the dropped `tray_pair_tNN` sweep | 64 |

Unchanged: the graphs, the variant names the cells use, and the `paper` config's protocol.
`vab_syringe_fast@v1` stays registered, but `fast.toml` no longer maps this suite to it: it was measured on the old
scene and needs its own check here.

## Checks

- `python -m vabar list` loads all 64 variants. Config and release tests pass (`tests/test_release.py`,
  `tests/test_config.py`, `tests/test_bench_config.py`).
- All 58 box-shifted workcells match the base workcell plus each name's shift.

## Results

One run of each full cell, `vabar bench --config default`, 2026-11-06: RTX observations, lockstep clock, unchanged graph,
6 MPS workers on one 48 GB GPU, shared tool servers.

| Cell | Whole instances | Mean reward | Syringes posted | Wall, all in | Before this change (`fast`) | Paper |
| --- | --- | --- | --- | --- | --- | --- |
| Level 1 | **28/30 (93%)** | 0.967 | 58/60 | 29.5 min | 7-13/30 | 87.0% |
| Level 2 | **14/30 (47%)** | 0.786 | 138/176 | 82 min | 0/30, reward 0.27 | 63.0% |

These compare a changed setup with earlier numbers, so they are not like for like: the scene, hand, graph and runtime all
differ from the paper's. This is one run per cell, other cells swing widely between runs, and it does not separate
which of the scene changes matters most. The "before" column ran 8 workers under `fast`.

**Failures, Level 1 (2):** `could not perceive 'medical syringe'` (1), and `verify_grasp` finding no syringe in either
camera (1).

**Failures, Level 2 (16):**

| Cause | Instances |
| --- | --- |
| Graph ended at `abort` with no failing node recorded, not yet diagnosed | 6 |
| `could not perceive 'medical syringe'` | 4 |
| Grasp approach tracking off by 13-18 mm, so release is refused | 3 |
| `verify_grasp` finding no syringe in either camera | 2 |
| cuRobo refusing the aperture approach | 1 |

`tray_clutter_t19` posts 1 of 5 here and did in an earlier RTX-camera run. `t25` posts none. Level 2 is below the paper,
so the suite still has open failures.

A bare `vabar run` on the default 30 s bus timeout failed at step 1 under the lockstep clock, so `default.toml` now
sets the cells' 180 s for this suite. That flag has not been rerun end to end outside `vabar bench`.

## Not done

- `gt` was not touched. RoboSimStudio-dev has later fixes for it (`114329fe`), and it does not post here yet.
- The frozen copies under `benchmark/syringe_packing/` still describe the old documents.
