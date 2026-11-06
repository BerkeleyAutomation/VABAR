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

**The full cells have not finished.** This table holds the only runs so far. They are single runs of the same graph on
four 5-syringe Level 2 trays, and they used a local clock and render patch equivalent to `default`'s runtime, not
`vabar bench`.

| Run | Camera | Syringes posted |
| --- | --- | --- |
| `t13`, `t14`, `t19`, `t20`, stock motion | tiled | 15/20 |
| `t14`, stock motion | RTX | 5/5 |

Before relying on the suite, run `vabar bench --cell syringe_packing/L1` and `/L2`, more than once: other cells swing
widely between runs. The README's syringe rows predate this change until then.

## Not done

- `gt` was not touched. RoboSimStudio-dev has later fixes for it (`114329fe`), and it does not post here yet.
- The frozen copies under `benchmark/syringe_packing/` still describe the old documents.
