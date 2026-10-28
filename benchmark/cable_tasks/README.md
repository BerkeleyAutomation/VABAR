# Fixed cable task files

100 fixed files per variant. Thirty per variant were preselected for evaluation by
farthest-point sampling over initialisations, before any policy ran.

| Variant | Fixed task files | Cell |
| --- | --- | --- |
| USB-C Easy | [100 files](dataset/usbcinsertion-easy-insertplug) | `cable_insertion/L1` |
| USB-C Medium | [100 files](dataset/usbcinsertion-medium-insertplug) | `cable_insertion/L2` |
| Routing Easy | [100 files](dataset/cablerouting-easy-routecable) | `cable_routing/L1` |
| Routing Medium | [100 files](dataset/cablerouting-medium-routecable) | `cable_routing/L2` |

The [manifest](dataset/manifest.json) records each file's seed, exact
initialisation, SHA-256, and the `selected_tests`.

Each file is its base variant (`usbc_insert/insert_easy`, `insert_medium`,
`cable_route/port_easy`, `port_medium`) drawn at a single seed S. Every base
document has seed 0, so a file is episode S of its variant:

```bash
uv run python -m vabar bench --cell cable_insertion/L1 --print    # the commands
uv run python -m vabar bench --cell cable_insertion/L1 --verify   # resets match the recorded initialisations
uv run python -m vabar bench --cell cable_insertion/L1 --solve graph --limit 3
```

All 400 files reproduce their recorded initialisation exactly on this release.
The evaluation protocol (graph, flags and `ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette`)
is recorded per cell in [`../vabar_v1.json`](../vabar_v1.json). What it gives on this
release is in [`docs/validation/cable.md`](../../docs/validation/cable.md).
