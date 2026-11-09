# Local asset quality reports

The local generator tests every dynamic industrial asset owned or used by this repository at the
currently checked-out revision. There is intentionally no CI job for this command.

The inventory contains 37 props. Static tables, bins, racks, holders, and
other kinematic scene fixtures are not drop-test candidates. Inventory tests fail if a registered
movable prop is omitted or duplicated.

Every asset starts its own fresh Isaac Sim process. The generator enumerates the inventory and isolates every asset,
its unchanged canonical workcell table, and its production physics profile. The active collision
AABB starts 10 cm above the table and runs for ten simulated seconds. A pass requires finite state,
table contact, retention inside the tabletop, at most 2 mm final penetration, and at most 1 mm and
1° of movement over the final second. Rest is judged on pose, not velocity: MuJoCo contact chatter makes
the instantaneous velocity of a resting asset spike to several cm/s while it moves less than 0.1 mm.

Every drop-tested asset follows this catalog layout:

```text
assets/
├── README.md        # generated catalog table
└── <registry_name>/
    ├── <registry_name>.usda
    ├── README.md
    ├── media/
    │   ├── collision.png
    │   ├── thumbnail.png
    │   └── drop.gif
    ├── payloads/        # optional composed implementation layers
    └── variants/        # optional named alternatives
```

The catalog README contains one row per movable asset with its name, thumbnail, drop GIF, and
enabled collision body. The canonical USDA, per-asset README, and three media files are required.
`payloads/` and
`variants/` are present only when the asset needs them. Each per-asset README includes:

- `README.md`, including the measured mass, authored friction and restitution, active collider
  types, drop verdict, and measured settling data;
- `media/thumbnail.png`, with the authored pivot marked and local X/Y/Z axes projected into the
  image;
- `media/collision.png`, a deterministic rendering of the enabled composed collider geometry;
- `media/drop.gif`, rendered from the complete drop-test recording.

After `uv run vabar-arena setup`, run:

```bash
uv run tools/generate_asset_docs.py
```

This starts 37 sequential fresh processes, validates each asset-named JSON and drop-media pair,
renders every enabled collision body,
immediately, and replaces only the generator-owned catalog README plus `README.md` and `media/` in
each asset folder.
A crash before evidence affects only that asset and is retried
up to twice; a validated teardown-only exit is accepted. Add `--fail-on-drop-failure` when the
command should generate all documentation and then return nonzero for a failing physical verdict.
The complete report is staged first; publication restores the catalog README and every previous
README/media pair if a replacement fails.

The live stage canonicalizes the sign of authored mass-frame quaternions before Newton import.
Quaternion signs `q` and `-q` describe exactly the same rotation; this transient opinion avoids a
Kit USD-physics importer heap corruption for a near-180-degree frame without modifying the source
asset or canonical table. Non-finite physical measurements are recorded as JSON `null` and rendered
as `Not finite` while retaining the failed verdict and its reasons.
