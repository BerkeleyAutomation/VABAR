# SPDX-License-Identifier: Apache-2.0
"""Print each run's outcome, final scalar goal metrics and drift, one line per episode.

python tools/fast_tune/show.py RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    for d in argv:
        run = Path(d)
        res = run / "episode_results.jsonl"
        if not res.exists():
            print(f"{run.name}: no results")
            continue
        meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").exists() else {}
        prov = meta.get("provenance", {})
        for line in res.read_text().splitlines():
            e = json.loads(line)
            m = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in (e.get("metrics") or {}).items()
                if not isinstance(v, list)
            }
            print(
                f"{run.name} ep{e.get('episode_id')}: success={e.get('success')} reward={e.get('reward', 0):.3f} "
                f"steps={e.get('episode_length')} close={e.get('close_kind')} wall={e.get('wall_s')} | {m}"
            )
        print(f"  steps/s={prov.get('steps_per_s')} wall_s={prov.get('wall_s')} run_wall_s={prov.get('run_wall_s')}")
        ft = run / "fast_tune.json"
        if ft.exists():
            rec = json.loads(ft.read_text())
            for dr in rec.get("drift") or []:
                print(
                    f"  drift {dr['steps']} steps: worst {dr['worst_mm']} mm ({dr['worst_obj']}), {dr['worst_deg']} deg; {dr['per_object']}"
                )
            if rec.get("peaks"):
                print(f"  peaks: {rec['peaks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
