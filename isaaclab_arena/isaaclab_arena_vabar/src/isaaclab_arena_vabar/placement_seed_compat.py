"""Work around Arena's overlapping placement-candidate seeds.

Arena seeds candidate ``i`` of environment ``e`` with
``placement_seed + e * candidates_per_env + i``, so consecutive placement seeds
share all but one candidate and usually select the same layout. Until
isaac-sim/IsaacLab-Arena#1523 is in the pinned Arena release, spread the seed before candidates
are generated so distinct seeds use distant candidate windows. The patch retires
itself once Arena seeds candidates independently.
"""

from __future__ import annotations

import hashlib

_installed = False


def spread_placement_seed(seed: int) -> int:
    """Return a reproducible seed far from the one derived for any other ``seed``."""
    digest = hashlib.sha256(f"vabar-placement:{seed}".encode()).digest()
    return int.from_bytes(digest[:6], "little")


def install_placement_seed_compat() -> None:
    """Generate placement candidates from a spread copy of the placement seed."""
    global _installed
    if _installed:
        return

    from isaaclab_arena.relations.placement_candidate_generator import (
        PlacementCandidateGenerator,
    )

    original = PlacementCandidateGenerator.generate_candidates
    if hasattr(PlacementCandidateGenerator, "_seed_candidate") or getattr(
        original, "_isaac_cap_placement_seed_compat", False
    ):
        _installed = True
        return

    def generate_candidates(self, *args, **kwargs):
        seed = self.params.placement_seed
        if seed is None:
            return original(self, *args, **kwargs)
        self.params.placement_seed = spread_placement_seed(seed)
        try:
            return original(self, *args, **kwargs)
        finally:
            self.params.placement_seed = seed

    generate_candidates._isaac_cap_placement_seed_compat = True
    PlacementCandidateGenerator.generate_candidates = generate_candidates
    _installed = True
    print("[vabar] installed placement-seed spreading compatibility fix", flush=True)
