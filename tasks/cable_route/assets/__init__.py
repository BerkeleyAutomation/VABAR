# SPDX-License-Identifier: Apache-2.0
"""What this task is about: a cable, the pegs it is routed around, and the rod.

Imported for its side effects -- each module registers what it declares -- and
the physical parameters live here rather than in a loader because they are
facts about the *cable* rather than about any one route. A weave that wants a
longer one says so per variant, in ``layout``, which is also where each
variant's stations stand and where the records a document names are minted.
"""

from __future__ import annotations

from . import cable, fixtures, layout, rod

__all__ = ["cable", "fixtures", "layout", "rod"]
