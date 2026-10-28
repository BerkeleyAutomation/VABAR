# SPDX-License-Identifier: Apache-2.0
"""The VAB-industrial hand tools these documents sort.

Imported for its side effects: the records register on import.

**This declaration is duplicated in ``astro``, deliberately.** Both tasks stand
in the same vendored workcell and each carries its own copy of what it needs, so
either directory can be handed to somebody on its own. The registries accept a
second *identical* declaration and still refuse a conflicting one, which is what
makes the redundancy safe rather than merely tolerated.
"""

from __future__ import annotations

from . import records

__all__ = ["records"]
