# SPDX-License-Identifier: Apache-2.0
"""What the documents name: the gear, board and mat records, registered at load.

RoboSimStudio's generic loader executes a suite's ``staging.py``, ``scenes.py``
and ``records.py`` before it lowers a document, and no longer the ``assets/``
package itself. This module is that third door: importing ``assets.records``
registers every entity class and asset record the documents type their objects
with, so ``(gears-seated board)`` resolves to the state the board publishes.
"""

from __future__ import annotations

from .assets import records as _records  # noqa: F401 -- registers on import
