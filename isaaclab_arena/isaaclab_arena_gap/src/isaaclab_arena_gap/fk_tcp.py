"""Shared DROID/Robotiq panda_hand -> TCP geometry constants.

The offset uses GaP's negated link-to-TCP convention (``o = (0, 0, -0.157)``);
the positive length is exported to the cuRobo skill process. The Robotiq TCP
frame is the panda_hand frame rotated by ``ROBOTIQ_TCP_ROT_Z`` about z.
"""

from __future__ import annotations

import numpy as np

ROBOTIQ_HAND_TO_TCP_Z = 0.157
ROBOTIQ_TCP_OFFSET = np.array([0.0, 0.0, -ROBOTIQ_HAND_TO_TCP_Z], dtype=np.float64)
ROBOTIQ_TCP_ROT_Z = np.pi / 4.0
