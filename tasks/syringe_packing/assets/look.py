# SPDX-License-Identifier: Apache-2.0
"""How the syringe bench looks under the path tracer.

The house room (:data:`~robosimstudio.core.render.room.HANGAR_CLOSED`, which
these documents rendered in when they stated no look) with the industrial
backdrop on the two walls the agentview camera faces. The bench is seen from
``+x``, so ``-x`` (behind the robot) is the lab and ``+y`` is the warehouse.
That is the pair RoboSimStudio-dev's ``sharps_disposal`` hung on its dressing
walls (``eac6c2a0``), and what the project page's clip shows.

Scenery is drawn by the path tracer alone. The tiled renderer this suite's
perception graph observes through gets the same house look as before, so
nothing the policy sees changes.
"""

from __future__ import annotations

from dataclasses import replace

from robosimstudio.core.render.room import HANGAR_CLOSED
from robosimstudio.core.render.scenery import register_scenery
from robosimstudio.core.render.spec import RenderSpec

from vabar.backdrop import papered_room

__all__ = ["SCENERY_NAME", "SYRINGE_PACKING_LOOK"]

SCENERY_NAME = "syringe_packing_lab"


@register_scenery(SCENERY_NAME)
def syringe_packing_lab(kit) -> None:
    """``house_room_closed`` with the lab behind the robot and the warehouse beside it."""
    papered_room(kit, {"-x": "wall_west.jpg", "+y": "wall_north.jpg"})


SYRINGE_PACKING_LOOK: RenderSpec = replace(HANGAR_CLOSED, scenery=SCENERY_NAME)
"""The house look, with :func:`syringe_packing_lab` as its room."""
