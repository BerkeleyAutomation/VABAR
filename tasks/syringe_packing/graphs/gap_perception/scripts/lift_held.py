"""Slow, checked vertical lift; observation after lift verifies the grasp."""

import os
import sys
from typing import TypedDict

from gap import NodeContext

# Helpers shared by this graph's scripts live beside them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_checks import planned_linear,vec,rotation,pose  # noqa: E402


class Output(TypedDict):
    lifted_pose: dict

def run(ctx:NodeContext)->Output:
    start=ctx.tool('robot.get_ee_pose')['pose']
    # The first short lift avoids impulsively accelerating a marginal pinch.
    for height in (0.025,0.20):
        target=pose(vec(start['position'])+[0,0,height],rotation(start))
        actual=planned_linear(ctx,target,scale=3.0)
    return {'lifted_pose':actual}

