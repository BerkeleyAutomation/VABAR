"""Approach the measured barrel, validate TCP tracking, then close."""

import os
import sys
from typing import Any,TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation

# Helpers shared by this graph's scripts live beside them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_checks import planned_linear,checked_execute,pose_error,vec,rotation  # noqa: E402


class Output(TypedDict):
    grasp_pose:dict[str,Any]

PRECLOSE_WIDTH_M=0.030
"""Jaw width the descent is made at. Fully open, the Robotiq's fingertip band spans
+/-49 mm across the jaw. In a packed tray that lands a finger on the next barrel:
tray_clutter_t16 and t17 have neighbours 37-49 mm away, the hand stopped 9 mm high,
and the tracking check refused the grasp. At 30 mm the band is +/-23.6 mm, which
clears a neighbour's barrel edge at 37 - 5.4 mm, and still leaves +/-10 mm for
the barrel to enter between the pads. On a geared hand compute_grasp.py raises
the TCP by the fingertip drop this costs; on a prismatic one there is no drop
(compute_grasp.fingertip_drop)."""

ALIGN_ROTATION_TOLERANCE_RAD=0.05

def _half_turn(pose):
    q=(Rotation.from_euler('z',np.pi)*rotation(pose)).as_quat()
    return {'position':dict(pose['position']),'rotation':{'x':float(q[0]),'y':float(q[1]),'z':float(q[2]),'w':float(q[3])}}

def run(ctx:NodeContext,grasp_pose:dict[str,Any],align_pose:dict[str,Any],object_width_m:float,
        preclose_width_m:float|None=None)->Output:
    plan=ctx.tool('motion.plan_joint',pose=align_pose,orientation='lock')
    # A parallel jaw grips a barrel the same way at yaw and yaw+180 deg. When the
    # wrist cannot reach the requested one, take the half-turn. Sweep s6 t20: the
    # align pose was planned with 0.80 rad of rotation error, executed 46 deg
    # off, and the grasp was refused.
    if float(plan.get('rotation_error_rad',0.0))>ALIGN_ROTATION_TOLERANCE_RAD:
        turned_align,turned_grasp=_half_turn(align_pose),_half_turn(grasp_pose)
        alternative=ctx.tool('motion.plan_joint',pose=turned_align,orientation='lock')
        if alternative.get('planned') and float(alternative.get('rotation_error_rad',9.0))<float(plan.get('rotation_error_rad',9.0)):
            plan,align_pose,grasp_pose=alternative,turned_align,turned_grasp
    if not plan.get('planned'):
        raise RuntimeError('pregrasp approach unavailable')
    checked_execute(ctx,plan['trajectory'],align_pose)
    # compute_grasp.py narrows the jaw below PRECLOSE_WIDTH_M when perception
    # measured a closer neighbour, and raises the TCP by that width's fingertip
    # drop when the hand in use is one whose pads swing.
    width=float(preclose_width_m) if preclose_width_m is not None else max(PRECLOSE_WIDTH_M,float(object_width_m)+0.012)
    ctx.tool('robot.set_grip',width_m=width,ramp_steps=40,settle_steps=40)
    # The global plan gets the hand to the contact neighbourhood. Finish the
    # last few millimetres with the simulator's Cartesian contact controller,
    # then require measured agreement before closing.
    planned_linear(ctx,grasp_pose,scale=3.0,position_tolerance=.010)
    ctx.tool('robot.go_to_pose_cartesian',pose=grasp_pose)
    ctx.tool('robot.wait_steps',steps=30)
    actual=ctx.tool('robot.get_ee_pose')['pose']
    _,angle=pose_error(actual,grasp_pose)
    local=rotation(grasp_pose).inv().apply(vec(actual['position'])-vec(grasp_pose['position']))
    # The feedback servo can retain a small Cartesian bias after the global
    # trajectory, especially with the wrist close to the table. Compensate
    # from the measured TCP error before deciding whether the barrel is inside
    # the bilateral pad envelope. The guard below remains unchanged: retries
    # improve centering instead of declaring a one-sided pinch acceptable.
    for _ in range(2):
        if abs(local[0])<=.0045 and abs(local[1])<=.010 and abs(local[2])<=.005 and angle<=.04:
            break
        world_error=vec(actual['position'])-vec(grasp_pose['position'])
        if np.linalg.norm(world_error)>.012 or angle>.06:
            break
        corrected={
            'position': {
                key: float(grasp_pose['position'][key]-world_error[index])
                for index,key in enumerate(('x','y','z'))
            },
            'rotation': dict(grasp_pose['rotation']),
        }
        ctx.tool('robot.go_to_pose_cartesian',pose=corrected)
        ctx.tool('robot.wait_steps',steps=30)
        actual=ctx.tool('robot.get_ee_pose')['pose']
        _,angle=pose_error(actual,grasp_pose)
        local=rotation(grasp_pose).inv().apply(vec(actual['position'])-vec(grasp_pose['position']))
    # The jaw gap is sensitive across tool X. Along the barrel/tool Y and
    # vertically within the pad depth there is more room. This rejects a miss
    # without treating a safe 4 mm station offset as an empty grasp.
    if abs(local[0])>.0045 or abs(local[1])>.010 or abs(local[2])>.005 or angle>.04:
        raise RuntimeError(
            'grasp alignment outside pad envelope: '
            f'local error {(local*1000).round(1).tolist()} mm, {np.degrees(angle):.1f} deg'
        )
    ctx.tool('robot.set_grip',object_width_m=object_width_m,squeeze_m=0.002,ramp_steps=100,settle_steps=120)
    return {'grasp_pose':actual}

