"""Check the held syringe is seated in the aperture before permitting release."""

import os
import sys
from typing import Any,TypedDict

import numpy as np
from gap import NodeContext

# Helpers shared by this graph's scripts live beside them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception_checks import detect,crop_for_world_box,rod_like,xyz  # noqa: E402
from motion_checks import rotation,vec  # noqa: E402


class Output(TypedDict):
    final_pose:dict[str,Any]
    observed_depth_m:float
    axis_alignment:float

MIN_DEPTH_M=-0.012
"""Release is allowed with the leading end up to 12 mm short of the rim plane, on
the aperture axis. Once released there, an upright syringe falls through a 55 mm
hole. VAB-industrial's scripted policy deliberately releases its lower end 10 mm
above the lid."""
MAX_LATERAL_ERROR_M=0.012
COLUMN_POINTS_MIN=8

def run(ctx:NodeContext,observation:dict[str,Any],placement_plan:dict[str,Any])->Output:
    center=xyz(placement_plan['aperture']);rim=float(placement_plan['rim_z'])
    axis=xyz(placement_plan['fixture_axis']);axis/=np.linalg.norm(axis)
    hand_pose=ctx.tool('robot.get_ee_pose')['pose'];hand=vec(hand_pose['position'])
    plane=np.array([center[0],center[1],rim])

    # Depth and centring come from live TCP proprioception and the registered
    # hand-map offset, not from the cameras. Cameras cannot see how far the
    # syringe is through: the lid hides everything below the rim and the
    # Robotiq fingers hide the barrel's top. Sweep s5 t08 and s3 t07 were
    # refused at -23 mm while the simulator's mesh end was +7.8 mm through
    # the hole, 10.6 mm off-centre. The hand map, checked against the same
    # ground truth, was within 8 mm of the mesh end: 59.6 vs 67.2 mm ahead of
    # the TCP.
    feature=vec(placement_plan['leading_end_in_hand'])
    world_direction=rotation(hand_pose).apply(feature)
    endpoint=hand+world_direction
    depth=float((endpoint-plane)@axis)
    lateral=float(np.linalg.norm((endpoint-plane)-depth*axis))
    alignment=abs(float((world_direction/np.linalg.norm(world_direction))@axis))
    if lateral>MAX_LATERAL_ERROR_M:
        raise RuntimeError(f'held endpoint is {lateral*1000:.1f} mm outside the aperture centre')

    # The cameras keep one job: catching a syringe that pivoted in the fingers
    # (tray_clutter_t24: 44 deg), which proprioception cannot see. Only a
    # detection whose lowest points are inside the aperture column counts; a
    # nearest rod-like mask outside it was the horizontal gripper (sweep dev2
    # t06 refused a syringe 1.0 deg from vertical at 89.2 deg).
    def in_column(item):
        rel=item['points']-plane;along=rel@axis
        lateral_all=np.linalg.norm(rel-np.outer(along,axis),axis=1)
        if np.count_nonzero(lateral_all<=MAX_LATERAL_ERROR_M)<COLUMN_POINTS_MIN:
            return False
        low=np.median(item['points'][along>=np.percentile(along,95)],axis=0)-plane
        return float(np.linalg.norm(low-(low@axis)*axis))<=MAX_LATERAL_ERROR_M
    column=[]
    for camera in observation.get('cameras',[]):
        if camera.get('name') not in ('agentview','overhead'):continue
        crop=crop_for_world_box(camera,center+[-.11,-.11,-.12],center+[.11,.11,.15])
        for item in detect(ctx,camera,crop=crop,prompt='medical syringe',floor=.02):
            if rod_like(item) and np.linalg.norm(item['center']-hand)<.16 and in_column(item):
                column.append(item)
    # Tilt only from points above the rim. Everything the box already holds is
    # below the rim, and a detection through the hole can merge the held barrel
    # with a pile syringe under it: sweep s6 t07 refused a syringe standing
    # 2 deg from vertical because syringe_0 leaned 54 deg in the column, 4 mm
    # off its axis.
    def exposed_axis(item):
        points=item['points'][item['points'][:,2]>rim+0.005]
        if len(points)<30:
            return None
        _,spread,basis=np.linalg.svd(points-np.median(points,axis=0),full_matrices=False)
        return basis[0] if spread[0]>2.0*max(spread[1],1e-9) else None
    # A near-horizontal detection above the rim is the Robotiq's fingers, which
    # straddle the barrel right over the hole. A syringe slipping in the pads
    # has so far measured 44-60 deg, not 90. Sweep s6 t14/t16 refused
    # syringes 0-1 deg from vertical at 86.6 deg.
    exposed=[a for a in (exposed_axis(item) for item in column)
             if a is not None and abs(float(a@axis))>=np.cos(np.deg2rad(75.0))]
    if exposed:
        alignment=min(alignment,max(abs(float(a@axis)) for a in exposed))
    if alignment<.80:
        raise RuntimeError(f'pre-release verification found a tilted syringe ({np.degrees(np.arccos(alignment)):.1f} deg from insertion axis)')
    if depth<MIN_DEPTH_M:
        raise RuntimeError(f'syringe is only {depth*1000:.1f} mm through the aperture; release prohibited')
    return {'final_pose':hand_pose,'observed_depth_m':depth,'axis_alignment':alignment}
