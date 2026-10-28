"""Visible cable endpoints near a terminal, without an assumed end direction."""
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
# Sibling helpers resolve inside the GaP skill package.
from .perceive_rod import _camera, _frame
from .fixtures_cv import bench_z


def measure_terminal_tail(rgb, depth, K, T, port_xy, *, surface_z=None):
    rgb=np.asarray(rgb,dtype=np.uint8);depth=np.asarray(depth,dtype=float).squeeze()
    K=np.asarray(K,dtype=float).reshape(3,3);T=np.asarray(T,dtype=float).reshape(4,4)
    surface=bench_z(depth,K,T) if surface_z is None else float(surface_z)
    height,width=depth.shape
    def xyz(v,u):
        v=np.asarray(v);u=np.asarray(u);d=depth[v,u]
        cam=np.column_stack(((u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d,np.ones(len(d))))
        return (T@cam.T).T[:,:3]
    values=rgb.astype(float);hi=values.max(axis=2);lo=values.min(axis=2)
    chroma=(hi-lo)/np.maximum(hi,1.)
    achromatic=chroma<.16
    bright=hi[achromatic]
    # Height gating below already removes the bench, so retain cable pixels in
    # deep shadow here; the scene-wide segmenter needs the higher 128 floor but
    # this terminal-local detector does not.
    floor=max(80.,float(np.percentile(bright,68.))) if bright.size else 80.
    mask=achromatic&(hi>=floor)&np.isfinite(depth)&(depth>0)
    v,u=np.nonzero(mask);world=xyz(v,u)
    valid=(world[:,2]>surface+.0005)&(world[:,2]<surface+.200)
    mask[:]=False;mask[v[valid],u[valid]]=True
    labels,count=ndimage.label(mask,np.ones((3,3)))
    sizes=np.bincount(labels.ravel());keep=sizes>=40;keep[0]=False;mask=keep[labels]
    skel=skeletonize(mask,method='lee')
    # RTX's soft lab panel preserves terminal hue but compresses the raw RGB
    # ratios (a typical pixel is 235, 218, 174). Use the same hue contract as
    # the campaign segmenter so the fixture remains an endpoint occluder under
    # both tiled diagnostics and RTX observations.
    import cv2
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    orange=(hsv[...,0]>=8)&(hsv[...,0]<=40)&(hsv[...,1]>=24)&(hsv[...,2]>48)
    near_occluder=ndimage.maximum_filter(orange,size=15)
    def neighbours(p):
        out=[]
        for dv in (-1,0,1):
            for du in (-1,0,1):
                if not (dv or du):continue
                v,u=p[0]+dv,p[1]+du
                if not (0<=v<height and 0<=u<width and skel[v,u]):continue
                # An orthogonal bridge already connects these diagonal pixels.
                # Counting the diagonal too creates a spurious triangle/junction.
                if dv and du and (skel[p[0]+dv,p[1]] or skel[p[0],p[1]+du]):continue
                out.append((v,u))
        return out
    ends=[tuple(map(int,p)) for p in np.argwhere(skel) if len(neighbours(tuple(p)))==1]
    candidates=[];rejected=[]
    for end in ends:
        v,u=map(int,end)
        if v<5 or u<5 or v>=height-5 or u>=width-5:continue
        point=xyz([v],[u])[0]
        if np.linalg.norm(point[:2]-np.asarray(port_xy))>.25:continue
        # Cable shadows may be dark, but a component must have the known
        # white cable's median colour; this excludes grey robot surfaces.
        if np.median(rgb[labels==labels[v,u]])<140:continue
        if near_occluder[v,u]:
            rejected.append('endpoint touches an orange fixture');continue
        path=[(v,u)];prev=None
        for _ in range(200):
            nexts=[q for q in neighbours(path[-1]) if q!=prev]
            if len(nexts)!=1:break
            prev=path[-1];path.append(nexts[0])
            if len(path)>80:break
        if len(path)<8:
            rejected.append('too short to trace a grasp');continue
        # Refuse a mask end caused by a taller object occluding the cable.
        forward=np.asarray(path[0],float)-np.asarray(path[min(5,len(path)-1)],float)
        forward/=max(np.linalg.norm(forward),1e-9)
        probes=np.rint(np.asarray(path[0])+np.arange(3,10)[:,None]*forward).astype(int)
        probes=probes[(probes[:,0]>=0)&(probes[:,0]<height)&(probes[:,1]>=0)&(probes[:,1]<width)]
        heights=xyz(probes[:,0],probes[:,1])[:,2]
        if np.any(heights>surface+.018):
            rejected.append('endpoint terminates behind a taller object');continue
        pixels=np.asarray(path);curve=xyz(pixels[:,0],pixels[:,1])
        rays=curve-T[:3,3];curve+=.0028*rays/np.maximum(np.linalg.norm(rays,axis=1)[:,None],1e-9)
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
        grasps=[]
        for distance in (.015,.025,.040,.060):
            if arc[-1]<distance:continue
            g=[float(np.interp(distance,arc,curve[:,k])) for k in (0,1)]
            tangent=np.array([np.interp(distance+.004,arc,curve[:,k])-np.interp(distance-.004,arc,curve[:,k]) for k in (0,1)])
            tangent/=max(np.linalg.norm(tangent),1e-9)
            grasps.append(dict(xy=g,axis_z=float(np.interp(distance,arc,curve[:,2])),
                               inboard_m=distance,tangent=tangent.tolist()))
        if not grasps:continue
        candidates.append(dict(endpoint=curve[0].tolist(),grasps=grasps,pixel=[u,v],distance_to_port_m=float(np.linalg.norm(curve[0,:2]-np.asarray(port_xy)))))
    # Two nearby pixels at a skeleton junction must not become two tail claims.
    unique=[]
    for c in candidates:
        if not any(np.linalg.norm(np.asarray(c['endpoint'])-np.asarray(o['endpoint']))<.010 for o in unique):unique.append(c)
    if len(unique)!=1:
        return dict(found=False,reason=f'{len(unique)} unambiguous exposed endpoints near the terminal',candidates=unique,rejected=rejected)
    return dict(found=True,**unique[0],surface_z=surface)


def observe_terminal_tail(ctx,port_xy):
    observation=ctx.tool('robot.get_observation');diagnostics=[];found=[]
    for camera in ('cable',):
        block=_camera(observation,camera);frame=_frame(block) if block else None
        if frame is None:continue
        result=measure_terminal_tail(*frame,port_xy)
        result['camera']=camera;diagnostics.append(result)
        if result.get('found'):found.append(result)
    if not found:return dict(found=False,reason='no unambiguous exposed terminal tail',views=diagnostics)
    if len(found)>1 and np.linalg.norm(np.asarray(found[0]['endpoint'])-np.asarray(found[1]['endpoint']))>.015:
        return dict(found=False,reason='camera endpoint estimates disagree by more than 15 mm',views=diagnostics)
    return max(found,key=lambda r:len(r['grasps']))
