#!/usr/bin/env python3
"""Editable AB3D part volumes, fixed palette baking, VOX export and diagnostics.

Authoring dependencies: numpy and Pillow. No dependency is used by the game.
Arrays use [y,z,x]; model coordinates are source-pixel units, Y down, front -Z.
VOX files use (x,z,ny-1-y), with their origin/pose preserved in JSON sidecars.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
import colorsys
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw


def unpack(snapshot):
    nx, ny, nz = snapshot["size"]
    grid = np.zeros((ny, nz, nx), dtype=np.uint8)
    for z in range(nz):
        for x in range(nx):
            c = z * nx + x
            for y0, y1, colour in snapshot["runs"][snapshot["columns"][c]:snapshot["columns"][c+1]]:
                grid[y0:y1, z, x] = colour
    return grid


def pack(grid):
    ny, nz, nx = grid.shape
    columns, runs = [], []
    for z in range(nz):
        for x in range(nx):
            columns.append(len(runs))
            y = 0
            while y < ny:
                colour = int(grid[y,z,x]); start = y; y += 1
                if not colour:
                    continue
                while y < ny and grid[y,z,x] == colour:
                    y += 1
                runs.append([start,y,colour])
    return columns + [len(runs)], runs


def rays(grid, quarter):
    """Cardinal view as [image y,image x,depth], front to back."""
    if quarter == 0: return grid.transpose(0,2,1)
    if quarter == 1: return grid[:,::-1,:]
    if quarter == 2: return grid[:,::-1,::-1].transpose(0,2,1)
    if quarter == 3: return grid[:,:,::-1]
    raise ValueError("Not a cardinal camera")


def first_hits(solid, view, canvas=64):
    """Original-resolution first-hit IDs AND depths, before any colour transfer."""
    n = solid.shape[0]
    ids = np.arange(1,solid.size+1,dtype=np.int32).reshape(solid.shape)
    q = view["yaw"]//90
    occupied = rays(solid,q)
    depth = occupied.argmax(axis=2)
    hit = np.take_along_axis(rays(ids,q),depth[:,:,None],axis=2)[:,:,0]
    hit[~occupied.any(axis=2)] = 0
    ux = np.floor((np.arange(canvas)+.5-view["shift"][0])*n/canvas).astype(int)
    vy = np.floor((np.arange(canvas)+.5-view["shift"][1])*n/canvas).astype(int)
    valid = (vy[:,None]>=0)&(vy[:,None]<n)&(ux[None,:]>=0)&(ux[None,:]<n)
    sample = np.ix_(np.clip(vy,0,n-1),np.clip(ux,0,n-1))
    return np.where(valid,hit[sample],0), np.where(valid,(depth[sample]+.5)*canvas/n,np.inf)


def source_image(view):
    return np.array(view["pixels"],np.uint8).reshape(64,64)


def reference_labels(manifest, yaw):
    if yaw == 270:
        return reference_labels(manifest,90)[:,::-1].copy()
    im = Image.new("L",(64,64)); draw = ImageDraw.Draw(im)
    for region in manifest["reference_regions"][str(yaw)]:
        draw.polygon([tuple(p) for p in region["polygon"]],fill=region["group"])
    return np.asarray(im)


def pose_parts(manifest, pose):
    parts = copy.deepcopy(manifest["parts"])
    weapon = next(p for p in parts if p['name']=='weapon')
    movement = np.array(manifest['weapon_centres'][pose])-np.array(weapon['center'])
    weapon['center'] = manifest['weapon_centres'][pose]
    for part in parts:
        if part['name'].endswith('_hand'):
            part['center'] = (np.array(part['center'])+movement).tolist()
        if part['name'].endswith('_forearm'):
            part['b'] = (np.array(part['b'])+movement).tolist()
    dims = manifest["leg_dimensions"]
    for side, points in manifest["leg_poses"][pose].items():
        hip,knee,ankle,foot = points
        parts += [
            dict(name=side+"_thigh",group=6,shape="capsule",a=hip,b=knee,radius=dims["thigh_radius"],material=6),
            dict(name=side+"_shin",group=6,shape="capsule",a=knee,b=ankle,radius=dims["shin_radius"],material=9),
            dict(name=side+"_foot",group=6,shape="ellipsoid",center=foot,radii=dims["foot_radii"],power=3,material=9)]
    return parts


def volumes(parts, n):
    y,z,x = np.indices((n,n,n),dtype=float)
    positions = np.stack((x,y,z),axis=-1)*64/n + 32/n
    ids = np.zeros((n,n,n),np.uint8)
    fields = []
    normals = np.zeros((n,n,n,3),float)
    for number,part in enumerate(parts,1):
        if part["shape"] == "ellipsoid":
            local = positions-np.array(part["center"])
            local[:,:,:,2] -= part.get('depth_tilt',0)*local[:,:,:,1]
            p = local/np.array(part["radii"])
            field = np.sum(np.abs(p)**part["power"],axis=-1)**(1/part["power"])
        elif part["shape"] == "capsule":
            a,b = np.array(part["a"]),np.array(part["b"]); ab = b-a
            t = np.clip(np.sum((positions-a)*ab,axis=-1)/np.dot(ab,ab),0,1)
            field = np.linalg.norm(positions-a-t[:,:,:,None]*ab,axis=-1)/part["radius"]
        else:
            raise ValueError("Unknown part primitive "+part["shape"])
        fields.append(field)
        occupied = field <= 1
        # Later explicit parts own overlapping cells (hands/weapon over chest).
        ids[occupied] = number
        gy,gz,gx = np.gradient(field)
        normal = np.stack((gx,gy,gz),axis=-1)
        normal /= np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12)
        normals[occupied] = normal[occupied]
    return ids, normals, np.array(fields)


def bake(solid, parts_map, normals, parts, views, manifest):
    """First-hit projection with semantic ownership; unobserved material is explicit.

    No image is projected through an occluder and no flood-fill crosses a part.
    Surface colour is selected once from original palette samples, not RGB means.
    """
    n = solid.shape[0]
    colour = np.zeros(solid.shape,np.uint8)
    owner = np.zeros(solid.shape,np.uint8)
    confidence = np.zeros(solid.shape,np.float32)
    best = np.full(solid.shape,-np.inf)
    group = np.zeros(solid.shape,np.uint8)
    for i,p in enumerate(parts,1):
        region = solid & (parts_map==i)
        colour[region] = p["material"]
        group[region] = p["group"]
    yy,zz,xx = np.indices(solid.shape)
    ids = np.arange(1,solid.size+1).reshape(solid.shape)
    directions = ((0,0,-1),(-1,0,0),(0,0,1),(1,0,0))
    rejected = {}
    for view in sorted(views,key=lambda v:v["yaw"]):
        q = view["yaw"]//90
        hit,depth = first_hits(solid,view)
        u = (xx,n-1-zz,n-1-xx,zz)[q]
        d = (zz,xx,n-1-zz,n-1-xx)[q]
        px = np.floor((u+.5)*64/n).astype(int)+view["shift"][0]
        py = np.floor((yy+.5)*64/n).astype(int)+view["shift"][1]
        valid = solid & (px>=0)&(px<64)&(py>=0)&(py<64)
        px,py = np.clip(px,0,63),np.clip(py,0,63)
        pixels = source_image(view)
        visible = valid & (hit[py,px]==ids) & (np.abs(depth[py,px]-(d+.5)*64/n)<1e-5)
        semantic = reference_labels(manifest,view["yaw"])[py,px] == group
        rejected[str(view["yaw"])] = int(np.count_nonzero(visible & (pixels[py,px]!=0) & ~semantic))
        valid = visible & semantic & (pixels[py,px]!=0)
        facing = np.maximum(np.sum(normals*np.array(directions[q]),axis=-1),0)
        # Analytic part normals give coherent face patches instead of cell-edge noise.
        score = facing * view["weight"]
        take = valid & (score>best) & (score>0.05)
        colour[take] = pixels[py[take],px[take]]
        owner[take] = q+1
        confidence[take] = score[take]
        best[take] = score[take]
    if np.any(solid & (colour==0)):
        raise ValueError("Occupied part lacks a palette material")
    return colour,owner,confidence,rejected


def legacy_bake(grid, views):
    """Replay the previous C material bake and verify provenance against export."""
    solid = grid!=0; n=grid.shape[0]
    yy,zz,xx=np.indices(grid.shape); coords=(xx,n-1-zz,n-1-xx,zz)
    pad=np.pad(solid.astype(int),2)
    nx=np.zeros(grid.shape,int); nz=nx.copy()
    for dy in range(-2,3):
        for d in range(-2,3):
            nx += pad[2+dy:2+dy+n,2+d:2+d+n,0:n]-pad[2+dy:2+dy+n,2+d:2+d+n,4:4+n]
            nz += pad[2+dy:2+dy+n,0:n,2+d:2+d+n]-pad[2+dy:2+dy+n,4:4+n,2+d:2+d+n]
    colour=np.zeros(grid.shape,np.uint8); owner=colour.copy(); score=np.full(grid.shape,-1000)
    ident=np.arange(1,grid.size+1).reshape(grid.shape)
    for view in sorted(views,key=lambda v:v['yaw']):
        q=view['yaw']//90; vr=rays(solid,q)
        hits=np.take_along_axis(rays(ident,q),vr.argmax(axis=2)[:,:,None],axis=2)[:,:,0]
        visible=solid & (hits[yy,coords[q]]==ident)
        u=np.floor((coords[q]+.5)*64/n).astype(int)+view['shift'][0]
        v=np.floor((yy+.5)*64/n).astype(int)+view['shift'][1]
        pixels=source_image(view)
        sampled=np.zeros(grid.shape,np.uint8); dist=np.full(grid.shape,10)
        for dy in range(-3,4):
            for dx in range(-3,4):
                distance=dx*dx+dy*dy
                if distance>=10: continue
                px,py=u+dx,v+dy
                valid=(px>=0)&(px<64)&(py>=0)&(py<64)
                c=pixels[np.clip(py,0,63),np.clip(px,0,63)]
                take=valid & (c!=0) & (distance<dist)
                sampled[take]=c[take]; dist[take]=distance
        value=(-nz,-nx,nz,nx)[q]
        take=visible & (value>score)
        colour[take]=sampled[take]; owner[take]=q+1; score[take]=value[take]
    # Preserve seed order and neighbour order of colour_hull in voxel_build.c.
    flat=colour.ravel(); own=owner.ravel(); occupied=solid.ravel()
    queue=deque(np.flatnonzero(flat).tolist())
    while queue:
        p=queue.popleft(); x=p%n; z=p//n%n; y=p//(n*n)
        for valid,offset in ((x>0,-1),(x+1<n,1),(z>0,-n),(z+1<n,n),(y>0,-n*n),(y+1<n,n*n)):
            other=p+offset
            if valid and occupied[other] and not flat[other]:
                flat[other]=flat[p]; own[other]=own[p]; queue.append(other)
    if not np.array_equal(colour,grid):
        raise AssertionError(f"Legacy bake replay differs in {np.count_nonzero(colour!=grid)} cells")
    direct=score>-1000
    return owner,np.where(direct,1.,0.)


def trace(grid,yaw,pitch=0,size=96):
    """Independent vectorised 3D orthographic DDA, used only by diagnostics."""
    n=grid.shape[0]; angle=math.radians(yaw); elev=math.radians(pitch)
    sn,cs=math.sin(angle),math.cos(angle); sp,cp=math.sin(elev),math.cos(elev)
    forward=np.array([sn*cp,sp,cs*cp]); forward[np.abs(forward)<1e-10]=0
    right=np.array([cs,0,-sn]); down=np.array([-sn*sp,cp,-cs*sp])
    row,col=np.indices((size,size)); factor=80/size
    origin=32+(col[:,:,None]+.5-size/2)*factor*right+(row[:,:,None]+.5-size/2)*factor*down-forward*150
    origin=origin.reshape(-1,3)
    low=np.full_like(origin,-np.inf); high=np.full_like(origin,np.inf)
    for axis in range(3):
        if forward[axis]:
            a=-origin[:,axis]/forward[axis]; b=(64-origin[:,axis])/forward[axis]
            low[:,axis]=np.minimum(a,b); high[:,axis]=np.maximum(a,b)
        else:
            outside=(origin[:,axis]<0)|(origin[:,axis]>=64)
            low[outside,axis]=np.inf
    entry=low.max(axis=1); end=high.min(axis=1)
    active=np.flatnonzero(entry<end)
    cells=np.zeros_like(origin,dtype=int)
    cells[active]=np.clip(np.floor((origin[active]+(entry[active,None]+1e-7)*forward)*n/64).astype(int),0,n-1)
    sign=np.where(forward>=0,1,-1)
    next_t=np.full_like(origin,np.inf); delta=np.full(3,np.inf)
    for axis in range(3):
        if forward[axis]:
            next_t[:,axis]=((cells[:,axis]+(sign[axis]>0))*64/n-origin[:,axis])/forward[axis]
            delta[axis]=abs(64/n/forward[axis])
    face=np.argmax(low,axis=1)
    hit=np.zeros(size*size,np.int32); depth=np.zeros(size*size); normal=np.zeros((size*size,3))
    while len(active):
        x,y,z=cells[active].T
        filled=grid[y,z,x]!=0
        found=active[filled]
        hit[found]=(cells[found,1]*n+cells[found,2])*n+cells[found,0]+1
        depth[found]=entry[found]
        normal[found,face[found]]=-sign[face[found]]
        active=active[~filled]
        if not len(active): break
        time=next_t[active].min(axis=1)
        cross=next_t[active]<=time[:,None]+1e-9
        face[active]=np.argmax(cross,axis=1)
        cells[active]+=cross*sign
        for axis in range(3):
            if np.isfinite(delta[axis]): next_t[active[cross[:,axis]],axis]+=delta[axis]
        entry[active]=time
        active=active[(time<end[active])&np.all((cells[active]>=0)&(cells[active]<n),axis=1)]
    return hit.reshape(size,size),depth.reshape(size,size),normal.reshape(size,size,3)


def palette_rgb(palette):
    p=np.asarray(palette,dtype=np.uint32)
    return np.stack(((p>>16)&255,(p>>8)&255,p&255),axis=-1).astype(np.uint8)


ID_COLOURS=np.array([[210,50,190],[230,100,50],[80,190,230],[220,190,50],[110,220,90],
                     [210,100,210],[100,120,240],[210,150,80],[80,200,170]],np.uint8)
PART_COLOURS=np.array([[28,30,36]]+[[round(c*255) for c in colorsys.hsv_to_rgb((i*.61803398875)%1,.65,.9)]
                                  for i in range(32)],np.uint8)


def diagnostic(grid,palette,yaw,pitch=0,mode='colour',parts=None,owner=None,confidence=None):
    ids,depth,normal=trace(grid,yaw,pitch)
    mask=ids!=0; ix=np.maximum(ids-1,0)
    if mode=='colour': pixels=palette_rgb(palette)[grid.ravel()[ix]]
    elif mode=='neutral':
        light=np.array([-.4,-.75,-.5]); light/=np.linalg.norm(light)
        shade=70+170*np.maximum(np.sum(normal*light,axis=-1),0)
        pixels=np.repeat(shade[:,:,None],3,axis=2).astype(np.uint8)
    elif mode=='normals': pixels=((normal+1)*127).astype(np.uint8)
    elif mode=='depth':
        value=np.clip((190-depth)*255/80,0,255).astype(np.uint8)
        pixels=np.repeat(value[:,:,None],3,axis=2)
    elif mode=='parts': pixels=PART_COLOURS[parts.ravel()[ix]]
    elif mode=='owner': pixels=ID_COLOURS[owner.ravel()[ix]%len(ID_COLOURS)]
    elif mode=='confidence':
        value=np.clip(confidence.ravel()[ix]*255,0,255).astype(np.uint8)
        pixels=np.stack((255-value,value,np.zeros_like(value)),axis=-1)
    else: raise ValueError(mode)
    pixels[~mask]=(28,30,36)
    return Image.fromarray(pixels).resize((192,192),Image.Resampling.NEAREST)


def chunk(name,content):
    return name+struct.pack('<II',len(content),0)+content


def write_vox(path,grid,palette,metadata):
    n=grid.shape[0]
    y,z,x=np.nonzero(grid)
    cells=np.stack((x,z,n-1-y,grid[y,z,x]),axis=-1).astype(np.uint8)
    colours=np.zeros((256,4),np.uint8)
    colours[:31,:3]=palette_rgb(palette)[1:32]; colours[:31,3]=255
    body=chunk(b'SIZE',struct.pack('<iii',n,n,n))+chunk(b'XYZI',struct.pack('<I',len(cells))+cells.tobytes())+chunk(b'RGBA',colours.tobytes())
    path.write_bytes(b'VOX '+struct.pack('<I',150)+b'MAIN'+struct.pack('<II',0,len(body))+body)
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')


def read_vox(path):
    data=path.read_bytes()
    if data[:8]!=b'VOX '+struct.pack('<I',150) or data[8:12]!=b'MAIN': raise ValueError('Expected VOX version 150')
    offset=20; grid=None; palette=None
    while offset<len(data):
        tag=data[offset:offset+4]; length,children=struct.unpack_from('<II',data,offset+4)
        body=data[offset+12:offset+12+length]
        if tag==b'SIZE':
            sx,sz,sy=struct.unpack('<iii',body)
            grid=np.zeros((sy,sz,sx),np.uint8)
        elif tag==b'XYZI':
            count=struct.unpack_from('<I',body)[0]
            cells=np.frombuffer(body[4:],np.uint8).reshape(count,4)
            x,z,y,c=cells.T
            grid[grid.shape[0]-1-y,z,x]=c
        elif tag==b'RGBA':
            p=np.frombuffer(body,np.uint8).reshape(256,4)
            palette=np.concatenate((np.zeros((1,3),np.uint8),p[:255,:3]))
        offset+=12+length+children
    if grid is None or palette is None: raise ValueError('Missing model/palette chunks')
    return grid,palette,json.loads(path.with_suffix('.json').read_text(encoding='utf8'))


def strict_hull(views,n):
    y,z,x=np.indices((n,n,n)); solid=np.ones((n,n,n),bool)
    for view in views:
        q=view['yaw']//90; u=(x,n-1-z,n-1-x,z)[q]
        px=np.floor((u+.5)*64/n).astype(int)+view['shift'][0]
        py=np.floor((y+.5)*64/n).astype(int)+view['shift'][1]
        valid=(px>=0)&(px<64)&(py>=0)&(py<64)
        solid &= valid & (source_image(view)[np.clip(py,0,63),np.clip(px,0,63)]!=0)
    return solid.astype(np.uint8)


def metrics(grid,palette,views):
    output=[]; pal=palette_rgb(palette).astype(float)
    for view in views:
        ids,_=first_hits(grid!=0,view); source=source_image(view)
        render=np.where(ids!=0,grid.ravel()[np.maximum(ids-1,0)],0)
        a=source!=0; b=render!=0; covered=a&b
        output.append(dict(yaw=view['yaw'],frame=view['frame'],iou=float(np.count_nonzero(covered)/max(1,np.count_nonzero(a|b))),
                           missing=int(np.count_nonzero(a&~b)),excess=int(np.count_nonzero(b&~a)),
                           palette_exact=float(np.mean(source[covered]==render[covered])) if covered.any() else 0,
                           colour_mae=float(np.abs(pal[source[covered]]-pal[render[covered]]).mean()) if covered.any() else 0))
    return output


def contact_sheet(path,rows,palette,angles=None):
    angles=angles or [(i*45,0) for i in range(8)]
    image=Image.new('RGB',(len(angles)*192,len(rows)*220),(20,22,27)); draw=ImageDraw.Draw(image)
    for row,(title,grid,mode,extras) in enumerate(rows):
        draw.text((4,row*220+3),title,fill='white')
        for column,(yaw,pitch) in enumerate(angles):
            tile=diagnostic(grid,palette,yaw,pitch,mode,**extras)
            image.paste(tile,(column*192,row*220+28))
            draw.text((column*192,row*220+16),f'yaw {yaw}, elevation {pitch}',fill='#bbbbbb')
    image.save(path)


def fnv(data):
    value=2166136261
    for b in data: value=((value^int(b))*16777619)&0xffffffff
    return value


def reference_sheet(path,grid,views,palette,manifest,pose):
    image=Image.new('RGB',(4*256,5*280),(20,22,27)); draw=ImageDraw.Draw(image)
    pal=palette_rgb(palette)
    parts={p['name']:p for p in pose_parts(manifest,pose)}
    for column,view in enumerate(views):
        source=source_image(view); ids,depth=first_hits(grid!=0,view)
        render=np.where(ids!=0,grid.ravel()[np.maximum(ids-1,0)],0)
        labels=reference_labels(manifest,view['yaw'])
        labelled=pal[source].copy(); mask=source!=0
        labelled[mask]=((labelled[mask].astype(int)+ID_COLOURS[labels[mask]%len(ID_COLOURS)].astype(int))//2).astype(np.uint8)
        difference=np.zeros((64,64,3),np.uint8)+[28,30,36]
        difference[(source!=0)&(ids!=0)]=(180,180,180)
        difference[(source!=0)&(ids==0)]=(0,255,255)
        difference[(source==0)&(ids!=0)]=(255,0,255)
        hit_ids=ID_COLOURS[ids%len(ID_COLOURS)]; hit_ids[ids==0]=(28,30,36)
        for row,(title,pixels) in enumerate((('Original + observed landmarks',pal[source]),('Editable material regions',labelled),
                                           ('Same camera + projected landmarks',pal[render]),('Missing cyan / excess magenta',difference),
                                           ('First-hit voxel IDs (hashed colours)',hit_ids))):
            tile=Image.fromarray(np.asarray(pixels,np.uint8)).resize((256,256),Image.Resampling.NEAREST)
            pen=ImageDraw.Draw(tile)
            if row in (0,2):
                for point in manifest['landmarks']:
                    if point['annotated_pose']!=pose: continue
                    observed=point['views'].get(str(view['yaw']))
                    if observed is None: continue
                    x,y,z=np.array(parts[point['part']]['center'])+np.array(point['offset']); q=view['yaw']//90
                    px=(x,64-z,64-x,z)[q]+view['shift'][0]; py=y+view['shift'][1]
                    if row==0: px,py=observed
                    pen.line((px*4-4,py*4,px*4+4,py*4),fill='#ff4488',width=1)
                    pen.line((px*4,py*4-4,px*4,py*4+4),fill='#ff4488',width=1)
            image.paste(tile,(column*256,row*280+24))
            draw.text((column*256,row*280+2),f"{title} / {view['yaw']}",fill='white')
    image.save(path)


def slice_sheet(path,parts):
    n=parts.shape[0]; image=Image.new('RGB',(8*192,3*220),(20,22,27)); draw=ImageDraw.Draw(image)
    for axis in range(3):
        for col in range(8):
            k=(2*col+1)*n//16
            plane=(parts[k,:,:] if axis==0 else parts[:,:,k] if axis==1 else parts[:,k,:])
            pixels=PART_COLOURS[plane].copy()
            image.paste(Image.fromarray(pixels).resize((192,192),Image.Resampling.NEAREST),(col*192,axis*220+28))
            draw.text((col*192,axis*220+5),f"{'Y horizontal' if axis==0 else 'X vertical' if axis==1 else 'Z vertical'} = {(k+.5)*64/n:.1f}",fill='white')
    image.save(path)


def generated_header(path,models,manifest_hash):
    lines=['/* Generated by tools/voxel_author.py; edit assets/voxels/marine.json. */',
           '#ifndef AB3D_VOXEL_MARINE_DATA_H','#define AB3D_VOXEL_MARINE_DATA_H',
           f'#define MARINE_VOXEL_GRID {models[0]["grid"].shape[0]}',
           f'/* Correction manifest SHA256: {manifest_hash} */']
    for pose,m in enumerate(models):
        columns,runs=pack(m['grid'])
        lines.append(f'static const uint32_t marine_columns_{pose}[] = {{')
        for i in range(0,len(columns),24): lines.append(','.join(map(str,columns[i:i+24]))+',')
        lines.append('};')
        lines.append(f'static const VoxelRun marine_runs_{pose}[] = {{')
        for i in range(0,len(runs),12): lines.append(','.join('{'+','.join(map(str,r))+'}' for r in runs[i:i+12])+',')
        lines.append('};')
    lines += ['typedef struct { const uint32_t *columns; const VoxelRun *runs; uint32_t count; uint32_t source_hash[4]; float iou[4]; uint32_t fp[4],fn[4]; VoxelSource sources[4]; } MarineVoxelData;',
              'static const MarineVoxelData marine_voxel_data[4] = {']
    for p,m in enumerate(models):
        _,runs=pack(m['grid']); v=m['views']; scores=m['metrics']
        array=lambda seq:'{'+','.join(map(str,seq))+'}'
        lines.append('{'+f'marine_columns_{p},marine_runs_{p},{len(runs)},'+array(f'0x{fnv(x["pixels"]):08x}u' for x in v)+','+
                     array(f'{s["iou"]:.9f}f' for s in scores)+','+array(s['excess'] for s in scores)+','+array(s['missing'] for s in scores)+','+
                     array(array([view['yaw']//90,*view['shift'],view['mirror'],f"{float(view['weight']):.3f}f",view['frame']]) for view in v)+'},')
    lines += ['};','#endif','']
    path.write_text('\n'.join(lines),encoding='utf8')


def run(args):
    manifest_path=Path(args.manifest); manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest['version']!=1 or manifest['canvas']!=[64,64] or manifest['grid']!=48:
        raise ValueError('This marine correction workflow requires version 1, a 64x64 source canvas and the frozen 48-cell grid')
    target=Path(args.output); target.mkdir(parents=True,exist_ok=True)
    models=[]; report={'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'poses':[]}
    for pose in range(4):
        snapshot_path=Path(args.snapshots)/f'marine-{pose}-baseline.json'
        snapshot=json.loads(snapshot_path.read_text(encoding='utf8'))
        baseline=unpack(snapshot); views=copy.deepcopy(sorted(snapshot['views'],key=lambda v:v['yaw'])); palette=snapshot['palette']
        views[2]['shift']=manifest['back_registration'][pose]
        parts=pose_parts(manifest,pose)
        part_map,normals,fields=volumes(parts,manifest['grid'])
        corrected,owner,confidence,rejected=bake(part_map!=0,part_map,normals,parts,views,manifest)
        reference_sheet(target/f'marine-{pose}-references.png',corrected,views,palette,manifest,pose)
        slice_sheet(target/f'marine-{pose}-slices.png',part_map)
        # Control experiment: repaint the frozen geometry without changing a cell.
        closest=(fields.argmin(axis=0)+1).astype(np.uint8)
        ny,nz,nx=baseline.shape
        approx_normals=np.zeros((*baseline.shape,3))
        for i,field in enumerate(fields,1):
            gy,gz,gx=np.gradient(field); normal=np.stack((gx,gy,gz),axis=-1)
            normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12)
            approx_normals[closest==i]=normal[closest==i]
        recoloured,_,_,_=bake(baseline!=0,closest,approx_normals,parts,views,manifest)
        legacy_owner,legacy_confidence=legacy_bake(baseline,snapshot['views'])
        scores=metrics(corrected,palette,views)
        model=dict(grid=corrected,views=views,metrics=scores); models.append(model)
        metadata=dict(version=1,pose=pose,pivot=manifest['pivot'],canvas=manifest['canvas'],
                      coordinates='VOX(x,y,z) = runtime(x,z,ny-1-y)',parts=parts,
                      part_coordinate_units='source pixels on [0,64] cubed; Y down',grid_size=[nx,ny,nz],
                      source_to_vox_boundary=[[nx/64,0,0,0],[0,0,nz/64,0],[0,-ny/64,0,ny],[0,0,0,1]],
                      part_grid=part_map.ravel().tolist(),source_owner=owner.ravel().tolist(),
                      source_confidence=confidence.ravel().round(4).tolist(),
                      views=[{k:v for k,v in view.items() if k!='pixels'} for view in views])
        path=target/f'marine-{pose}.vox'; write_vox(path,corrected,palette,metadata)
        loaded,loaded_palette,loaded_meta=read_vox(path)
        assert np.array_equal(loaded,corrected) and np.array_equal(loaded_palette[1:32],palette_rgb(palette)[1:32])
        assert loaded_meta==metadata
        columns,runs=pack(loaded)
        assert np.array_equal(unpack(dict(size=[nx,ny,nz],columns=columns,runs=runs)),corrected)
        if args.verify_runtime:
            runtime=json.loads((Path(args.verify_runtime)/f'marine-{pose}-runtime.json').read_text(encoding='utf8'))
            if not np.array_equal(unpack(runtime),loaded): raise AssertionError(f'Runtime model {pose} differs from the exported/reloaded authoring model')
        item=dict(pose=pose,source_snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                  baseline=metrics(baseline,palette,views),colour_only=metrics(recoloured,palette,views),corrected=scores,
                  semantic_rejections=rejected,occupied=int(np.count_nonzero(corrected)),
                  directly_observed=int(np.count_nonzero(owner)),inferred_material=int(np.count_nonzero(corrected)-np.count_nonzero(owner)))
        report['poses'].append(item)
        extras=dict(parts=part_map,owner=owner,confidence=confidence)
        contact_sheet(target/f'marine-{pose}-comparison.png',[
            ('Frozen runtime geometry / original fixed colour',baseline,'colour',{}),
            ('Same geometry / first-hit + part ownership colouring',recoloured,'colour',{}),
            ('Edited parts / same first-hit + ownership colouring',loaded,'colour',{})],palette)
        contact_sheet(target/f'marine-{pose}-geometry.png',[
            ('Strict undilated visual hull / diagnostic only',strict_hull(views,nx),'neutral',{}),
            ('Frozen runtime geometry / neutral material',baseline,'neutral',{}),
            ('Edited parts / neutral material',loaded,'neutral',{})],palette)
        contact_sheet(target/f'marine-{pose}-diagnostics.png',[
            ('Unlit palette / exported and reloaded object',loaded,'colour',{}),
            ('Neutral material / fixed world light',loaded,'neutral',{}),
            ('World surface normals',loaded,'normals',{}),
            ('Camera depth (diagnostic only)',loaded,'depth',{}),
            ('Part IDs',loaded,'parts',extras),
            ('Colour source: front orange / right blue / back yellow / left green / prior magenta',loaded,'owner',extras),
            ('Colour confidence: green observed / red explicit part material',loaded,'confidence',extras)],palette,
            angles=[(0,0),(45,0),(90,0),(135,0),(180,0),(270,0),(45,25),(0,90)])
        contact_sheet(target/f'marine-{pose}-provenance-before.png',[
            ('Previous source ownership (replayed exactly from C bake)',baseline,'owner',dict(owner=legacy_owner)),
            ('Previous direct projection green / propagated colour red',baseline,'confidence',dict(confidence=legacy_confidence))],palette)
        print(f'pose {pose}: {item["occupied"]} voxels; source IoUs '+', '.join(f'{s["iou"]:.3f}' for s in scores))
    barrel=json.loads((Path(args.snapshots)/'barrel-0-baseline.json').read_text(encoding='utf8'))
    barrel_grid=unpack(barrel)
    write_vox(target/'barrel.vox',barrel_grid,barrel['palette'],dict(pivot=[32,64,32],pose=0,coordinates='VOX(x,y,z) = runtime(x,z,ny-1-y)'))
    assert np.array_equal(read_vox(target/'barrel.vox')[0],barrel_grid)
    contact_sheet(target/'barrel.png',[('Existing barrel / unchanged geometry',barrel_grid,'neutral',{}),
                                      ('Existing barrel / unchanged palette',barrel_grid,'colour',{})],barrel['palette'],
                  angles=[(0,0),(22.5,0),(45,0),(90,0),(135,0),(180,0),(45,25),(0,90)])
    (target/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if args.runtime_header: generated_header(Path(args.runtime_header),models,report['manifest_sha256'])
    html=['<!doctype html><meta charset="utf-8"><title>Marine parts and visibility audit</title>',
          '<style>body{background:#14161b;color:#eee;font:16px system-ui;margin:24px}img{max-width:100%;image-rendering:pixelated}a{color:#8cf}</style>',
          '<h1>Marine parts and visibility audit</h1><p>Fixed cameras and scale; no silhouette score certifies anatomy. Geometry controls use neutral material; colour controls freeze geometry. Magenta source IDs and red confidence mark inferred part materials.</p>',
          '<p><a href="metrics.json">Per-view silhouette and colour errors</a>. VOX exports have sidecars with origin, pose, part IDs, correction data and source ownership. Exported and reloaded geometry is used for every corrected preview. No external editor was used.</p>']
    for pose in range(4):
        html += [f'<h2>Walking pose {pose}</h2><p><a href="marine-{pose}.vox">VOX model</a> / <a href="marine-{pose}.json">Origin and correction data</a></p>']
        html.append('<p>Part colours: '+', '.join(f'<span style="color:#{int(PART_COLOURS[i,0]):02x}{int(PART_COLOURS[i,1]):02x}{int(PART_COLOURS[i,2]):02x}">{i}: {p["name"]}</span>'
                                              for i,p in enumerate(pose_parts(manifest,pose),1))+'</p>')
        for suffix in ('references','comparison','geometry','diagnostics','slices','provenance-before'):
            html.append(f'<h3>{suffix}</h3><img loading="lazy" src="marine-{pose}-{suffix}.png">')
    html += ['<h2>Barrel</h2><a href="barrel.vox">Unchanged barrel export</a><br><img src="barrel.png">']
    (target/'index.html').write_text('\n'.join(html),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots',required=True)
    parser.add_argument('--manifest',default='assets/voxels/marine.json')
    parser.add_argument('--output',default='build/voxel-authoring')
    parser.add_argument('--runtime-header')
    parser.add_argument('--verify-runtime',help='Compare exported/reloaded models with C runtime snapshots in this directory')
    run(parser.parse_args())
