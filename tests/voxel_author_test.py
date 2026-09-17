"""Small known-answer fixtures for the offline voxel authoring pipeline."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import voxel_author as author


class AuthoringTests(unittest.TestCase):
    def fixture(self):
        solid=np.zeros((8,8,8),bool)
        solid[2:6,1:3,2:6]=True
        solid[2:6,5:7,2:6]=True
        parts=np.zeros(solid.shape,np.uint8)
        parts[:,1:3,:]=1; parts[:,5:7,:]=2
        normals=np.zeros((*solid.shape,3)); normals[:,:,:,2]=-1
        definitions=[dict(group=1,material=3),dict(group=1,material=14)]
        source=np.ones((64,64),np.uint8)  # opaque BLACK palette index 1
        view=dict(yaw=0,shift=[0,0],weight=1,pixels=source.ravel().tolist(),frame=0)
        regions=[dict(group=1,polygon=[[0,0],[63,0],[63,63],[0,63]])]
        manifest=dict(reference_regions={str(q):regions for q in (0,90,180)})
        return solid,parts,normals,definitions,view,manifest

    def test_foreground_cannot_paint_hidden_block(self):
        solid,parts,normals,definitions,view,manifest=self.fixture()
        colour,owner,_,_=author.bake(solid,parts,normals,definitions,[view],manifest)
        self.assertTrue(np.all(colour[2:6,1,2:6]==1))
        self.assertTrue(np.all(owner[2:6,1,2:6]==1))
        self.assertTrue(np.all(colour[2:6,5:7,2:6]==14))
        self.assertFalse(np.any(owner[2:6,5:7,2:6]))
        manifest['reference_regions']['0'][0]['group']=2
        rejected=author.bake(solid,parts,normals,definitions,[view],manifest)
        self.assertFalse(np.any(rejected[1]))
        self.assertGreater(rejected[3]['0'],0)

    def test_camera_ids_and_source_order(self):
        solid,parts,normals,definitions,view,manifest=self.fixture()
        # An asymmetric extension makes a left/right camera error observable.
        solid[3:5,3:5,1:3]=True; parts[3:5,3:5,1:3]=1
        views=[dict(view,yaw=q*90,pixels=[q+1]*4096) for q in range(4)]
        for q in range(4):
            ids,_=author.first_hits(solid,views[q])
            independent,_,_=author.trace(solid,q*90,size=80)
            np.testing.assert_array_equal(ids,independent[8:72,8:72])
        expected=author.bake(solid,parts,normals,definitions,views,manifest)
        for permuted in itertools.permutations(views):
            actual=author.bake(solid,parts,normals,definitions,permuted,manifest)
            np.testing.assert_array_equal(actual[0],expected[0])
            np.testing.assert_array_equal(actual[1],expected[1])

    def test_vox_and_runtime_runs_round_trip(self):
        grid=np.zeros((8,8,8),np.uint8)
        grid[1:7,2:4,3:6]=1; grid[5,1,4]=25
        palette=[0]+[i*0x030507 for i in range(1,32)]
        metadata=dict(pivot=[32,64,32],pose=3,parts=[dict(name='asymmetric fixture',translation=[1,2,3])])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.vox'
            author.write_vox(path,grid,palette,metadata)
            reloaded,colours,meta=author.read_vox(path)
            np.testing.assert_array_equal(reloaded,grid)
            np.testing.assert_array_equal(colours[1:32],author.palette_rgb(palette)[1:32])
            self.assertEqual(meta,metadata)
            columns,runs=author.pack(reloaded)
            np.testing.assert_array_equal(author.unpack(dict(size=[8,8,8],columns=columns,runs=runs)),grid)

    def test_strict_intersection_is_monotone(self):
        views=[]; previous=np.ones((8,8,8),bool)
        for q in range(4):
            pixels=np.zeros((64,64),np.uint8); pixels[8:56,8+q:48+q]=1
            views.append(dict(yaw=q*90,shift=[0,0],pixels=pixels.ravel().tolist()))
            current=author.strict_hull(views,8)!=0
            self.assertFalse(np.any(current & ~previous))
            previous=current


if __name__=='__main__': unittest.main()
