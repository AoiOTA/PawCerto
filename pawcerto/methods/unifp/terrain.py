# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES; ETH Zurich, Nikita Rudin.
"""Original UniFP active rough-flat terrain and Delatin conversion.

The active source terrain has 10x20 rough-flat tiles and a 25 m zero border.
RectBivariateSpline(kx=1, ky=1) is SciPy's regular-grid replacement for removed
interp2d(kind='linear'); quantization and rounding retain Gym's operation order.
"""
import random
import numpy as np
from scipy.interpolate import RectBivariateSpline


class UniFPTerrain:
    def __init__(self, cfg):
        from pydelatin import Delatin
        self.cfg = cfg
        proportions = np.asarray(cfg.terrain_proportions)
        if cfg.mesh_type != 'trimesh' or cfg.hf2mesh_method != 'fast' or np.count_nonzero(proportions) != 1 or proportions[8] != 1:
            raise ValueError('This port implements the original active rough-flat/Delatin terrain configuration')
        pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        width = int(cfg.terrain_width / cfg.horizontal_scale)
        border = int(cfg.border_size / cfg.horizontal_scale)
        self.height_field_raw = np.zeros((cfg.num_rows*pixels+2*border, cfg.num_cols*width+2*border), dtype=np.int16)
        self.env_origins = np.zeros((cfg.num_rows,cfg.num_cols,3))
        for j in range(cfg.num_cols):
            for i in range(cfg.num_rows):
                # Upstream consumes a difficulty draw but calls add_roughness
                # with its default difficulty=1 for rough-flat terrain.
                if not cfg.curriculum:
                    np.random.uniform(0,1)
                height = random.uniform(cfg.height[0], cfg.height[1])
                minimum, maximum = int(-height/cfg.vertical_scale), int(height/cfg.vertical_scale)
                step = int(.005/cfg.vertical_scale)
                down = np.random.choice(np.arange(minimum,maximum+step,step),
                    (int(width*cfg.horizontal_scale/cfg.downsampled_scale),
                     int(width*cfg.horizontal_scale/cfg.downsampled_scale)))
                x = np.linspace(0,width*cfg.horizontal_scale,down.shape[0])
                y = np.linspace(0,width*cfg.horizontal_scale,down.shape[1])
                up = np.linspace(0,width*cfg.horizontal_scale,width)
                tile = np.rint(RectBivariateSpline(x,y,down,kx=1,ky=1)(up,up)).astype(np.int16)
                self.height_field_raw[border+i*pixels:border+(i+1)*pixels,border+j*width:border+(j+1)*width] = tile
                a,b = int((cfg.terrain_length/2-1)/cfg.horizontal_scale), int((cfg.terrain_length/2+1)/cfg.horizontal_scale)
                z = 0 if cfg.origin_zero_z else tile[a:b,a:b].max()*cfg.vertical_scale
                self.env_origins[i,j] = ((i+.5)*cfg.terrain_length,(j+.5)*cfg.terrain_width,z)
        self.heightsamples = self.height_field_raw
        mesh = Delatin(np.flip(self.height_field_raw,axis=1).T,z_scale=cfg.vertical_scale,max_error=cfg.max_error)
        self.vertices = np.array(mesh.vertices,copy=True)
        self.vertices[:,:2] *= cfg.horizontal_scale
        self.triangles = np.asarray(mesh.triangles)


def make_terrain_cfg(terrain):
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import get_current_stage
    from pxr import UsdGeom, UsdPhysics, UsdShade, Vt
    path = '/World/UniFPTerrain'
    def spawn(prim_path, cfg, translation=None, orientation=None):
        stage = get_current_stage()
        root = UsdGeom.Xform.Define(stage,prim_path)
        # PhysX BV4 cooking has a finite triangle count per collision shape.
        # Partition source faces, retaining each exactly once and rebasing only
        # vertex indices; geometry is shared by every environment.
        material = UsdShade.Material.Define(stage,prim_path+'/material')
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(terrain.cfg.static_friction)
        api.CreateDynamicFrictionAttr(terrain.cfg.dynamic_friction)
        api.CreateRestitutionAttr(terrain.cfg.restitution)
        for index, begin in enumerate(range(0,len(terrain.triangles),250000)):
            faces = terrain.triangles[begin:begin+250000]
            ids, inverse = np.unique(faces.reshape(-1),return_inverse=True)
            vertices = terrain.vertices[ids].copy()
            vertices[:,:2] -= terrain.cfg.border_size
            mesh = UsdGeom.Mesh.Define(stage,f'{prim_path}/mesh_{index:03d}')
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(inverse.astype(np.int32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,dtype=np.int32)))
            mesh.CreateSubdivisionSchemeAttr('none')
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,materialPurpose='physics')
        return root.GetPrim()
    return AssetBaseCfg(prim_path=path,collision_group=-1,spawn=SpawnerCfg(func=spawn)), [f'{path}/mesh_{i:03d}' for i in range((len(terrain.triangles)+249999)//250000)]
