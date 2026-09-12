# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""Pinned DeepWBC terrain, independent of Gym and Sim at import time.

Perlin routines retain the original operation order, including int16 overflow.
Mesh conversion is the BSD-3-Clause Isaac Lab implementation, revision
412fb31b30ee605b4ffec4327436fc0fe53281d8 (2022-2026 Isaac Lab developers).
See docs/deepwbc-terrain.md for source and runtime evidence boundaries.
"""
from types import SimpleNamespace
import numpy as np
from .config import default_config


def _config(cfg=None):
    if cfg is None:
        cfg = default_config()["terrain"]
    return SimpleNamespace(**cfg) if isinstance(cfg, dict) else cfg


def environment_origin_bounds(cfg=None):
    """World-frame uniform reset bounds, from WidowGo1._get_env_origins."""
    cfg = _config(cfg)
    half_x = cfg.tot_cols * cfg.horizontal_scale / 2
    half_y = cfg.tot_rows * cfg.horizontal_scale / 2
    return ((-2.5 * half_x / 5, -2 * half_x / 5), (-half_y + 10, half_y - 10))


def build_terrain(seed=1, cfg=None):
    """Build the entire original collision mesh on CPU using a local MT19937 RNG."""
    return Terrain_Perlin(cfg, seed)


class Terrain_Perlin:
    def __init__(self, cfg=None, seed=1):
        cfg = _config(cfg)
        self.rng = np.random.RandomState(seed)
        self.cfg = cfg
        self.xSize = int(cfg.horizontal_scale * cfg.tot_cols)
        self.ySize = int(cfg.horizontal_scale * cfg.tot_rows)
        assert(self.xSize == cfg.horizontal_scale * cfg.tot_cols and self.ySize == cfg.horizontal_scale * cfg.tot_rows)
        self.tot_cols = cfg.tot_cols
        self.tot_rows = cfg.tot_rows
        self.heightsamples_float = self.generate_fractal_noise_2d(self.xSize, self.ySize, self.tot_cols, self.tot_rows, zScale=cfg.zScale)
        self.heightsamples_float[self.tot_cols//2 - 100:, :] += 100000
        # self.heightsamples_float[self.tot_cols//2 - 40: self.tot_cols//2 + 40, :] = np.mean(self.heightsamples_float)
        self.heightsamples = (self.heightsamples_float * (1 / cfg.vertical_scale)).astype(np.int16)
        

        # self.heightsamples = np.zeros((800, 800)).astype(np.int16)
        self.vertices, self.triangles = convert_height_field_to_mesh(   self.heightsamples,
                                                                                        cfg.horizontal_scale,
                                                                                        cfg.vertical_scale,
                                                                                        cfg.slope_treshold)
    
    def generate_perlin_noise_2d(self, shape, res):
        def f(t):
            return 6*t**5 - 15*t**4 + 10*t**3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2*np.pi*self.rng.rand(res[0]+1, res[1]+1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1) * 0.5 + 0.5
    
    def generate_fractal_noise_2d(self, xSize=20, ySize=20, xSamples=1600, ySamples=1600, \
        frequency=10, fractalOctaves=2, fractalLacunarity = 2.0, fractalGain=0.25, zScale = 0.23):
        xScale = frequency * xSize
        yScale = frequency * ySize
        amplitude = 1
        shape = (xSamples, ySamples)
        noise = np.zeros(shape)
        for _ in range(fractalOctaves):
            noise += amplitude * self.generate_perlin_noise_2d((xSamples, ySamples), (xScale, yScale)) * zScale
            amplitude *= fractalGain
            xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

        return noise

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
def convert_height_field_to_mesh(
    height_field: np.ndarray, horizontal_scale: float, vertical_scale: float, slope_threshold: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a height-field array to a triangle mesh represented by vertices and triangles.

    This function converts a height-field array to a triangle mesh represented by vertices and triangles.
    The height-field array is assumed to be a 2D array of floats, where each element represents the height
    of the terrain at that location. The height-field array is assumed to be in the form of a matrix, where
    the first dimension represents the x-axis and the second dimension represents the y-axis.

    The function can also correct vertical surfaces above the provide slope threshold. This is helpful to
    avoid having long vertical surfaces in the mesh. The correction is done by moving the vertices of the
    vertical surfaces to minimum of the two neighboring vertices.

    The correction is done in the following way:
    If :math:`\\frac{y_2 - y_1}{x_2 - x_1} > threshold`, then move A to A' (i.e., set :math:`x_1' = x_2`).
    This is repeated along all directions.

    .. code-block:: none

                B(x_2,y_2)
                    /|
                   / |
                  /  |
        (x_1,y_1)A---A'(x_1',y_1)

    Args:
        height_field: The input height-field array.
        horizontal_scale: The discretization of the terrain along the x and y axis.
        vertical_scale: The discretization of the terrain along the z axis.
        slope_threshold: The slope threshold above which surfaces are made vertical.
            Defaults to None, in which case no correction is applied.

    Returns:
        The vertices and triangles of the mesh:
        - **vertices** (np.ndarray(float)): Array of shape (num_vertices, 3).
          Each row represents the location of each vertex (in m).
        - **triangles** (np.ndarray(int)): Array of shape (num_triangles, 3).
          Each row represents the indices of the 3 vertices connected by this triangle.
    """
    # read height field
    num_rows, num_cols = height_field.shape
    # create a mesh grid of the height field
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    # copy height field to avoid modifying the original array
    hf = height_field.copy()

    # correct vertical surfaces above the slope threshold
    if slope_threshold is not None:
        # scale slope threshold based on the horizontal and vertical scale
        slope_threshold *= horizontal_scale / vertical_scale
        # allocate arrays to store the movement of the vertices
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        # move vertices along the x-axis
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        # move vertices along the y-axis
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        # move vertices along the corners
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create vertices for the mesh
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    # create triangles for the mesh
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return vertices, triangles


def iter_mesh_tiles(terrain, cell_rows=32):
    """Partition original cell rows, retaining each face once and shared edge vertices.

    Vertices keep their original local coordinates (including slope corrections).
    Only face indices are rebased. Consecutive tiles duplicate one vertex row;
    no triangle is duplicated, simplified, or regenerated.
    """
    if cell_rows < 1:
        raise ValueError("cell_rows must be positive")
    rows, cols = terrain.heightsamples.shape
    for row in range(0, rows - 1, cell_rows):
        end = min(row + cell_rows, rows - 1)
        vertex_start = row * cols
        vertices = terrain.vertices[vertex_start:(end + 1) * cols]
        faces = terrain.triangles[2 * row * (cols - 1):2 * end * (cols - 1)]
        yield vertices, faces - np.uint32(vertex_start)


def make_terrain_cfg(terrain, prim_path="/World/DeepWBCTerrain"):
    """Return ``(AssetBaseCfg, contact_filter_paths)`` after AppLauncher starts.

    Tile the complete static mesh to fit PhysX BV4 cooking. Geometry is exact;
    separate collision shapes can affect contact solving at tile boundaries.
    The callback captures CPU arrays by reference, avoiding config deep copies.
    """
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import get_current_stage
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Vt

    def spawn_mesh(path, spawn_cfg, translation=None, orientation=None):
        stage = get_current_stage()
        if stage.GetPrimAtPath(path).IsValid():
            raise ValueError(f"Terrain prim already exists: {path}")
        root = UsdGeom.Xform.Define(stage, path)
        root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0., 0., 0.))))
        # Isaac Lab passes xyzw; USD Gf.Quatf takes real (w), imaginary (xyz).
        quat = orientation or (0., 0., 0., 1.)
        root.AddOrientOp().Set(Gf.Quatf(quat[3], Gf.Vec3f(*quat[:3])))
        material = UsdShade.Material.Define(stage, path + "/PhysicsMaterial")
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(terrain.cfg.static_friction)
        physics.CreateDynamicFrictionAttr(terrain.cfg.dynamic_friction)
        physics.CreateRestitutionAttr(terrain.cfg.restitution)
        for index, (vertices, triangles) in enumerate(iter_mesh_tiles(terrain)):
            mesh = UsdGeom.Mesh.Define(stage, f"{path}/tile_{index:03d}")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.reshape(-1).astype(np.int32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32)))
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateOrientationAttr("rightHanded")
            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        return root.GetPrim()

    cfg = terrain.cfg
    asset = AssetBaseCfg(
        prim_path=prim_path, collision_group=-1,
        spawn=SpawnerCfg(func=spawn_mesh),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(cfg.transform_x, cfg.transform_y, cfg.transform_z)),
    )
    paths = [f"{prim_path}/tile_{i:03d}" for i, _ in enumerate(range(0, terrain.heightsamples.shape[0] - 1, 32))]
    return asset, paths
