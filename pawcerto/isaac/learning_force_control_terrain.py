"""Released boxes_tm default: 400 rough Perlin triangle-mesh tiles.

Perlin helpers below extracted from fixed b1_gym/utils/terrain.py.
License: see pawcerto/methods/learning_force_control/LICENSE and
pawcerto/methods/learning_force_control/LICENSES/legged_gym/LICENSE.
"""
import numpy as np
import torch

class ForceControlTerrain:
    def __init__(self, cfg, device):
        self.cfg = cfg.terrain
        c = self.cfg
        if c.mesh_type != "boxes_tm" or c.terrain_proportions != [0,0,0,0,0,0,0,0,1.0]:
            raise ValueError("Runtime implements the released default boxes_tm distribution")
        rows, cols = c.num_rows, c.num_cols
        def uniform(bounds):
            return torch.rand(rows,cols,device=device)*(bounds[1]-bounds[0])+bounds[0]
        self.terrain_cell_frictions = uniform(cfg.domain_rand.ground_friction_range)
        self.terrain_cell_restitutions = uniform(cfg.domain_rand.ground_restitution_range)
        self.terrain_cell_roughnesses = uniform(cfg.domain_rand.tile_roughness_range)
        nx, ny = int(c.terrain_length/c.horizontal_scale), int(c.terrain_width/c.horizontal_scale)
        self.height_samples = torch.zeros(rows*nx,cols*ny,device=device)
        self.terrain_cell_center_heights = torch.zeros(rows,cols,device=device)
        self.env_origins = torch.zeros(rows,cols,3,device=device)
        for i in range(rows):
            for j in range(cols):
                # Source draws these even for its otherwise unused type slots.
                np.random.random(); np.random.random()
                x,y=np.meshgrid(np.linspace(0,c.terrain_length*4,nx,endpoint=False),np.linspace(0,c.terrain_width*4,ny,endpoint=False))
                self.height_samples[i*nx:(i+1)*nx,j*ny:(j+1)*ny] = torch.as_tensor(perlin(x,y,seed=i*cols+j),device=device)*self.terrain_cell_roughnesses[i,j]/c.vertical_scale
                self.env_origins[i,j,:2] = torch.tensor(((i+.5)*c.terrain_length,(j+.5)*c.terrain_width),device=device)
        self.height_samples[:10,:]=self.height_samples[-10:,:]=3/c.vertical_scale
        self.height_samples[:,:10]=self.height_samples[:,-10:]=3/c.vertical_scale
        self.height_field_raw=self.height_samples
        self.friction_samples=self.terrain_cell_frictions.repeat_interleave(nx,0).repeat_interleave(ny,1)
        self.restitution_samples=self.terrain_cell_restitutions.repeat_interleave(nx,0).repeat_interleave(ny,1)

    def spawn(self):
        from pxr import UsdGeom, UsdPhysics, PhysxSchema
        import isaaclab.sim as sim_utils
        stage=sim_utils.get_current_stage()
        c=self.cfg
        nx,ny=int(c.terrain_length/c.horizontal_scale),int(c.terrain_width/c.horizontal_scale)
        heights=self.height_samples.cpu().numpy()
        self.collision_paths=[]
        for i in range(c.num_rows):
            for j in range(c.num_cols):
                # Same float16 quantization and slope-to-vertical transformation
                # as Gym convert_heightfield_to_trimesh used by the source.
                h=heights[i*nx:(i+1)*nx+1,j*ny:(j+1)*ny+1].astype(np.float16)
                a,b=h.shape
                xx,yy=np.meshgrid(np.arange(a)*c.horizontal_scale,np.arange(b)*c.horizontal_scale,indexing='ij')
                mx=np.zeros(h.shape);my=np.zeros(h.shape);mc=np.zeros(h.shape)
                threshold=c.slope_treshold*c.horizontal_scale/c.vertical_scale
                mx[:-1,:]+=(h[1:,:]-h[:-1,:]>threshold);mx[1:,:]-=(h[:-1,:]-h[1:,:]>threshold)
                my[:,:-1]+=(h[:,1:]-h[:,:-1]>threshold);my[:,1:]-=(h[:,:-1]-h[:,1:]>threshold)
                mc[:-1,:-1]+=(h[1:,1:]-h[:-1,:-1]>threshold);mc[1:,1:]-=(h[:-1,:-1]-h[1:,1:]>threshold)
                xx+=(mx+mc*(mx==0))*c.horizontal_scale;yy+=(my+mc*(my==0))*c.horizontal_scale
                vertices=np.stack((xx+i*c.terrain_length,yy+j*c.terrain_width,h*c.vertical_scale),-1).reshape(-1,3).astype(np.float32)
                grid=np.arange(a*b).reshape(a,b);v=grid[:-1,:-1].ravel()
                faces=np.stack((np.stack((v,v+b+1,v+1),-1),np.stack((v,v+b,v+b+1),-1)),1).reshape(-1,3)
                path=f"/World/Terrain/tile_{i}_{j}"
                mesh=UsdGeom.Mesh.Define(stage,path);mesh.CreatePointsAttr(vertices.tolist());mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
                collision=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim());collision.CreateContactOffsetAttr(.01);collision.CreateRestOffsetAttr(0.)
                material=f"{path}/material"
                sim_utils.spawn_rigid_body_material(material,sim_utils.RigidBodyMaterialCfg(static_friction=float(self.terrain_cell_frictions[i,j]),dynamic_friction=float(self.terrain_cell_frictions[i,j]),restitution=float(self.terrain_cell_restitutions[i,j])))
                sim_utils.bind_physics_material(path,material)
                self.collision_paths.append(path)

def perlin(x, y, seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y
