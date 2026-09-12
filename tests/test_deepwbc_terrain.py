"""Independent source parity, using AST to avoid importing Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings
import numpy as np
from pawcerto.methods.deepwbc.config import default_config
from pawcerto.methods.deepwbc.terrain import (
    build_terrain, convert_height_field_to_mesh, environment_origin_bounds, iter_mesh_tiles,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'third_party/deepwbc-reference/legged_gym/legged_gym/utils/terrain.py'
GYM_HELPER = ROOT / 'third_party/umi-on-legs/mani-centric-wbc/legged_gym/env/isaacgym/terrain_utils.py'


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, 'name', None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@unittest.skipUnless(SOURCE.exists() and GYM_HELPER.exists(), 'Pinned original sources required for parity')
class TerrainParity(unittest.TestCase):
    def test_original_perlin_and_cast(self):
        cfg = default_config()['terrain']
        cfg.update(tot_cols=240, tot_rows=80)
        cfg = SimpleNamespace(**cfg)
        helper = extract(GYM_HELPER, 'convert_heightfield_to_trimesh', {'np': np})
        original = extract(SOURCE, 'Terrain_Perlin', {
            'np': np, 'terrain_utils': SimpleNamespace(convert_heightfield_to_trimesh=helper),
        })
        # Compare actual independent pinned source, including the offset/int16 cast.
        for seed in (1, 19):
            np.random.seed(seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                expected = original(cfg)
                actual = build_terrain(seed, cfg)
            for name in ('heightsamples_float', 'heightsamples', 'vertices', 'triangles'):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    def test_original_mesh_slopes_and_winding(self):
        helper = extract(GYM_HELPER, 'convert_heightfield_to_trimesh', {'np': np})
        hf = np.array([[0, 4000, 8000], [9000, -2000, 4000], [1000, 3000, -5000]], dtype=np.int16)
        for slope in (None, 0.5, 1e8):
            expected = helper(hf, 0.025, 1e-5, slope)
            actual = convert_height_field_to_mesh(hf, 0.025, 1e-5, slope)
            for a, e in zip(actual, expected):
                np.testing.assert_array_equal(a, e)
        np.testing.assert_array_equal(actual[1][:2], [[0, 4, 1], [0, 3, 4]])

    def test_origins_and_rng_isolation(self):
        self.assertEqual(environment_origin_bounds(), ((-3.75, -3.0), (-115.0, 115.0)))
        cfg = default_config()['terrain']
        cfg.update(tot_cols=240, tot_rows=80)
        np.random.seed(7)
        expected = np.random.rand(4)
        np.random.seed(7)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            build_terrain(1, cfg)
        np.testing.assert_array_equal(np.random.rand(4), expected)


class TerrainTiles(unittest.TestCase):
    def test_partition_preserves_every_original_face_and_boundary(self):
        # Nonflat surface and nonzero slope correction: tiling must use existing vertices.
        hf = (np.arange(67 * 4).reshape(67, 4) % 7 * 1000).astype(np.int16)
        vertices, triangles = convert_height_field_to_mesh(hf, .025, 1e-5, .5)
        terrain = SimpleNamespace(heightsamples=hf, vertices=vertices, triangles=triangles)
        restored = []
        tiles = list(iter_mesh_tiles(terrain))
        self.assertEqual(len(tiles), 3)
        for index, (points, faces) in enumerate(tiles):
            base = index * 32 * 4
            np.testing.assert_array_equal(points, vertices[base:base + len(points)])
            restored.append(faces + base)
            self.assertGreaterEqual(faces.min(), 0)
            self.assertLess(faces.max(), len(points))
            if index:
                np.testing.assert_array_equal(tiles[index - 1][0][-4:], points[:4])
        # Equality of ordered index triples proves face coverage and winding together.
        np.testing.assert_array_equal(np.concatenate(restored), triangles)


class TerrainUsd(unittest.TestCase):
    def test_actual_usd_mesh_callback_without_sim(self):
        try:
            from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Vt
        except ImportError:
            self.skipTest('USD Python libraries unavailable')
        stage = Usd.Stage.CreateInMemory()
        hf = np.arange(67 * 4).reshape(67, 4).astype(np.int16)
        vertices, triangles = convert_height_field_to_mesh(hf, .025, 1e-5)
        cfg = SimpleNamespace(**default_config()['terrain'])
        terrain = SimpleNamespace(heightsamples=hf, vertices=vertices, triangles=triangles, cfg=cfg)
        # Execute the actual factory and USD callback. Substitute only Lab config
        # containers and stage lookup, so no AppLauncher/PhysX is started.
        tree = ast.parse((ROOT / 'pawcerto/methods/deepwbc/terrain.py').read_text())
        factory = next(n for n in tree.body if getattr(n, 'name', None) == 'make_terrain_cfg')
        factory.body = [n for n in factory.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        class InitialStateCfg(SimpleNamespace):
            # Current supported Lab contract is xyzw, including its default.
            rot = (0., 0., 0., 1.)
        class AssetBaseCfg(SimpleNamespace):
            pass
        AssetBaseCfg.InitialStateCfg = InitialStateCfg
        namespace = dict(np=np, get_current_stage=lambda: stage, iter_mesh_tiles=iter_mesh_tiles,
                         AssetBaseCfg=AssetBaseCfg, SpawnerCfg=SimpleNamespace,
                         UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, UsdShade=UsdShade, Gf=Gf, Vt=Vt)
        exec(compile(ast.Module(body=[factory], type_ignores=[]), 'terrain.py', 'exec'), namespace)
        asset, paths = namespace['make_terrain_cfg'](terrain)
        root = asset.spawn.func(asset.prim_path, asset.spawn, asset.init_state.pos, asset.init_state.rot)
        collider_paths = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        self.assertEqual(paths, collider_paths)
        self.assertEqual(len(paths), 3)
        self.assertEqual(asset.collision_group, -1)
        for path, (points, faces) in zip(paths, iter_mesh_tiles(terrain)):
            prim = stage.GetPrimAtPath(path)
            mesh = UsdGeom.Mesh(prim)
            np.testing.assert_array_equal(np.asarray(mesh.GetPointsAttr().Get()), points)
            np.testing.assert_array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), faces)
            self.assertEqual(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), 'none')
            self.assertTrue(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            np.testing.assert_array_equal(UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation(), [-7.5, -125., 0.])
            world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            np.testing.assert_allclose(world.Transform(Gf.Vec3d(*map(float, points[-1]))), points[-1] + np.array(asset.init_state.pos), atol=1e-7)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            physics = UsdPhysics.MaterialAPI(material.GetPrim())
            self.assertEqual(physics.GetStaticFrictionAttr().Get(), 1.)
            self.assertEqual(physics.GetDynamicFrictionAttr().Get(), 1.)
            self.assertEqual(physics.GetRestitutionAttr().Get(), 0.)

        yaw90_xyzw = (0., 0., np.sqrt(.5), np.sqrt(.5))
        rotated = asset.spawn.func('/World/RotatedTerrain', asset.spawn, (0., 0., 0.), yaw90_xyzw)
        rotation = UsdGeom.XformCache().GetLocalToWorldTransform(rotated)
        np.testing.assert_allclose(rotation.TransformDir(Gf.Vec3d(1., 0., 0.)), [0., 1., 0.], atol=1e-6)
        # Catch future drift of the real installed source default without starting Sim.
        lab_cfg = ROOT / 'third_party/IsaacLab-develop-sim610/source/isaaclab/isaaclab/assets/asset_base_cfg.py'
        if lab_cfg.exists():
            tree = ast.parse(lab_cfg.read_text())
            real_rot = next(n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == 'rot')
            self.assertEqual(ast.literal_eval(real_rot.value), asset.init_state.rot)


if __name__ == '__main__':
    unittest.main()
