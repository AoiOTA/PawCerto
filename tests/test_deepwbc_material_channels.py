import tempfile
import unittest

import torch
from pawcerto.isaac.deepwbc_material_channels import discover_channel_masks, channel_material_properties


class MaterialChannelsTests(unittest.TestCase):
    def test_native_permuted_roles_preserve_original_samples(self):
        m = torch.tensor([[[0.,0.,.1],[.5,.5,.2],[.5,.5,.3],[0.,0.,.4]],
                          [[.5,.5,.1],[0.,0.,.2],[0.,0.,.3],[.5,.5,.4]]])
        masks = discover_channel_masks(m,2)
        f = torch.tensor([[-.5],[3.]])
        result = channel_material_properties(m,f,masks)
        torch.testing.assert_close(result[...,0],torch.tensor([[0.,.25,.25,0.],[2.,3.,3.,2.]]))
        torch.testing.assert_close(result[...,2],m[...,2])
        torch.testing.assert_close(f,torch.tensor([[-.5],[3.]]))
        with self.assertRaises(ValueError): discover_channel_masks(torch.ones_like(m),2)

    def test_cloned_mixed_geometry_keeps_mass_and_instances(self):
        try:
            from pxr import Usd, UsdGeom, UsdPhysics
        except ImportError:
            self.skipTest('Standalone USD not installed')
        from pawcerto.isaac.deepwbc_material_channels import author_material_channels
        with tempfile.TemporaryDirectory() as tmp:
            source=Usd.Stage.CreateNew(tmp+'/collision.usda')
            root=UsdGeom.Xform.Define(source,'/Geometry');source.SetDefaultPrim(root.GetPrim())
            UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(source,'/Geometry/Shape').GetPrim())
            source.GetRootLayer().Save()
            s=Usd.Stage.CreateInMemory()
            UsdPhysics.Scene.Define(s,"/physicsScene")
            env0=UsdGeom.Xform.Define(s,'/World/envs/env_0').GetPrim()
            robot=UsdGeom.Xform.Define(s,str(env0.GetPath())+'/Robot').GetPrim()
            body=UsdGeom.Xform.Define(s,str(robot.GetPath())+'/Body').GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body);UsdPhysics.MassAPI.Apply(body).CreateMassAttr(2.)
            inst=UsdGeom.Xform.Define(s,str(body.GetPath())+'/Collision').GetPrim()
            inst.GetReferences().AddReference(tmp+'/collision.usda');inst.SetInstanceable(True)
            UsdPhysics.CollisionAPI.Apply(UsdGeom.Sphere.Define(s,str(body.GetPath())+'/Sphere').GetPrim())
            box=UsdGeom.Xform.Define(s,str(env0.GetPath())+'/Box').GetPrim();UsdPhysics.RigidBodyAPI.Apply(box)
            env1=UsdGeom.Xform.Define(s,'/World/envs/env_1').GetPrim();env1.GetInherits().AddInherit(env0.GetPath())
            ground=UsdGeom.Cube.Define(s,'/World/Ground').GetPrim();UsdPhysics.CollisionAPI.Apply(ground)
            UsdGeom.Scope.Define(s,'/World/collisions')
            UsdPhysics.CollisionGroup.Define(s,'/World/collisions/old')
            result=author_material_channels(s,[f'/World/envs/env_{i}/Robot' for i in range(2)],
                [f'/World/envs/env_{i}/Box' for i in range(2)],['/World/Ground'])
            self.assertEqual(result['collision_count'],2)
            self.assertEqual(result['group_count'],7)
            self.assertFalse(s.GetPrimAtPath('/World/collisions').IsActive())
            self.assertEqual(result['shared_prototypes'],1)
            for i in range(2):
                b=s.GetPrimAtPath(f'/World/envs/env_{i}/Robot/Body')
                self.assertEqual(UsdPhysics.MassAPI(b).GetMassAttr().Get(),2.)
                self.assertEqual(sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(b,Usd.TraverseInstanceProxies())),4)
                self.assertEqual(len(UsdPhysics.FilteredPairsAPI(b).GetFilteredPairsRel().GetTargets()),2)


if __name__=='__main__': unittest.main()
