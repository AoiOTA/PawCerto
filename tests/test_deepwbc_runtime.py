"""CPU asset preparation, USD sharing and frame math; no physics-run evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from pawcerto.isaac.deepwbc_runtime import (SOURCE_URDF, SIM_DOF_NAMES, FEET,
    source_tree, prepare_urdf, link_paths, transform_wrench_to_link, lab_initial_joint_positions, material_properties_for_friction)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('convert_deepwbc',ROOT/'scripts/convert_deepwbc_usd.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class DeepWBCRuntimePreparationTests(unittest.TestCase):
    def test_prepared_source_preserves_physics_and_mimic(self):
        if not SOURCE_URDF.exists():
            self.skipTest('Fetch pinned DeepWBC assets first')
        original = ET.parse(SOURCE_URDF)
        with tempfile.TemporaryDirectory() as directory:
            tree, manifest = prepare_urdf(directory)
            self.assertEqual(manifest['source_mass'],14.150879)
            self.assertEqual(manifest['source_collisions'],25)
            self.assertEqual(len(tree.findall('.//collision')),len(original.findall('.//collision')))
            self.assertEqual(manifest['joint_names'],list(SIM_DOF_NAMES))
            self.assertIsNone(manifest['limits']['widow_waist']['lower'])
            self.assertIsNone(manifest['limits']['widow_waist']['upper'])
            for new, old in zip(tree.findall('link'),original.findall('link')):
                self.assertEqual(new.get('name'),old.get('name').replace('/','_'))
                if old.find('inertial') is not None:
                    self.assertEqual(ET.tostring(new.find('inertial')),ET.tostring(old.find('inertial')))
            for new, old in zip(tree.findall('joint'),original.findall('joint')):
                self.assertEqual(new.get('name'),old.get('name'))
                self.assertEqual(new.get('type'),old.get('type'))
                for key in ('origin','axis','limit','dynamics','mimic'):
                    if old.find(key) is not None:
                        self.assertEqual(ET.tostring(new.find(key)),ET.tostring(old.find(key)))
            self.assertEqual(tree.find("joint[@name='widow_right_finger']/mimic").attrib,
                             {'joint':'widow_left_finger','multiplier':'-1','offset':'0'})
            self.assertEqual(set(manifest['protected_fixed_joints']),
                {'ee_gripper','floating_base',*(f'{leg}_foot_fixed' for leg in ('FL','FR','RL','RR'))})
            self.assertTrue(all(Path(m.get('filename')).is_file() for m in tree.findall('.//mesh')))
            self.assertEqual(json.loads((Path(directory)/'source-manifest.json').read_text()),manifest)

    def test_lab_construction_pose_keeps_original_task_defaults(self):
        from pawcerto.methods.deepwbc.config import default_config
        cfg=default_config()
        with tempfile.TemporaryDirectory() as directory:
            _,manifest=prepare_urdf(directory)
        pose=lab_initial_joint_positions(cfg,manifest['limits'])
        changed={n for n in pose if pose[n]!=cfg['init_state']['default_joint_angles'][n]}
        self.assertEqual(changed,{'widow_left_finger','widow_right_finger'})
        self.assertEqual(pose['widow_left_finger'],.015)
        self.assertEqual(pose['widow_right_finger'],-.015)
        self.assertEqual(cfg['init_state']['default_joint_angles']['widow_left_finger'],0)
        self.assertEqual(cfg['init_state']['default_joint_angles']['widow_right_finger'],0)
        for name, bounds in manifest['limits'].items():
            if bounds['lower'] is not None:
                self.assertGreaterEqual(pose[name],bounds['lower'])
                self.assertLessEqual(pose[name],bounds['upper'])

    def test_material_payload_layout_and_negative_friction_failure(self):
        current=torch.rand(4,25,3)
        original=current.clone()
        friction=torch.tensor([[0.],[.2],[1.],[3.]])
        payload=material_properties_for_friction(current,friction)
        self.assertTrue(payload.is_contiguous())
        self.assertEqual(payload.dtype,torch.float32)
        torch.testing.assert_close(payload[:,:,0],friction.expand(4,25))
        torch.testing.assert_close(payload[:,:,1],friction.expand(4,25))
        torch.testing.assert_close(payload[:,:,2],original[:,:,2])
        torch.testing.assert_close(current,original)
        friction[1]=-.1
        with self.assertRaisesRegex(ValueError,'nonnegative friction'):
            material_properties_for_friction(current,friction)
        with self.assertRaisesRegex(ValueError,r'friction\[N,1\]'):
            material_properties_for_friction(current,friction[:,0])

    def test_original_mount_and_named_tree(self):
        tree = source_tree()
        mount = tree.find("joint[@name='widow_mount']")
        self.assertEqual(mount.find('origin').get('xyz'),'0.03 0 0.057')
        self.assertEqual(mount.find('origin').get('rpy'),'0 0 0')
        paths = link_paths(tree)
        self.assertEqual(len(paths),len(tree.findall('link')))
        for foot in FEET:
            self.assertTrue(paths[foot].endswith('/'+foot))
        self.assertIn('wx250s_ee_gripper_link',paths)
        # Bad hierarchy must fail explicitly instead of looping indefinitely.
        tree.find("joint[@name='widow_mount']/parent").set('link','missing')
        with self.assertRaises(ValueError):
            link_paths(tree)

    def test_full_wrench_transform_rotation_and_anchor(self):
        rng = np.random.default_rng(9)
        rot = Rotation.random(4,random_state=rng)
        f = rng.standard_normal((3,4,3));t=rng.standard_normal((3,4,3));r=rng.standard_normal((4,3))
        rotated_force=np.einsum('bij,nbj->nbi',rot.as_matrix(),f)
        rotated_torque=np.einsum('bij,nbj->nbi',rot.as_matrix(),t)+np.cross(r,rotated_force)
        actual=transform_wrench_to_link(torch.tensor(f),torch.tensor(t),torch.tensor(r),torch.tensor(rot.as_quat()))
        np.testing.assert_allclose(actual.numpy(),np.concatenate((rotated_force,rotated_torque),axis=-1),rtol=1e-12,atol=1e-12)
        self.assertEqual(actual.shape,(3,4,6))
        self.assertGreater(actual[...,3:].abs().sum().item(),0)

    def test_module_import_does_not_start_isaac(self):
        self.assertNotIn('isaaclab.app',sys.modules)
        self.assertNotIn('omni.kit.app',sys.modules)

    def test_source_inertia_restoration_uses_full_tensor(self):
        try:
            from pxr import Usd,UsdGeom,UsdPhysics,Gf
        except ImportError:
            self.skipTest('Standalone USD bindings required')
        source=ET.ElementTree(ET.fromstring('''<robot name="inertia"><link name="body"><inertial>
        <origin xyz=".1 .2 .3" rpy=".2 -.3 .4"/><mass value="2"/>
        <inertia ixx=".3" ixy=".01" ixz=".02" iyy=".4" iyz=".03" izz=".5"/>
        </inertial></link></robot>'''))
        stage=Usd.Stage.CreateInMemory()
        prim=UsdGeom.Xform.Define(stage,'/body').GetPrim()
        mass=UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(2.)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(.1,.2,.3))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.,1.,1.))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.))
        report=converter._restore_source_inertias(stage,source)
        self.assertGreater(report[0]['before_max_abs_error'],.1)
        self.assertLess(report[0]['after_max_abs_error'],1e-7)

    def test_urdf_missing_com_origin_is_authored_not_usd_auto(self):
        try:
            from pxr import Usd,UsdGeom,UsdPhysics,Gf
        except ImportError:
            self.skipTest('Standalone USD bindings required')
        source=ET.ElementTree(ET.fromstring('''<robot name="marker"><link name="foot"><inertial>
        <mass value=".06"/><inertia ixx=".0000096" ixy="0" ixz="0" iyy=".0000096" iyz="0" izz=".0000096"/>
        </inertial></link></robot>'''))
        stage=Usd.Stage.CreateInMemory()
        prim=UsdGeom.Xform.Define(stage,'/foot').GetPrim()
        mass=UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(.06)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.0000096))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.))
        self.assertFalse(mass.GetCenterOfMassAttr().HasAuthoredValueOpinion())
        self.assertTrue(np.isneginf(np.array(mass.GetCenterOfMassAttr().Get())).all())
        report=converter._restore_source_inertias(stage,source)
        self.assertTrue(report[0]['com_was_implicit'])
        self.assertEqual(tuple(mass.GetCenterOfMassAttr().Get()),(0.,0.,0.))
        self.assertTrue(mass.GetCenterOfMassAttr().HasAuthoredValueOpinion())

    def test_measured_source_unbounded_waist_restores_usd_limits(self):
        try:
            from pxr import Usd,UsdPhysics
        except ImportError:
            self.skipTest('Standalone USD bindings required')
        stage=Usd.Stage.CreateInMemory()
        waist=UsdPhysics.RevoluteJoint.Define(stage,'/widow_waist')
        waist.CreateLowerLimitAttr(0.);waist.CreateUpperLimitAttr(0.)
        shoulder=UsdPhysics.RevoluteJoint.Define(stage,'/widow_shoulder')
        shoulder.CreateLowerLimitAttr(-108.);shoulder.CreateUpperLimitAttr(114.)
        report=converter._restore_source_unbounded_joints(stage,source_tree())
        self.assertEqual(report,[{'joint':'widow_waist','before_limits':[0.,0.],'source_has_limits':False}])
        self.assertEqual(waist.GetLowerLimitAttr().Get(),-float('inf'))
        self.assertEqual(waist.GetUpperLimitAttr().Get(),float('inf'))
        self.assertEqual(shoulder.GetLowerLimitAttr().Get(),-108.)
        self.assertEqual(shoulder.GetUpperLimitAttr().Get(),114.)

    def test_shared_usd_collision_layer_preserves_two_instances(self):
        try:
            from pxr import Usd,UsdGeom,UsdPhysics
        except ImportError:
            self.skipTest('Standalone USD bindings required; available in existing Lab Python')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            geometry=Usd.Stage.CreateNew(str(root/'geometry.usda'))
            definition=UsdGeom.Xform.Define(geometry,'/Shared')
            geometry.SetDefaultPrim(definition.GetPrim())
            shape=UsdGeom.Cylinder.Define(geometry,'/Shared/collision')
            shape.CreateRadiusAttr(.05);shape.CreateHeightAttr(.2)
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            geometry.GetRootLayer().Save()
            stage=Usd.Stage.CreateNew(str(root/'robot.usda'))
            for name in ('Robot0','Robot1'):
                prim=UsdGeom.Xform.Define(stage,'/'+name).GetPrim()
                prim.GetReferences().AddReference('geometry.usda','/Shared')
                prim.SetInstanceable(True)
            stage.GetRootLayer().Save()
            report=converter.author_shared_collision_properties(root/'robot.usda',root)
            self.assertEqual(report['instance_proxy_collisions'],2)
            stage.Reload()
            for name in ('Robot0','Robot1'):
                prim=stage.GetPrimAtPath('/'+name)
                self.assertTrue(prim.IsInstance())
                proxy=stage.GetPrimAtPath('/'+name+'/collision')
                self.assertTrue(proxy.IsInstanceProxy())
                self.assertTrue(proxy.IsA(UsdGeom.Capsule))
                self.assertAlmostEqual(proxy.GetAttribute('physxCollision:contactOffset').Get(),.01,places=7)
            self.assertEqual(stage.GetPrimAtPath('/Robot0').GetPrototype(),stage.GetPrimAtPath('/Robot1').GetPrototype())


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
