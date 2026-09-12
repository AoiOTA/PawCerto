"""Convert the nominal AS2/Piper-H via official unmodified Isaac Lab/PhysX."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pawcerto.robots.as2_piper import prepare_urdf


def author_shared_collision_offsets(stage, output_root, contact_offset=.01, rest_offset=0.):
    """Put nominal solver offsets on generated shared layers, not every clone."""
    import math
    from pxr import Sdf, Usd, UsdPhysics
    output_root = Path(output_root).resolve()
    changed_layers = []
    for layer in stage.GetUsedLayers():
        if not layer.realPath or not Path(layer.realPath).resolve().is_relative_to(output_root):
            continue
        layer_stage = Usd.Stage.Open(layer)
        changed = False
        for prim in layer_stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.AddAppliedSchema('PhysxCollisionAPI')
                prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float, custom=False).Set(contact_offset)
                prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float, custom=False).Set(rest_offset)
                changed = True
        if changed:
            layer.Save()
            changed_layers.append(layer.realPath)
    collisions = [p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
                  if p.HasAPI(UsdPhysics.CollisionAPI)]
    for prim in collisions:
        contact = prim.GetAttribute('physxCollision:contactOffset').Get()
        rest = prim.GetAttribute('physxCollision:restOffset').Get()
        if contact is None or not math.isclose(contact, contact_offset, rel_tol=1e-6) or rest != rest_offset:
            raise ValueError(f'Shared collision offsets did not compose: {prim.GetPath()}')
    return dict(contact_offset=contact_offset, rest_offset=rest_offset, changed_layers=changed_layers,
                collision_count=len(collisions), instance_proxy_collision_count=sum(p.IsInstanceProxy() for p in collisions))


def author_source_physx_limits(stage, tree):
    """Preserve source limits when importer only emits Newton velocity fields."""
    import math
    from pxr import Sdf, UsdPhysics
    physics = stage.GetDefaultPrim().GetVariantSet('Physics')
    if 'physx' not in physics.GetVariantNames():
        raise ValueError('Official converted asset has no PhysX variant')
    physics.SetVariantSelection('physx')
    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    for joint in tree.findall('joint'):
        if joint.get('type') == 'revolute':
            prim = joints[joint.get('name')]
            prim.AddAppliedSchema('PhysxJointAPI')
            # PhysX's installed USD schema documents angular velocity in deg/s.
            prim.CreateAttribute('physxJoint:maxJointVelocity', Sdf.ValueTypeNames.Float, custom=False).Set(
                math.degrees(float(joint.find('limit').get('velocity'))))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.AddAppliedSchema('PhysxArticulationAPI')
            prim.CreateAttribute('physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool, custom=False).Set(True)


def restore_and_validate(stage, tree, manifest):
    """Audit actual USD bodies/joints and restore source inertia representation."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from pawcerto.mujoco.as2_piper_asset import assemble, urdf_fk

    def rotation(quat):
        return Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()

    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if set(bodies) != set(manifest['body_names']):
        raise ValueError(f'USD rigid bodies differ from source: {set(bodies) ^ set(manifest["body_names"])}')
    records = []
    for link in tree.findall('link'):
        name = link.get('name')
        inertial = link.find('inertial')
        origin = inertial.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        i = inertial.find('inertia')
        xx, xy, xz, yy, yz, zz = [float(i.get(k)) for k in ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')]
        rot = Rotation.from_euler('xyz', rpy).as_matrix()
        tensor = rot @ np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) @ rot.T
        moments, axes = np.linalg.eigh(tensor)
        if not np.isfinite(tensor).all() or not np.all(moments > 0):
            raise ValueError(f'Invalid source inertia: {name}')
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        api = UsdPhysics.MassAPI(bodies[name])
        np.testing.assert_allclose(api.GetMassAttr().Get(), float(inertial.find('mass').get('value')), rtol=1e-6)
        # Omitted URDF origin is zero, while unauthored USD COM is an -inf auto sentinel.
        before_com = list(api.GetCenterOfMassAttr().Get())
        before = rotation(api.GetPrincipalAxesAttr().Get()) @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ rotation(api.GetPrincipalAxesAttr().Get()).T
        api.GetCenterOfMassAttr().Set(Gf.Vec3f(*map(float, xyz)))
        api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, moments)))
        quat = Rotation.from_matrix(axes).as_quat()
        api.GetPrincipalAxesAttr().Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))
        after_rot = rotation(api.GetPrincipalAxesAttr().Get())
        after = after_rot @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ after_rot.T
        np.testing.assert_allclose(after, tensor, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(api.GetCenterOfMassAttr().Get(), xyz, atol=1e-7)
        records.append({'body': name, 'com_was_implicit': not np.isfinite(before_com).all().item(),
                        'before_com_error': float(np.max(abs(np.array(before_com)-xyz))) if np.isfinite(before_com).all() else None,
                        'before_tensor_error': float(np.max(abs(before-tensor))),
                        'after_tensor_error': float(np.max(abs(after-tensor)))})
    source = assemble()
    poses = urdf_fk(source, dict.fromkeys(manifest['controlled_joint_order'], 0.), [0, 0, 0], [1, 0, 0, 0])
    cache = UsdGeom.XformCache()
    for name, prim in bodies.items():
        actual = np.array(cache.GetLocalToWorldTransform(prim)).T
        np.testing.assert_allclose(actual, poses[name], atol=2e-6, err_msg=f'{name}: zero-pose FK')
    usd_joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    if set(usd_joints) != set(manifest['controlled_joint_order']):
        raise ValueError('USD controlled joint set differs')
    fixed = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)}
    if set(fixed) != set(manifest['fixed_joint_names']):
        raise ValueError(f'USD fixed joints differ: {set(fixed) ^ set(manifest["fixed_joint_names"])}')
    for joint in tree.findall('joint'):
        if joint.get('type') != 'fixed':
            continue
        api = UsdPhysics.FixedJoint(fixed[joint.get('name')])
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        if api.GetBody0Rel().GetTargets() != [bodies[parent].GetPath()] or api.GetBody1Rel().GetTargets() != [bodies[child].GetPath()]:
            raise ValueError(f'USD fixed joint endpoints differ: {joint.get("name")}')
        anchor0 = poses[parent][:3,:3] @ np.array(api.GetLocalPos0Attr().Get()) + poses[parent][:3,3]
        anchor1 = poses[child][:3,:3] @ np.array(api.GetLocalPos1Attr().Get()) + poses[child][:3,3]
        np.testing.assert_allclose(anchor0, anchor1, atol=2e-6, err_msg=f'{joint.get("name")}: fixed anchor')
        np.testing.assert_allclose(poses[parent][:3,:3] @ rotation(api.GetLocalRot0Attr().Get()),
                                   poses[child][:3,:3] @ rotation(api.GetLocalRot1Attr().Get()),
                                   atol=2e-6, err_msg=f'{joint.get("name")}: fixed orientation')
    joint_records = {}
    for joint in tree.findall('joint'):
        if joint.get('type') != 'revolute':
            continue
        name = joint.get('name')
        api = UsdPhysics.RevoluteJoint(usd_joints[name])
        limit = joint.find('limit')
        np.testing.assert_allclose(usd_joints[name].GetAttribute('physxJoint:maxJointVelocity').Get(),
                                   np.rad2deg(float(limit.get('velocity'))), rtol=1e-6)
        drive = UsdPhysics.DriveAPI(usd_joints[name], 'angular')
        np.testing.assert_allclose(drive.GetMaxForceAttr().Get(), float(limit.get('effort')), rtol=1e-6)
        if drive.GetStiffnessAttr().Get() != 0. or drive.GetDampingAttr().Get() != 0.:
            raise ValueError(f'Nonzero drive gains in direct torque asset: {name}')
        np.testing.assert_allclose(np.deg2rad([api.GetLowerLimitAttr().Get(), api.GetUpperLimitAttr().Get()]),
                                   [float(limit.get('lower')), float(limit.get('upper'))], atol=1e-6)
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        if api.GetBody0Rel().GetTargets() != [bodies[parent].GetPath()] or api.GetBody1Rel().GetTargets() != [bodies[child].GetPath()]:
            raise ValueError(f'USD joint endpoints differ: {name}')
        axis = np.eye(3)['XYZ'.index(str(api.GetAxisAttr().Get()))]
        actual_axis = poses[parent][:3,:3] @ rotation(api.GetLocalRot0Attr().Get()) @ axis
        origin = joint.find('origin')
        rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        expected_anchor = poses[parent][:3,:3] @ xyz + poses[parent][:3,3]
        anchor0 = poses[parent][:3,:3] @ np.array(api.GetLocalPos0Attr().Get()) + poses[parent][:3,3]
        anchor1 = poses[child][:3,:3] @ np.array(api.GetLocalPos1Attr().Get()) + poses[child][:3,3]
        np.testing.assert_allclose(anchor0, expected_anchor, atol=2e-6, err_msg=f'{name}: parent anchor')
        np.testing.assert_allclose(anchor1, expected_anchor, atol=2e-6, err_msg=f'{name}: child anchor')
        expected_axis = poses[parent][:3,:3] @ Rotation.from_euler('xyz', rpy).apply(np.fromstring(joint.find('axis').get('xyz'), sep=' '))
        np.testing.assert_allclose(actual_axis, expected_axis, atol=2e-6, err_msg=f'{name}: joint axis')
        child_axis = poses[child][:3,:3] @ rotation(api.GetLocalRot1Attr().Get()) @ axis
        np.testing.assert_allclose(child_axis, expected_axis, atol=2e-6, err_msg=f'{name}: child joint axis')
        joint_records[name] = {'path': str(api.GetPath()), **{k: float(limit.get(k)) for k in ('lower', 'upper', 'effort', 'velocity')}}
    collisions = [p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
    source_count = len(tree.findall('.//collision'))
    if len(collisions) != source_count or not any(p.IsInstanceProxy() for p in collisions):
        raise ValueError(f'Expected {source_count} shared source collisions, got {len(collisions)}')
    from collections import Counter
    expected_shapes = Counter(c.find('geometry')[0].tag.lower() for c in tree.findall('.//collision'))
    actual_shapes = Counter(p.GetTypeName().lower() for p in collisions)
    expected_shapes['cube'] = expected_shapes.pop('box', 0)
    if +expected_shapes != +actual_shapes:
        raise ValueError(f'Source/USD collision shape types differ: {expected_shapes} / {actual_shapes}')
    # Compare primitive centers and shape matrices, which are invariant to
    # equivalent cylinder-axis and cube-orientation representations in USD.
    expected_primitives = []
    for link in tree.findall('link'):
        for collision in link.findall('collision'):
            geom = collision.find('geometry')[0]
            if geom.tag == 'mesh':
                continue
            origin = collision.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            pose = poses[link.get('name')]
            rot = pose[:3,:3] @ Rotation.from_euler('xyz', rpy).as_matrix()
            if geom.tag == 'box':
                radii = np.fromstring(geom.get('size'), sep=' ') / 2
            elif geom.tag == 'cylinder':
                radii = np.array([float(geom.get('radius'))]*2 + [float(geom.get('length'))/2])
            else:
                radii = np.full(3, float(geom.get('radius')))
            expected_primitives.append((pose[:3,:3] @ xyz + pose[:3,3], rot @ np.diag(radii**2) @ rot.T))
    for prim in collisions:
        if prim.IsA(UsdGeom.Mesh):
            continue
        pose = np.array(cache.GetLocalToWorldTransform(prim)).T
        if prim.IsA(UsdGeom.Cube):
            radii = np.full(3, UsdGeom.Cube(prim).GetSizeAttr().Get()/2)
        elif prim.IsA(UsdGeom.Cylinder):
            shape = UsdGeom.Cylinder(prim)
            radii = np.full(3, shape.GetRadiusAttr().Get())
            radii['XYZ'.index(str(shape.GetAxisAttr().Get()))] = shape.GetHeightAttr().Get()/2
        else:
            radii = np.full(3, UsdGeom.Sphere(prim).GetRadiusAttr().Get())
        matrix = pose[:3,:3] @ np.diag(radii**2) @ pose[:3,:3].T
        match = next((i for i,(center, shape) in enumerate(expected_primitives)
                      if np.allclose(center, pose[:3,3], atol=2e-6, rtol=0)
                      and np.allclose(shape, matrix, atol=1e-7, rtol=1e-5)), None)
        if match is None:
            raise ValueError(f'Collision primitive transform/dimensions differ: {prim.GetPath()}')
        expected_primitives.pop(match)
    if expected_primitives:
        raise ValueError('Source primitive geometry missing from USD')
    # Retain the virtual source stroke frame and nominal TCP as kinematic Xforms.
    import xml.etree.ElementTree as ET
    gripper = bodies['piper_gripper_base'].GetPath()
    virtual = next(r for r in manifest['removed_massless_frames'] if r['name'] == 'piper_gripper_link')
    virtual_origin = ET.fromstring(virtual['joint']).find('origin')
    virtual_xyz = np.fromstring(virtual_origin.get('xyz', '0 0 0'), sep=' ')
    virtual_rpy = np.fromstring(virtual_origin.get('rpy', '0 0 0'), sep=' ')
    for name, xyz, rpy in [('piper_gripper_link', virtual_xyz, virtual_rpy),
                           ('tcp', manifest['tcp']['xyz'], manifest['tcp']['rpy'])]:
        marker = UsdGeom.Xform.Define(stage, gripper.AppendChild(name))
        marker.ClearXformOpOrder()
        marker.AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))
        q = Rotation.from_euler('xyz', rpy).as_quat()
        marker.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if stage.GetDefaultPrim().GetVariantSet('Physics').GetVariantSelection() != 'physx':
        raise ValueError('Expected PhysX variant')
    if len(roots) != 1:
        raise ValueError(f'Expected one articulation root: {roots}')
    return {**manifest, 'articulation_root_path': roots[0], 'default_prim': str(stage.GetDefaultPrim().GetPath()),
            'body_paths': {n: str(p.GetPath()) for n,p in bodies.items()}, 'joints': joint_records,
            'tcp_path': str(gripper.AppendChild('tcp')), 'inertias': records,
            'collision_shapes': dict(actual_shapes), 'collision_count': len(collisions), 'instance_proxy_collision_count': sum(p.IsInstanceProxy() for p in collisions),
            'evidence': 'USD source inertias and zero-pose FK; revolute/fixed constraints, source limits and zero gains; primitive collision dimensions/transforms and mesh instancing. No simulation/training acceptance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reference/isaac/as2_piper')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--prepare-collision-offsets-only', action='store_true')
    parser.add_argument('--contact-offset', type=float, default=.01)
    parser.add_argument('--rest-offset', type=float, default=0.)
    if '--prepare-collision-offsets-only' in sys.argv:
        from pxr import Usd
        args = parser.parse_args()
        stage = Usd.Stage.Open((args.output / 'usd_path.txt').read_text().strip())
        report = author_shared_collision_offsets(stage, args.output, args.contact_offset, args.rest_offset)
        (args.output / 'collision-offset-validation.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return
    if '--validate-only' in sys.argv:
        import xml.etree.ElementTree as ET
        from pxr import Usd
        args = parser.parse_args()
        tree = ET.parse(args.output / 'merged.urdf')
        manifest = json.loads((args.output / 'preparation.json').read_text())
        usd_path = (args.output / 'usd_path.txt').read_text().strip()
        stage = Usd.Stage.Open(usd_path)
        stage.SetEditTarget(stage.GetSessionLayer())
        report = restore_and_validate(stage, tree, manifest)
        if any(r['com_was_implicit'] or r['before_com_error'] > 1e-7 or r['before_tensor_error'] > 1e-6 for r in report['inertias']):
            raise ValueError('Delivered USD source inertias differ before validation')
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    if '--prepare-only' in sys.argv:
        args = parser.parse_args()
        _, manifest = prepare_urdf(args.output)
        print(json.dumps(manifest, indent=2))
        return
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    tree, manifest = prepare_urdf(args.output)
    launcher = AppLauncher(args)
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from pxr import Usd
        converter = UrdfConverter(UrdfConverterCfg(asset_path=manifest['merged_urdf'],
            usd_dir=str(args.output.resolve()), usd_file_name='robot.usd', force_usd_conversion=True,
            fix_base=False, merge_fixed_joints=False, self_collision=True,
            collision_type='Convex Hull', replace_cylinders_with_capsules=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(target_type='none')))
        stage = Usd.Stage.Open(converter.usd_path)
        author_source_physx_limits(stage, tree)
        offsets = author_shared_collision_offsets(stage, args.output, args.contact_offset, args.rest_offset)
        report = restore_and_validate(stage, tree, manifest)
        report['collision_offsets'] = offsets
        report['usd_path'] = converter.usd_path
        stage.GetRootLayer().Save()
        (args.output / 'conversion-validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (args.output / 'usd_path.txt').write_text(converter.usd_path + '\n')
        print('AS2_PIPER_USD', converter.usd_path, flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__ == '__main__':
    main()
