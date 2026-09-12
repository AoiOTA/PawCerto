"""Prepare/convert original DeepWBC with the official unmodified Lab importer."""
import argparse
from pathlib import Path
import sys
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pawcerto.isaac.deepwbc_runtime import prepare_urdf
from pawcerto.methods.deepwbc.config import default_config


def _restore_source_inertias(stage, source_tree):
    """Write source COM tensors through USD's principal-moment representation.

    Some importer versions transpose the principal-axis basis. Reconstruct from
    the source tensor instead of assuming every imported quaternion needs an
    inverse; an importer that already writes the right tensor stays equivalent.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Gf, UsdPhysics

    source = {}
    for link in source_tree.findall('link'):
        inertial = link.find('inertial')
        if inertial is None or float(inertial.find('mass').get('value')) == 0:
            continue
        origin = inertial.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        inertia = inertial.find('inertia')
        xx, xy, xz, yy, yz, zz = [float(inertia.get(k)) for k in ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')]
        rotation = Rotation.from_euler('xyz', rpy).as_matrix()
        tensor = rotation @ np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) @ rotation.T
        moments, axes = np.linalg.eigh(tensor)
        if not np.isfinite(tensor).all() or not np.all(moments > 0):
            raise ValueError(f'Invalid source inertia tensor for {link.get("name")}')
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        source[link.get('name')] = (float(inertial.find('mass').get('value')), xyz, tensor, moments, axes)

    bodies = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            name = prim.GetName()
            if name in bodies:
                raise ValueError(f'Duplicate USD mass body name: {name}')
            bodies[name] = UsdPhysics.MassAPI(prim)
    if set(bodies) != set(source):
        raise ValueError(f'Source/USD mass bodies differ: {set(source) ^ set(bodies)}')

    records = []
    for name, (mass, com, tensor, moments, axes) in source.items():
        api = bodies[name]
        np.testing.assert_allclose(float(api.GetMassAttr().Get()), mass, rtol=1e-6, atol=1e-7, err_msg=f'{name}: mass')
        com_attr = api.GetCenterOfMassAttr()
        com_was_implicit = not com_attr.HasAuthoredValueOpinion()
        if com_was_implicit:
            # URDF omitted origin means zero; USD's unauthored -inf means auto.
            # Author the actual source COM rather than letting geometry infer it.
            com_attr.Set(Gf.Vec3f(*map(float, com)))
        np.testing.assert_allclose(list(com_attr.Get()), com, rtol=1e-6, atol=1e-7, err_msg=f'{name}: COM')
        old = api.GetPrincipalAxesAttr().Get()
        old_rotation = Rotation.from_quat([*old.GetImaginary(), old.GetReal()]).as_matrix()
        before = old_rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ old_rotation.T
        quat = Rotation.from_matrix(axes).as_quat()
        api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, moments)))
        api.GetPrincipalAxesAttr().Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))
        written = api.GetPrincipalAxesAttr().Get()
        written_rotation = Rotation.from_quat([*written.GetImaginary(), written.GetReal()]).as_matrix()
        after = written_rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ written_rotation.T
        np.testing.assert_allclose(after, tensor, rtol=1e-5, atol=1e-8)
        records.append({'body': name, 'source_com': com.tolist(), 'com_was_implicit': com_was_implicit, 'source_tensor': tensor.tolist(),
                        'before_max_abs_error': float(np.max(abs(before - tensor))),
                        'after_max_abs_error': float(np.max(abs(after - tensor)))})
    return records



def _restore_source_unbounded_joints(stage, source_tree):
    """Restore omitted revolute limits as disabled, matching measured Gym flags.

    Pinned widow_waist has asset and actor hasLimits=False in the original
    Gym probe. The stock URDF importer instead authors 0/0 into USD.
    """
    from pxr import UsdPhysics
    import math
    unbounded = {j.get('name') for j in source_tree.findall('joint')
                 if j.get('type') == 'revolute' and j.find('limit') is not None
                 and j.find('limit').get('lower') is None and j.find('limit').get('upper') is None}
    found = set()
    report = []
    for prim in stage.Traverse():
        if prim.GetName() in unbounded and prim.IsA(UsdPhysics.RevoluteJoint):
            name = prim.GetName()
            if name in found:
                raise ValueError(f'Duplicate unbounded USD joint: {name}')
            joint = UsdPhysics.RevoluteJoint(prim)
            before = [joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()]
            joint.GetLowerLimitAttr().Set(-math.inf)
            joint.GetUpperLimitAttr().Set(math.inf)
            if joint.GetLowerLimitAttr().Get() != -math.inf or joint.GetUpperLimitAttr().Get() != math.inf:
                raise ValueError(f'Failed to disable source-unbounded joint limits: {name}')
            report.append({'joint':name,'before_limits':before,'source_has_limits':False})
            found.add(name)
    if found != unbounded:
        raise ValueError(f'Missing source-unbounded joints in USD: {unbounded-found}')
    return report


def author_shared_collision_properties(usd_path, output_directory, *, contact_offset=.01, rest_offset=0., capsules=True):
    """Author once in owning geometry layers, preserving every instance.

    Prototype prims are read-only composed objects. Their defining layer is
    edited before cloning; never make instances unique to author a property.
    """
    from pxr import Usd, UsdPhysics, UsdGeom, Sdf
    output_directory = Path(output_directory).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f'Cannot open converted USD: {usd_path}')
    changed = []
    for layer in stage.GetUsedLayers():
        if layer.anonymous:
            continue
        layer_path = Path(layer.realPath).resolve()
        if not layer_path.is_relative_to(output_directory):
            continue
        own = Usd.Stage.Open(layer)
        for prim in own.Traverse():
            # Do not author a local opinion over somebody else's reference.
            stack = prim.GetPrimStack()
            if not stack or stack[0].layer != layer or prim.IsInstanceProxy():
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if capsules and prim.IsA(UsdGeom.Cylinder):
                    prim.SetTypeName('Capsule')
                prim.AddAppliedSchema('PhysxCollisionAPI')
                prim.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float,custom=False).Set(contact_offset)
                prim.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float,custom=False).Set(rest_offset)
                changed.append(str(prim.GetPath()))
        layer.Save()
    stage.Reload()
    collisions = proxies = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collisions += 1
            proxies += int(prim.IsInstanceProxy())
            if (abs(prim.GetAttribute('physxCollision:contactOffset').Get()-contact_offset) > 1e-7
                    or prim.GetAttribute('physxCollision:restOffset').Get() != rest_offset):
                raise ValueError(f'Collision prototype properties not composed: {prim.GetPath()}')
            if capsules and prim.IsA(UsdGeom.Cylinder):
                raise ValueError(f'Unconverted collision cylinder: {prim.GetPath()}')
    if not collisions or not proxies:
        raise ValueError('Expected collision-bearing instanceable geometry; do not de-instance to continue')
    return {'collision_prims':collisions,'instance_proxy_collisions':proxies,'authored_prims':changed}


def convert(output, launcher_args):
    from isaaclab.app import AppLauncher
    tree, manifest = prepare_urdf(output)
    launcher = AppLauncher(launcher_args)
    try:
        import omni.kit.app
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.asset.importer.urdf',True)
        from isaacsim.asset.importer.urdf.impl.urdf_utils import merge_fixed_joints
        # Honor original dont_collapse for feet, trunk and the massive EE marker.
        for joint in tree.findall('joint'):
            if joint.get('name') in manifest['protected_fixed_joints']:
                joint.set('type','floating')
        protected_path = output/'protected.urdf'
        tree.write(protected_path,encoding='utf-8',xml_declaration=True)
        merged = output/'merged.urdf'
        merge_fixed_joints(str(protected_path),str(merged))
        tree = ET.parse(merged)
        for joint in tree.findall('joint'):
            if joint.get('name') in manifest['protected_fixed_joints']:
                joint.set('type','fixed')
        tree.write(merged,encoding='utf-8',xml_declaration=True)
        cfg = default_config()
        converter = UrdfConverter(UrdfConverterCfg(asset_path=str(merged),usd_dir=str(output),
            usd_file_name='robot.usd',force_usd_conversion=True,fix_base=False,merge_fixed_joints=False,
            self_collision=True,link_density=cfg['asset']['density'],
            joint_drive=UrdfConverterCfg.JointDriveCfg(target_type='none')))
        report = author_shared_collision_properties(converter.usd_path,output,
            contact_offset=cfg['sim']['physx']['contact_offset'],rest_offset=cfg['sim']['physx']['rest_offset'],
            capsules=cfg['asset']['replace_cylinder_with_capsule'])
        from pxr import Usd
        stage = Usd.Stage.Open(converter.usd_path)
        report['inertias'] = _restore_source_inertias(stage,tree)
        report['unbounded_joints'] = _restore_source_unbounded_joints(stage,tree)
        stage.GetRootLayer().Save()
        (output/'conversion-validation.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'usd_path.txt').write_text(converter.usd_path+'\n')
        print('DEEPWBC_USD',converter.usd_path,flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


def main():
    # This branch does not import IsaacLab or start a simulator.
    if '--prepare-only' in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--prepare-only',action='store_true')
        parser.add_argument('--output',type=Path,default=ROOT/'reference/isaac/go1_widowx')
        args = parser.parse_args()
        _,manifest = prepare_urdf(args.output)
        print(json.dumps(manifest,indent=2))
        return
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reference/isaac/go1_widowx')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    convert(args.output,args)


if __name__ == '__main__':
    main()
