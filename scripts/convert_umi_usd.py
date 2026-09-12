"""Import the original Go2+ARX5 asset into Isaac Lab without editing upstream."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


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
        np.testing.assert_allclose(float(api.GetMassAttr().Get()), mass, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(list(api.GetCenterOfMassAttr().Get()), com, rtol=1e-6, atol=1e-7)
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
        records.append({'body': name, 'source_tensor': tensor.tolist(),
                        'before_max_abs_error': float(np.max(abs(before - tensor))),
                        'after_max_abs_error': float(np.max(abs(after - tensor)))})
    return records


def main():
    import json
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'reference/isaac/go2_arx5')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    source_root = ROOT / 'third_party/umi-on-legs/mani-centric-wbc'
    tree = ET.parse(source_root / 'resources/robots/go2_arx5/go2_arx5_finray_x85_z94.urdf')
    for mesh in tree.findall('.//mesh'):
        path = source_root / mesh.attrib['filename']
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set('filename', str(path.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)
    urdf = args.output / 'go2_arx5.urdf'
    tree.write(urdf, encoding='utf-8', xml_declaration=True)
    launcher = AppLauncher(args)
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.asset.importer.urdf', True)
        from isaacsim.asset.importer.urdf.impl.urdf_utils import merge_fixed_joints
        # Gym preserves fixed frames carrying dont_collapse=true. The Lab 3 merger
        # does not understand that attribute, so protect only those original joints
        # during its existing merge operation, then restore their fixed type.
        # Keep collision-bearing fixed links. Massless marker-only frames remain
        # kinematic transforms: importing them as PhysX bodies invents 1 kg of mass.
        collision_links = {link.get('name') for link in tree.findall('link') if link.find('collision') is not None}
        protected = [j.attrib['name'] for j in tree.findall('joint')
                     if j.get('dont_collapse') == 'true' and j.find('child').get('link') in collision_links]
        for joint in tree.findall('joint'):
            if joint.get('name') in protected:
                joint.set('type', 'floating')
        tree.write(urdf, encoding='utf-8', xml_declaration=True)
        merged = args.output / 'go2_arx5_merged.urdf'
        merge_fixed_joints(str(urdf), str(merged))
        merged_tree = ET.parse(merged)
        for joint in merged_tree.findall('joint'):
            if joint.get('name') in protected:
                joint.set('type', 'fixed')
        merged_tree.write(merged, encoding='utf-8', xml_declaration=True)
        converter = UrdfConverter(UrdfConverterCfg(
            asset_path=str(merged), usd_dir=str(args.output), usd_file_name='robot.usd',
            force_usd_conversion=True, fix_base=False, merge_fixed_joints=False,
            replace_cylinders_with_capsules=True, self_collision=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(target_type='none')))
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = Usd.Stage.Open(converter.usd_path)
        # Lab 3 accepts but ignores replace_cylinders_with_capsules. Apply it on
        # collision primitives in the generated USD, leaving visual shapes alone.
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Cylinder) and prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.SetTypeName('Capsule')
        inertia_validation = _restore_source_inertias(stage, merged_tree)
        stage.GetRootLayer().Save()
        (args.output / 'inertia-validation.json').write_text(json.dumps(
            {'status': 'source_COM_tensors_verified', 'bodies': inertia_validation}, indent=2) + '\n')
        (args.output / 'usd_path.txt').write_text(converter.usd_path + '\n')
        print('UMI_USD', converter.usd_path, flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        import sys
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__ == '__main__':
    main()
