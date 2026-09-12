"""Import the original Go2+ARX5 asset into Isaac Lab without editing upstream."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
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
    stage.GetRootLayer().Save()
    (args.output / 'usd_path.txt').write_text(converter.usd_path + '\n')
    print('UMI_USD', converter.usd_path, flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    import sys
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
