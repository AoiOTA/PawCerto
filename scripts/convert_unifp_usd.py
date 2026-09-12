"""Convert fixed UniFP source with the official Isaac Lab URDF importer."""
import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def source_tree():
    source = ROOT / 'third_party/unifp-reference/resources/robots/b2z1/b2z1.urdf'
    tree = ET.parse(source)
    for mesh in tree.findall('.//mesh'):
        mesh.set('filename', str((source.parent / mesh.get('filename')).resolve()))
    return tree
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=ROOT / 'reference/isaac/b2_z1')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(args)
try:
    import omni.kit.app
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.asset.importer.urdf', True)
    from isaacsim.asset.importer.urdf.impl.urdf_utils import merge_fixed_joints
    tree = source_tree()
    protected = [j.get('name') for j in tree.findall('joint') if j.get('dont_collapse') == 'true']
    for joint in tree.findall('joint'):
        if joint.get('name') in protected:
            joint.set('type', 'floating')
    original = args.output / 'source.urdf'
    tree.write(original, encoding='utf-8', xml_declaration=True)
    merged = args.output / 'merged.urdf'
    merge_fixed_joints(str(original), str(merged))
    tree = ET.parse(merged)
    for joint in tree.findall('joint'):
        if joint.get('name') in protected:
            joint.set('type', 'fixed')
    tree.write(merged, encoding='utf-8', xml_declaration=True)
    converter = UrdfConverter(UrdfConverterCfg(asset_path=str(merged), usd_dir=str(args.output),
        usd_file_name='robot.usd', force_usd_conversion=True, fix_base=False, merge_fixed_joints=False,
        replace_cylinders_with_capsules=True, self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(target_type='none')))
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    stage = Usd.Stage.Open(converter.usd_path)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Cylinder) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.SetTypeName('Capsule')
    stage.GetRootLayer().Save()
    # Author the runtime collision offsets in the generated shared geometry
    # layers. Editing every cloned instance would remove geometry sharing and
    # exhaust host RAM at the original 4096-environment training scale.
    output_root = args.output.resolve()
    for layer in stage.GetUsedLayers():
        if not layer.realPath or not Path(layer.realPath).resolve().is_relative_to(output_root):
            continue
        layer_stage = Usd.Stage.Open(layer)
        changed = False
        for prim in layer_stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr(.01)
                collision.CreateRestOffsetAttr(0.)
                changed = True
        if changed:
            layer.Save()
    (args.output / 'usd_path.txt').write_text(converter.usd_path + '\n')
    print('UNIFP_USD', converter.usd_path, flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
