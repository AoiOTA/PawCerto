"""Go1+ARX5 source asset and independent MuJoCo conversion."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'third_party/roboduet-reference/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf'
JOINT_NAMES = [f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'RL', 'RR') for joint in ('hip', 'thigh', 'calf')] + [f'zarx_j{i}' for i in range(1, 9)]
DEFAULT_POS = [.1, .8, -1.5, -.1, .8, -1.5, .1, 1., -1.5, -.1, 1., -1.5, 0., .8, .8, 0., 0., 0., 0., 0.]


def source_tree(urdf=URDF):
    urdf = Path(urdf)
    tree = ET.parse(urdf)
    for mesh in tree.findall('.//mesh'):
        path = (urdf.parent / mesh.get('filename')).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set('filename', str(path))
    return tree


def build_model(output, urdf=URDF, timestep=.005):
    """Independent FK/mass asset; camera inertia uses proportional scaling.

    Gym recomputeInertia is measured separately; this is not its equivalence claim.
    """
    import mujoco
    import trimesh
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = source_tree(urdf)
    root = tree.getroot()
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            link.remove(visual)
    for mesh in root.findall('.//mesh'):
        path = Path(mesh.get('filename'))
        if path.suffix.lower() == '.dae':
            target = output.parent / (path.stem + '.obj')
            loaded = trimesh.load(path, force='mesh')
            loaded.export(target)
            mesh.set('filename', str(target.resolve()))
    ee = root.find("link[@name='zarx_body6']/inertial")
    mass = float(ee.find('mass').get('value'))
    ee.find('mass').set('value', str(mass + .1))
    for key, value in ee.find('inertia').attrib.items():
        ee.find('inertia').set(key, str(float(value) * (mass + .1) / mass))
    ext = ET.SubElement(root, 'mujoco')
    ET.SubElement(ext, 'compiler', discardvisual='true', fusestatic='false', strippath='false', inertiafromgeom='false')
    imported = mujoco.MjModel.from_xml_string(ET.tostring(root).decode())
    with tempfile.TemporaryDirectory() as temp:
        intermediate = Path(temp) / 'model.xml'
        mujoco.mj_saveLastXML(str(intermediate), imported)
        model = ET.parse(intermediate).getroot()
    # Merge massless base into a floating mount body while retaining named trunk.
    base = model.find("worldbody/body[@name='base']")
    # MuJoCo permits massless ancestors with a massive fixed descendant.
    ET.SubElement(base, 'freejoint', name='root')
    ET.SubElement(model, 'option', timestep=str(timestep), gravity='0 0 -9.81', integrator='implicitfast')
    ET.SubElement(model.find('worldbody'), 'geom', name='ground', type='plane', size='0 0 .1', friction='1 .005 .0001')
    for geom in model.findall('.//body/geom'):
        if geom.get('type') == 'cylinder':
            geom.set('type', 'capsule')
    ET.indent(model)
    xml = ET.tostring(model, encoding='unicode')
    mujoco.MjModel.from_xml_string(xml)
    output.write_text(xml)
    return output
