"""Build the official Go2 + ARX5 collision/inertial asset for independent MuJoCo."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import mujoco

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'third_party/umi-on-legs/mani-centric-wbc/resources/robots/go2_arx5/go2_arx5_finray_x85_z94.urdf'
DEFAULT_MODEL = ROOT / 'pawcerto/mujoco/go2_arx5.xml'


def build_model(output=DEFAULT_MODEL, urdf=URDF, timestep=0.005):
    urdf, output = Path(urdf), Path(output)
    robot = ET.parse(urdf).getroot()
    # The reference uses primitive and convex-mesh collisions. DAE visuals are
    # omitted here; explicit URDF masses/inertias, mounting and collisions remain.
    for link in robot.findall('link'):
        for visual in link.findall('visual'):
            link.remove(visual)
        inertial = link.find('inertial')
        if inertial is not None and float(inertial.find('mass').get('value')) == 0:
            link.remove(inertial)  # end_effector has no inertia element in source
    for mesh in robot.findall('.//mesh'):
        mesh.set('filename', str((urdf.parents[3] / mesh.get('filename')).resolve()))
    extension = ET.SubElement(robot, 'mujoco')
    ET.SubElement(extension, 'compiler', discardvisual='true', fusestatic='false',
                  strippath='false', inertiafromgeom='false')
    imported = mujoco.MjModel.from_xml_string(ET.tostring(robot).decode())
    with tempfile.TemporaryDirectory() as directory:
        intermediate = Path(directory) / 'import.xml'
        mujoco.mj_saveLastXML(str(intermediate), imported)
        model = ET.parse(intermediate).getroot()
    model.find('compiler').set('inertiafromgeom', 'false')
    # mj_saveLastXML rounds attributes; retain imported transforms and inertia
    # at float precision so independent FK and tensor comparisons stay faithful.
    def numbers(values):
        return ' '.join(format(float(x), '.17g') for x in values)
    for body_xml in model.findall('.//body'):
        body = imported.body(body_xml.get('name'))
        body_xml.set('pos', numbers(body.pos))
        body_xml.set('quat', numbers(body.quat))
        inertial = body_xml.find('inertial')
        if inertial is not None:
            inertial.set('pos', numbers(body.ipos))
            inertial.set('quat', numbers(body.iquat))
            inertial.set('diaginertia', numbers(body.inertia))
            inertial.set('mass', format(float(body.mass[0]), '.17g'))
    ET.SubElement(model, 'option', timestep=str(timestep), gravity='0 0 -9.81', integrator='implicitfast')
    world = model.find('worldbody')
    ET.SubElement(world.find("body[@name='base']"), 'freejoint', name='root')
    ET.SubElement(world, 'geom', name='ground', type='plane', size='0 0 0.1', friction='1 0.005 0.0001')
    # Official asset option replace_cylinder_with_capsule=true.
    for geom in model.findall('.//body/geom'):
        if geom.get('type') == 'cylinder':
            geom.set('type', 'capsule')
        geom.set('friction', '1 0.005 0.0001')
    ET.indent(model)
    xml = ET.tostring(model, encoding='unicode')
    mujoco.MjModel.from_xml_string(xml)  # compile before replacing output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(xml)
    return output
