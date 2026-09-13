"""Independent AS2/Piper-H vendor URDF composition, CPU conversion and audit."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/as2_piper.json'
SOURCES = ROOT / 'third_party/as2_piper'


def numbers(value):
    return ' '.join(str(float(x)) for x in value)


def verify_sources(source_dir):
    """Refuse changed source bytes after the pinned fetch."""
    manifest = json.loads((source_dir / 'provenance.json').read_text())
    for entry in manifest['files']:
        path = source_dir / entry['path']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256'] or hashlib.sha1(
                f'blob {len(data)}\0'.encode() + data).hexdigest() != entry['git_blob']:
            raise ValueError(f'Source content changed: {path}')
    return manifest


def mount_plate_properties(plate):
    """The fixed, base-axis-aligned rectangular mounting plate only."""
    size = np.asarray(plate['size_m'], dtype=float)
    com = np.asarray(plate['com_base_m'], dtype=float)
    density = float(plate['density_kg_m3'])
    if (size.shape != (3,) or com.shape != (3,) or not np.isfinite(size).all()
            or not np.isfinite(com).all() or np.any(size <= 0)
            or not np.isfinite(density) or density <= 0):
        raise ValueError('Mount plate requires positive dimensions/density and finite base COM')
    mass = density * np.prod(size)
    inertia = np.diag(mass / 12 * (np.sum(size**2) - size**2))
    return mass, com, inertia


def merge_mount_plate_inertia(link, plate, *, merged_base_ratio=None):
    """Add the plate, or scale an already merged original base while fixing plate mass.

    First/second moments about the link origin are additive. Subtracting the
    known plate moments before scaling avoids scaling its density with AS2.
    """
    added, added_com, added_tensor = mount_plate_properties(plate)
    inertial = link.find('inertial')
    origin = inertial.find('origin')
    com = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    rotation = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    mass = float(inertial.find('mass').get('value'))
    keys = ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')
    xx, xy, xz, yy, yz, zz = (float(inertial.find('inertia').get(k)) for k in keys)
    tensor = rotation @ np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) @ rotation.T
    def parallel(m, c):
        return m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
    first = mass * com
    second = tensor + parallel(mass, com)
    plate_second = added_tensor + parallel(added, added_com)
    if merged_base_ratio is not None:
        ratio = float(merged_base_ratio)
        if not np.isfinite(ratio) or ratio <= 0 or mass <= added:
            raise ValueError('Expected positive original base mass and density ratio')
        mass = (mass - added) * ratio
        first = (first - added * added_com) * ratio
        second = (second - plate_second) * ratio
    mass += added
    com = (first + added * added_com) / mass
    tensor = second + plate_second - parallel(mass, com)
    eig = np.linalg.eigvalsh(tensor)
    if eig.min() <= 0 or eig[-1] > eig[0] + eig[1] + 1e-10:
        raise ValueError('Invalid combined base/plate inertia')
    inertial.find('mass').set('value', str(float(mass)))
    origin.set('xyz', numbers(com))
    origin.set('rpy', '0 0 0')
    for key, value in zip(keys, (tensor[0, 0], tensor[0, 1], tensor[0, 2], tensor[1, 1], tensor[1, 2], tensor[2, 2])):
        inertial.find('inertia').set(key, str(float(value)))


def assemble(source_dir=SOURCES, config=None):
    """Retain vendor inertials, joints and collisions; omit rendering-only visuals."""
    source_dir = Path(source_dir).resolve()
    config = config or json.loads(CONFIG.read_text())
    verify_sources(source_dir)
    root = ET.parse(source_dir / config['as2_urdf']).getroot()
    root.set('name', 'as2_piper_nominal')
    for extension in root.findall('mujoco'):
        root.remove(extension)
    arm = ET.parse(source_dir / config['piper_urdf']).getroot()
    if 'gripper' in config:
        gripper = ET.parse(source_dir / config['gripper']['source_xacro']).getroot()
        opening = config['gripper']['opening_m']
        for element in gripper:
            if element.tag == '{http://www.ros.org/wiki/xacro}include':
                # The pinned file has one plain include of the selected bare arm.
                if not element.get('filename', '').endswith('/piper_h/urdf/piper_h_description.urdf'):
                    raise ValueError('Unexpected stock-gripper include')
                continue
            if element.tag not in ('link', 'joint'):
                raise ValueError(f'Unsupported stock-gripper source element: {element.tag}')
            if element.tag == 'joint' and element.get('type') == 'prismatic':
                mimic = element.find('mimic')
                position = opening
                if mimic is not None:
                    if mimic.get('joint') != 'gripper':
                        raise ValueError('Unexpected stock-gripper mimic target')
                    position = opening * float(mimic.get('multiplier', '1')) + float(mimic.get('offset', '0'))
                limit = element.find('limit')
                if not float(limit.get('lower')) <= position <= float(limit.get('upper')):
                    raise ValueError('Gripper opening is outside vendor source limits')
                origin = element.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
                axis = np.fromstring(element.find('axis').get('xyz'), sep=' ')
                xyz += Rotation.from_euler('xyz', rpy).apply(axis * position)
                origin.set('xyz', numbers(xyz))
                element.set('type', 'fixed')
                for tag in ('axis', 'limit', 'mimic'):
                    for child in element.findall(tag):
                        element.remove(child)
            arm.append(copy.deepcopy(element))
    for element in arm:
        if element.tag == 'link' and element.get('name') != 'world':
            element.set('name', 'piper_' + element.get('name'))
            root.append(copy.deepcopy(element))
        elif element.tag == 'joint' and element.get('name') != 'world_to_base_link':
            element.set('name', 'piper_' + element.get('name'))
            for tag in ('parent', 'child'):
                reference = element.find(tag)
                reference.set('link', 'piper_' + reference.get('link'))
            root.append(copy.deepcopy(element))
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            link.remove(visual)
        for mesh in link.findall('.//mesh'):
            filename = mesh.get('filename')
            if filename.startswith('package://agx_arm_description/agx_arm_urdf/'):
                path = source_dir / 'agilex_piper_h' / filename.split('agx_arm_urdf/', 1)[1]
            else:
                path = (source_dir / config['as2_urdf']).parent / filename
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            mesh.set('filename', str(path))
    if 'rail_mount' in config:
        mounting = config['rail_mount']
        base = root.find("link[@name='base_link']")
        # Exported rail vertices already use base_link axes and metres.
        if len(mounting['rails']) != 2:
            raise ValueError('Expected the two registered AS2 rails')
        for rail in mounting['rails']:
            path = Path(rail['mesh_path']).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest() != rail['sha256']:
                raise ValueError(f'Changed registered rail mesh: {path}')
            collision = ET.SubElement(base, 'collision', name=rail['name'])
            ET.SubElement(collision, 'origin', xyz='0 0 0', rpy='0 0 0')
            ET.SubElement(ET.SubElement(collision, 'geometry'), 'mesh', filename=str(path))
        plate = mounting['plate']
        mount_plate_properties(plate)
        collision = ET.SubElement(base, 'collision', name='mount_plate')
        ET.SubElement(collision, 'origin', xyz=numbers(plate['com_base_m']), rpy='0 0 0')
        ET.SubElement(ET.SubElement(collision, 'geometry'), 'box', size=numbers(plate['size_m']))
        merge_mount_plate_inertia(base, plate)
    mount = config['mount']
    joint = ET.SubElement(root, 'joint', name='piper_mount', type='fixed')
    ET.SubElement(joint, 'parent', link=mount['parent'])
    ET.SubElement(joint, 'child', link=mount['child'])
    ET.SubElement(joint, 'origin', xyz=numbers(mount['xyz']), rpy=numbers(mount['rpy']))
    extension = ET.SubElement(root, 'mujoco')
    ET.SubElement(extension, 'compiler', discardvisual='true', fusestatic='false',
                  strippath='false', inertiafromgeom='false')
    return root


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def urdf_fk(root, positions, base_xyz, base_quat):
    """Independent URDF chain FK (not MuJoCo's imported transforms)."""
    poses = {'world': np.eye(4)}
    remaining = list(root.findall('joint'))
    while remaining:
        previous = len(remaining)
        for joint in remaining[:]:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent not in poses:
                continue
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            local = transform(xyz, rpy)
            if joint.get('type') == 'floating':
                local = transform(base_xyz)
                local[:3, :3] = Rotation.from_quat(np.asarray(base_quat)[[1, 2, 3, 0]]).as_matrix()
            elif joint.get('type') in ('revolute', 'prismatic'):
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                motion = np.eye(4)
                if joint.get('type') == 'revolute':
                    motion[:3, :3] = Rotation.from_rotvec(axis * positions[joint.get('name')]).as_matrix()
                else:
                    motion[:3, 3] = axis * positions[joint.get('name')]
                local = local @ motion
            elif joint.get('type') != 'fixed':
                raise ValueError(f'Unsupported source joint: {joint.attrib}')
            poses[child] = poses[parent] @ local
            remaining.remove(joint)
        if len(remaining) == previous:
            raise ValueError('Disconnected/cyclic source URDF')
    return poses


def validate(model, root, config, *, static_only=False):
    """Check reachable model against URDF FK, per-link inertials and joint limits."""
    import mujoco
    names = config['controlled_joint_order']
    if len(names) != 18 or len(set(names)) != 18 or model.nu != 18 or model.nq != 25 or model.nv != 24:
        raise ValueError('Expected floating base plus exactly 18 controlled joints')
    joints = {j.get('name'): j for j in root.findall('joint') if j.get('type') == 'revolute'}
    if set(joints) != set(names):
        raise ValueError('Controlled joint set differs from source joints')
    limits = {}
    for i, name in enumerate(names):
        joint = joints[name]
        bound = joint.find('limit')
        low, high = float(bound.get('lower')), float(bound.get('upper'))
        limits[name] = {key: float(bound.get(key)) for key in ('lower', 'upper', 'effort', 'velocity')}
        jid = model.joint(name).id
        if model.actuator_trnid[i, 0] != jid or not model.jnt_limited[jid]:
            raise ValueError(f'Actuator order or limit missing: {name}')
        np.testing.assert_allclose(model.jnt_range[jid], [low, high], atol=1e-5)
        np.testing.assert_allclose(model.actuator_ctrlrange[i], [-limits[name]['effort'], limits[name]['effort']])
        if not low <= config['default_joint_positions'][i] <= high:
            raise ValueError(f'Default position outside source limits: {name}')
    mass = 0.0
    inertia_error = 0.0
    for link in root.findall('link'):
        inertial = link.find('inertial')
        if inertial is None:
            continue
        name = link.get('name')
        expected_mass = float(inertial.find('mass').get('value'))
        mass += expected_mass
        body = model.body(name)
        np.testing.assert_allclose(body.mass[0], expected_mass, atol=1e-8)
        terms = {k: float(v) for k, v in inertial.find('inertia').attrib.items()}
        tensor = np.array([[terms['ixx'], terms['ixy'], terms['ixz']],
                           [terms['ixy'], terms['iyy'], terms['iyz']],
                           [terms['ixz'], terms['iyz'], terms['izz']]])
        origin = inertial.find('origin')
        rotation = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        tensor = rotation @ tensor @ rotation.T
        quaternion = body.iquat[[1, 2, 3, 0]]
        compiled_rotation = Rotation.from_quat(quaternion).as_matrix()
        compiled = compiled_rotation @ np.diag(body.inertia) @ compiled_rotation.T
        inertia_error = max(inertia_error, float(np.max(np.abs(compiled - tensor))))
        np.testing.assert_allclose(compiled, tensor, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(body.ipos, np.fromstring(origin.get('xyz', '0 0 0'), sep=' '), atol=1e-6)
        if np.linalg.eigvalsh(tensor).min() <= 0:
            raise ValueError(f'Nonpositive source inertia: {name}')
    np.testing.assert_allclose(model.body_mass.sum(), mass, atol=1e-8)
    data = mujoco.MjData(model)
    fk_error = orientation_error = 0.0
    # Default plus two interior source-limit poses expose rotation/axis/name mapping errors.
    poses = [config['default_joint_positions']] + [[limits[n]['lower'] * (1-f) + limits[n]['upper'] * f for n in names] for f in (.35, .65)]
    for values in poses:
        data.qpos[:3] = config['default_base_xyz']
        data.qpos[3:7] = config['default_base_quat_wxyz']
        for name, value in zip(names, values):
            data.qpos[model.joint(name).qposadr[0]] = value
        if static_only:
            mujoco.mj_kinematics(model, data)
        else:
            mujoco.mj_forward(model, data)
        independent = urdf_fk(root, dict(zip(names, values)), data.qpos[:3], data.qpos[3:7])
        for name, pose in independent.items():
            if name == 'world':
                continue
            body = data.body(name)
            fk_error = max(fk_error, float(np.max(np.abs(body.xpos - pose[:3, 3]))))
            orientation_error = max(orientation_error, float(np.max(np.abs(body.xmat.reshape(3, 3) - pose[:3, :3]))))
        tcp = independent[config['tcp']['body']] @ transform(config['tcp']['xyz'], config['tcp']['rpy'])
        np.testing.assert_allclose(data.site('tcp').xpos, tcp[:3, 3], atol=1e-5)
        np.testing.assert_allclose(data.site('tcp').xmat.reshape(3, 3), tcp[:3, :3], atol=1e-5)
    if fk_error > 1e-5 or orientation_error > 1e-5:
        raise ValueError(f'FK mismatch: {fk_error}, {orientation_error}')
    if static_only:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_kinematics(model, data)
        ground = model.geom('ground')
        if ground.type[0] != mujoco.mjtGeom.mjGEOM_PLANE or model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            raise ValueError('Expected original ground plane and implicitfast integrator')
        np.testing.assert_allclose(model.opt.timestep, config['timestep'], atol=0, rtol=0)
        np.testing.assert_allclose(ground.pos, [0, 0, 0], atol=0, rtol=0)
        np.testing.assert_allclose(model.actuator_gear[:, 0], 1, atol=0, rtol=0)
        np.testing.assert_allclose(model.actuator_gear[:, 1:], 0, atol=0, rtol=0)
        independent = urdf_fk(root, dict(zip(names, config['default_joint_positions'])), data.qpos[:3], data.qpos[3:7])
        expected_com = np.zeros(3)
        for link in root.findall('link'):
            inertial = link.find('inertial')
            if inertial is None:
                continue
            pose = independent[link.get('name')]
            local_com = np.fromstring(inertial.find('origin').get('xyz', '0 0 0'), sep=' ')
            expected_com += float(inertial.find('mass').get('value')) * (pose[:3, :3] @ local_com + pose[:3, 3])
        expected_com /= mass
        actual_com = (model.body_mass[:, None] * data.xipos).sum(0) / mass
        np.testing.assert_allclose(actual_com, expected_com, atol=1e-6, rtol=0)
        return dict(evidence='Static model compilation, source inertials, three-pose FK, ground and motor declarations. No mj_forward, integration, gravity probe or policy evaluation.',
                    mujoco_version=mujoco.__version__, nq=model.nq, nv=model.nv, nu=model.nu,
                    total_mass_kg=mass, max_inertia_tensor_error=inertia_error,
                    max_fk_position_error_m=fk_error, max_fk_rotation_matrix_error=orientation_error,
                    default_tcp_world_xyz=data.site('tcp').xpos.tolist(), default_com_world_xyz=actual_com.tolist(),
                    source_com_world_xyz=expected_com.tolist(), source_joint_limits=limits,
                    controlled_joint_order=names, ground=dict(name='ground', type='plane', position=ground.pos.tolist()),
                    timestep_s=model.opt.timestep, integrator='implicitfast', physics_steps=0)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    dense = np.zeros((model.nv, model.nv))
    # Both current and older supported MuJoCo Python bindings.
    try:
        mujoco.mj_fullM(model, data, dense)
    except TypeError:
        mujoco.mj_fullM(model, dense, data.qM)
    eigenvalues = np.linalg.eigvalsh(dense)
    if not np.isfinite(data.qacc).all() or eigenvalues.min() <= 0:
        raise ValueError('Nonfinite default dynamics or nonpositive generalized inertia')
    report = {'evidence': 'CPU compilation, source preservation, three-pose FK, gravity and direct torque mapping; no controller or policy transfer',
            'mujoco_version': mujoco.__version__, 'nq': model.nq, 'nv': model.nv, 'nu': model.nu,
            'total_mass_kg': mass, 'max_inertia_tensor_error': inertia_error,
            'max_fk_position_error_m': fk_error, 'max_fk_rotation_matrix_error': orientation_error,
            'default_tcp_world_xyz': data.site('tcp').xpos.tolist(),
            'default_contacts': [{'geom1': model.geom(c.geom1).name, 'geom2': model.geom(c.geom2).name, 'distance_m': float(c.dist)} for c in data.contact],
            'default_mass_matrix_min_eigenvalue': float(eigenvalues.min()), 'source_joint_limits': limits,
            'controlled_joint_order': names,
            'qpos_indices': [int(model.joint(n).qposadr[0]) for n in names],
            'qvel_indices': [int(model.joint(n).dofadr[0]) for n in names]}
    data.ctrl[:] = np.arange(1, 19) * .01
    mujoco.mj_forward(model, data)
    expected_force = np.zeros(model.nv)
    expected_force[report['qvel_indices']] = data.ctrl
    np.testing.assert_allclose(data.qfrc_actuator, expected_force, atol=1e-12)
    data.ctrl[:] = 1e6
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, model.actuator_ctrlrange[:, 1], atol=1e-12)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(10):
        mujoco.mj_step(model, data)
        if data.ncon:
            raise ValueError('Gravity probe initial pose reaches contact within 50 ms')
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    com_velocity = data.subtree_linvel[model.body('base_link').id]
    np.testing.assert_allclose(com_velocity, model.opt.gravity * data.time, atol=1e-6)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(w.number for w in data.warning):
        raise ValueError('Nonfinite integration or MuJoCo warning during gravity probe')
    report['cpu_probe'] = {'duration_s': float(data.time), 'steps': 10,
                           'simulation_frequency_hz': 1 / model.opt.timestep,
                           'unactuated_com_velocity_m_s': com_velocity.tolist(),
                           'expected_gravity_velocity_m_s': (model.opt.gravity * data.time).tolist(),
                           'direct_torque_mapping_and_effort_clipping': 'passed',
                           'warning_count': sum(w.number for w in data.warning)}
    return report


def build(output_dir, source_dir=SOURCES, config_path=CONFIG, *, source_root=None, static_only=False):
    import mujoco
    output_dir, source_dir = Path(output_dir).resolve(), Path(source_dir).resolve()
    config = json.loads(Path(config_path).read_text())
    root = assemble(source_dir, config) if source_root is None else copy.deepcopy(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    urdf = ET.tostring(root, encoding='unicode')
    imported = mujoco.MjModel.from_xml_string(urdf)
    with tempfile.TemporaryDirectory() as temp:
        intermediate = Path(temp) / 'imported.xml'
        mujoco.mj_saveLastXML(str(intermediate), imported)
        model_root = ET.parse(intermediate).getroot()
    if source_root is not None:
        # mj_saveLastXML rounds physical values to six significant digits.
        # Research density/payload variants need their imported inertials intact;
        # keep the historical default source construction byte behavior unchanged.
        for element in model_root.findall('.//body'):
            inertial = element.find('inertial')
            if inertial is None:
                continue
            original = imported.body(element.get('name'))
            inertial.set('mass', str(float(original.mass[0])))
            inertial.set('pos', numbers(original.ipos))
            inertial.set('quat', numbers(original.iquat))
            inertial.set('diaginertia', numbers(original.inertia))
    option = model_root.find('option')
    if option is None:
        option = ET.SubElement(model_root, 'option')
    option.set('timestep', str(config['timestep']))
    option.set('integrator', 'implicitfast')
    world = model_root.find('worldbody')
    ET.SubElement(world, 'geom', name='ground', type='plane', size='0 0 .1')
    body = model_root.find(f".//body[@name='{config['tcp']['body']}']")
    quat = Rotation.from_euler('xyz', config['tcp']['rpy']).as_quat()[[3, 0, 1, 2]]
    ET.SubElement(body, 'site', name='tcp', pos=numbers(config['tcp']['xyz']), quat=numbers(quat), size='.01')
    actuator = ET.SubElement(model_root, 'actuator')
    for name in config['controlled_joint_order']:
        effort = float(root.find(f"joint[@name='{name}']/limit").get('effort'))
        ET.SubElement(actuator, 'motor', name=name + '_torque', joint=name, gear='1',
                      ctrllimited='true', ctrlrange=numbers([-effort, effort]))
    preliminary = mujoco.MjModel.from_xml_string(ET.tostring(model_root, encoding='unicode'))
    qpos = preliminary.qpos0.copy()
    qpos[:3] = config['default_base_xyz']
    qpos[3:7] = config['default_base_quat_wxyz']
    for name, value in zip(config['controlled_joint_order'], config['default_joint_positions']):
        qpos[preliminary.joint(name).qposadr[0]] = value
    ET.SubElement(ET.SubElement(model_root, 'keyframe'), 'key', name='default', qpos=numbers(qpos))
    ET.indent(model_root)
    xml = ET.tostring(model_root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    report = validate(model, root, config, static_only=static_only)
    if source_root is None:
        source_as2 = ET.parse(source_dir / config['as2_urdf']).getroot()
        source_arm = ET.parse(source_dir / config['piper_urdf']).getroot()
        alternate = ET.parse(source_dir / 'unitree_as2_dynamics/unitree_robots/as2/as2.xml')
        report['component_masses_kg'] = {
            'as2_urdf_selected': sum(float(m.get('value')) for m in source_as2.findall('.//inertial/mass')),
            'piper_h_bare_urdf_selected': sum(float(m.get('value')) for m in source_arm.findall('.//inertial/mass')),
            'as2_vendor_mjcf_not_selected': sum(float(m.get('mass')) for m in alternate.findall('.//body/inertial'))}
        if 'gripper' in config:
            gripper = ET.parse(source_dir / config['gripper']['source_xacro']).getroot()
            report['component_masses_kg']['stock_gripper_with_flange_selected'] = sum(
                float(m.get('value')) for m in gripper.findall('.//inertial/mass'))
            report['frozen_gripper_source_joints'] = {
                j.get('name'): {'limit': dict(j.find('limit').attrib),
                               'mimic': dict(j.find('mimic').attrib) if j.find('mimic') is not None else None}
                for j in gripper.findall('joint') if j.get('type') == 'prismatic'}
        report['source_provenance'] = verify_sources(source_dir)
    else:
        report['source_provenance'] = {'mode': 'caller supplied AS2 source tree', 'input_tree_sha256': hashlib.sha256(ET.tostring(source_root)).hexdigest()}
    report['config'] = config
    (output_dir / 'robot.urdf').write_text(urdf + '\n')
    (output_dir / 'robot.xml').write_text(xml + '\n')
    report['artifact_sha256'] = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in ('robot.urdf', 'robot.xml')}
    (output_dir / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
