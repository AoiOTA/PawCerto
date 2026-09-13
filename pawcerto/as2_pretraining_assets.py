"""Finite, same-topology AS2/Piper research assets; CPU preparation only.

Geometry scales isotropically at constant density. Mass-only ratios describe
fixed-geometry density changes. Payload is merged into the existing TCP body.
Neither operation identifies hardware or certifies load/reach capability.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from pawcerto.mujoco.as2_piper_asset import CONFIG, numbers, transform, urdf_fk, merge_mount_plate_inertia
from pawcerto.robots.as2_piper import prepare_urdf

ROOT = Path(__file__).resolve().parents[1]
FAMILY_CONFIG = ROOT / 'configs/as2_pretraining_family.json'
INERTIA_KEYS = ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')


def _vec(node, key, default='0 0 0'):
    return np.fromstring(node.get(key, default) if node is not None else default, sep=' ')


def inertial_properties(link):
    """Mass, COM and COM inertia expressed in link axes (not origin inertia)."""
    inertial = link.find('inertial')
    origin = inertial.find('origin')
    entries = inertial.find('inertia').attrib
    xx, xy, xz, yy, yz, zz = (float(entries[k]) for k in INERTIA_KEYS)
    local = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    rot = Rotation.from_euler('xyz', _vec(origin, 'rpy')).as_matrix()
    return float(inertial.find('mass').get('value')), _vec(origin, 'xyz'), rot @ local @ rot.T


def _valid_inertia(matrix, allow_zero=False):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
        raise ValueError('Inertia must be a finite symmetric 3x3 tensor')
    values = np.linalg.eigvalsh(matrix)
    if (values.min() < -1e-12 if allow_zero else values.min() <= 0) or values[-1] > values[0] + values[1] + 1e-10:
        raise ValueError('Inertia violates positive mass-distribution constraints')
    return matrix


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _scale_origin(node, scale):
    origin = node.find('origin')
    if origin is not None:
        origin.set('xyz', numbers(_vec(origin, 'xyz') * scale))


def _scale_link(link, scale, mass_ratio):
    if scale == 1 and mass_ratio == 1:
        return
    inertial = link.find('inertial')
    if scale != 1:
        _scale_origin(inertial, scale)
        for tag in ('collision', 'visual'):
            for shape in link.findall(tag):
                _scale_origin(shape, scale)
                geom = shape.find('geometry')[0]
                if geom.tag == 'box':
                    geom.set('size', numbers(_vec(geom, 'size') * scale))
                elif geom.tag == 'sphere':
                    geom.set('radius', str(float(geom.get('radius')) * scale))
                elif geom.tag == 'cylinder':
                    for key in ('radius', 'length'):
                        geom.set(key, str(float(geom.get(key)) * scale))
                elif geom.tag == 'mesh':
                    geom.set('scale', numbers(_vec(geom, 'scale', '1 1 1') * scale))
                else:
                    raise ValueError(f'Unsupported AS2 geometry {geom.tag}')
    mass = inertial.find('mass')
    mass.set('value', str(float(mass.get('value')) * mass_ratio * scale**3))
    inertia = inertial.find('inertia')
    for key in INERTIA_KEYS:
        inertia.set(key, str(float(inertia.get(key)) * mass_ratio * scale**5))


def merge_payload(link, tcp, payload):
    """Combine a TCP-frame payload with link inertia using parallel axes."""
    unknown = set(payload) - {'mass_kg', 'model', 'com_tcp_m', 'inertia_com_tcp_kg_m2'}
    if unknown:
        raise ValueError(f'Unknown payload fields: {sorted(unknown)}')
    added = float(payload['mass_kg'])
    if not np.isfinite(added) or added < 0:
        raise ValueError('Payload mass must be finite and nonnegative')
    kind = payload.get('model', 'point')
    offset = np.asarray(payload.get('com_tcp_m', [0, 0, 0]), dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('Payload COM must be a finite 3-vector')
    inertia = _valid_inertia(payload.get('inertia_com_tcp_kg_m2', np.zeros((3, 3))), allow_zero=True)
    if kind not in ('point', 'rigid') or (kind == 'point' and np.any(inertia != 0)):
        raise ValueError('Point payload has zero own inertia; use rigid for supplied inertia')
    if added == 0:
        if np.any(inertia != 0):
            raise ValueError('Zero-mass payload cannot have nonzero inertia')
        return
    if kind == 'rigid' and not np.any(inertia):
        raise ValueError('Rigid payload requires supplied nonzero COM inertia')
    m, c, tensor = inertial_properties(link)
    tcp_pose = transform(tcp['xyz'], tcp['rpy'])
    added_com = tcp_pose[:3, 3] + tcp_pose[:3, :3] @ offset
    combined_com = (m*c + added*added_com) / (m+added)
    def parallel(mass, d):
        return mass * (np.dot(d, d)*np.eye(3) - np.outer(d, d))
    combined = tensor + parallel(m, c-combined_com) + tcp_pose[:3, :3] @ inertia @ tcp_pose[:3, :3].T + parallel(added, added_com-combined_com)
    _valid_inertia(combined)
    inertial = link.find('inertial')
    inertial.find('mass').set('value', str(m+added))
    origin = inertial.find('origin')
    if origin is None:
        origin = ET.SubElement(inertial, 'origin')
    origin.set('xyz', numbers(combined_com))
    origin.set('rpy', '0 0 0')
    for key, value in zip(INERTIA_KEYS, (combined[0, 0], combined[0, 1], combined[0, 2], combined[1, 1], combined[1, 2], combined[2, 2])):
        inertial.find('inertia').set(key, str(value))


def transform_variant(root, parameters, nominal):
    """Return a transformed copy; never mutate the nominal input tree."""
    unknown = set(parameters) - {'as2_mass_ratio', 'base_mass_ratio', 'leg_scale', 'piper_mass_ratios', 'payload', 'link_com_offsets_m'}
    if unknown:
        raise ValueError(f'Unknown variant parameters: {sorted(unknown)}')
    as2_ratio = _positive(parameters.get('as2_mass_ratio', 1), 'as2_mass_ratio')
    base_ratio = _positive(parameters.get('base_mass_ratio', 1), 'base_mass_ratio')
    leg_scale = _positive(parameters.get('leg_scale', 1), 'leg_scale')
    ratios = parameters.get('piper_mass_ratios', {})
    out = deepcopy(root)
    links = {l.get('name'): l for l in out.findall('link')}
    offsets = parameters.get('link_com_offsets_m', {})
    if not isinstance(offsets, dict):
        raise ValueError('link_com_offsets_m must map physical link names to local metre offsets')
    physical = {n for n, l in links.items() if l.find('inertial/mass') is not None
                and float(l.find('inertial/mass').get('value')) > 0}
    if set(offsets) - physical:
        raise ValueError(f'Unknown positive-mass COM links: {sorted(set(offsets) - physical)}')
    offsets = {name: np.asarray(value, dtype=float) for name, value in offsets.items()}
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in offsets.values()):
        raise ValueError('Each link COM offset must be a finite 3-vector in local metres')
    physical_arm = {n for n,l in links.items() if n.startswith('piper_') and l.find('inertial') is not None}
    if set(ratios) - physical_arm:
        raise ValueError(f'Unknown positive-mass Piper links: {sorted(set(ratios)-physical_arm)}')
    leg_names = {f'{leg}_{part}' for leg in ('FL','FR','RL','RR') for part in ('hip','thigh','calf','foot')}
    if not leg_names <= links.keys():
        raise ValueError('Expected AS2 four hip/thigh/calf/foot links')
    for name, link in links.items():
        if link.find('inertial') is None:
            continue
        scale = leg_scale if name in leg_names else 1
        ratio = base_ratio if name == 'base_link' else _positive(ratios.get(name, 1), f'{name} mass ratio')
        if not name.startswith('piper_'):
            ratio *= as2_ratio
        if name == 'base_link' and 'rail_mount' in nominal and (ratio != 1 or name in offsets):
            merge_mount_plate_inertia(link, nominal['rail_mount']['plate'], merged_base_ratio=ratio,
                                     original_base_com_offset_m=offsets.get(name))
        else:
            _scale_link(link, scale, ratio)
            if name in offsets:
                # Applied after geometry scaling, in final link axes; retain COM tensor.
                origin = link.find('inertial/origin')
                if origin is None:
                    origin = ET.SubElement(link.find('inertial'), 'origin')
                origin.set('xyz', numbers(_vec(origin, 'xyz') + offsets[name]))
    if leg_scale != 1:
        for joint in out.findall('joint'):
            if joint.find('parent').get('link') in leg_names:
                _scale_origin(joint, leg_scale)
    if parameters.get('payload'):
        merge_payload(links[nominal['tcp']['body']], nominal['tcp'], parameters['payload'])
    return out


def _interface(root):
    joints = []
    for j in root.findall('joint'):
        joints.append((dict(j.attrib), [ET.tostring(e, encoding='unicode') for e in j if e.tag != 'origin']))
    collisions = [(l.get('name'), c.get('name'), i, c.find('geometry')[0].tag)
                  for l in root.findall('link') for i,c in enumerate(l.findall('collision'))]
    return [l.get('name') for l in root.findall('link')], joints, collisions


def audit_variant(source, merged, original_source, original_merged, nominal):
    if _interface(source) != _interface(original_source) or _interface(merged) != _interface(original_merged):
        raise ValueError('Variant changed body/joint/collision interface or source joint constraints')
    poses = urdf_fk(source, dict(zip(nominal['controlled_joint_order'], nominal['default_joint_positions'])), [0,0,0], [1,0,0,0])
    records, total, weighted = {}, 0., np.zeros(3)
    for link in merged.findall('link'):
        mass, com, inertia = inertial_properties(link)
        _positive(mass, 'mass')
        _valid_inertia(inertia)
        if not np.isfinite(com).all():
            raise ValueError('Nonfinite link COM')
        records[link.get('name')] = dict(mass_kg=mass, com_local_m=com.tolist(), inertia_com_link_kg_m2=inertia.tolist())
        total += mass
        pose = poses[link.get('name')]
        weighted += mass * (pose[:3,:3] @ com + pose[:3,3])
        counterpart = source.find(f"link[@name='{link.get('name')}']")
        for actual, expected in zip(inertial_properties(counterpart), (mass, com, inertia)):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    tcp = poses[nominal['tcp']['body']] @ transform(nominal['tcp']['xyz'], nominal['tcp']['rpy'])
    return dict(total_mass_kg=total, per_link=records,
                nominal_fk=dict(frame='base_link', tcp_pose=tcp.tolist(), whole_com_m=(weighted/total).tolist(),
                    feet_m={f'{leg}_foot': poses[f'{leg}_foot'][:3,3].tolist() for leg in ('FR','FL','RR','RL')}))


def build_family(output, config_path=FAMILY_CONFIG, *, nominal_config_path=CONFIG):
    output = Path(output).resolve()
    config_path = Path(config_path).resolve()
    family = json.loads(config_path.read_text())
    if family.get('schema_version') != 1 or set(family) != {'schema_version','description','parameter_ranges','variants'}:
        raise ValueError('Expected finite AS2 family schema_version 1')
    entries = family['variants']
    names = [v['name'] for v in entries]
    if not entries or names[0] != 'nominal' or entries[0]['parameters'] or len(set(names)) != len(names):
        raise ValueError('Family starts with unchanged nominal and has unique names')
    for entry in entries:
        if set(entry) != {'name','assumption','parameters'} or not entry['assumption']:
            raise ValueError('Each variant requires name, assumption and parameters')
        if not entry['name'].replace('_','').isalnum() or not entry['name'].isascii():
            raise ValueError('Variant name must be an ASCII alphanumeric/underscore directory name')
    nominal_config_path = Path(nominal_config_path).resolve()
    required_assembly = family['parameter_ranges'].get('source', {}).get('assembly_sha256')
    if required_assembly is not None and hashlib.sha256(nominal_config_path.read_bytes()).hexdigest() != required_assembly:
        raise ValueError('Selected nominal assembly differs from this target distribution; pass its matching --nominal-config')
    nominal = json.loads(nominal_config_path.read_text())
    original_tree, preparation = prepare_urdf(output/'nominal', config_path=nominal_config_path)
    original_merged = original_tree.getroot()
    original_source = ET.parse(output/'nominal/source.urdf').getroot()
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = dict(schema_version=1, description=family['description'], parameter_ranges=family['parameter_ranges'],
        source=dict(family_config_path=str(config_path), family_config_sha256=digest(config_path),
                    nominal_config_sha256=digest(nominal_config_path),
                    nominal_source_sha256=digest(output/'nominal/source.urdf')),
        joint_names=nominal['controlled_joint_order'], body_names=preparation['body_names'],
        collision_count=len(original_merged.findall('.//collision')),
        default_joint_positions=nominal['default_joint_positions'], tcp=nominal['tcp'],
        default_base_xyz=nominal['default_base_xyz'], variants=[],
        evidence='CPU URDF geometry/inertia/FK and unchanged interface; not converted USD, training, calibrated hardware or WBC acceptance.')
    if nominal_config_path != CONFIG.resolve():
        result['source']['nominal_config_path'] = str(nominal_config_path)
    for entry in entries:
        folder = output/entry['name']; folder.mkdir(parents=True, exist_ok=True)
        source = transform_variant(original_source, entry['parameters'], nominal)
        merged = transform_variant(original_merged, entry['parameters'], nominal)
        audit = audit_variant(source, merged, original_source, original_merged, nominal)
        if entry['name'] == 'nominal':
            assert ET.tostring(source) == ET.tostring(original_source)
            assert ET.tostring(merged) == ET.tostring(original_merged)
        else:
            for filename, tree in [('source.urdf',source),('merged.urdf',merged)]:
                ET.ElementTree(tree).write(folder/filename, encoding='utf-8', xml_declaration=True)
        prepared = dict(preparation, merged_urdf=str(folder/'merged.urdf'), mass_kg=audit['total_mass_kg'],
            collision_policy='Same source shape count/types; per-leg isotropic dimension scaling when selected',
            inertial_policy='Source tensors with specified density/geometry scaling and parallel-axis payload merge')
        (folder/'preparation.json').write_text(json.dumps(prepared,indent=2)+'\n')
        result['variants'].append(dict(entry, urdf_path=str(folder/'merged.urdf'), merged_urdf_path=str(folder/'merged.urdf'), usd_path=None, source_urdf_path=str(folder/'source.urdf'),
            urdf_sha256=digest(folder/'merged.urdf'), source_urdf_sha256=digest(folder/'source.urdf'),
            preparation_path=str(folder/'preparation.json'), **audit))
    (output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def build_mujoco_family(output):
    """Convert an existing frozen family using the original AS2 ground/motor path.

    Record a separate manifest; never rewrite the Lab/source family manifest.
    """
    from pawcerto.mujoco.as2_piper_asset import build
    output = Path(output).resolve()
    manifest_path = output/'manifest.json'
    original_bytes = manifest_path.read_bytes()
    family = json.loads(original_bytes)
    nominal_config_path = Path(family['source'].get('nominal_config_path', CONFIG))
    if hashlib.sha256(nominal_config_path.read_bytes()).hexdigest() != family['source']['nominal_config_sha256']:
        raise ValueError('Changed nominal assembly configuration in AS2 family')
    records = []
    for entry in family['variants']:
        source_path = Path(entry['source_urdf_path'])
        source_bytes = source_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != entry['source_urdf_sha256']:
            raise ValueError(f'Changed family source: {source_path}')
        folder = source_path.parent/'mujoco'
        report = build(folder, config_path=nominal_config_path, source_root=ET.fromstring(source_bytes), static_only=True)
        if source_path.read_bytes() != source_bytes:
            raise ValueError(f'Family source changed during conversion: {source_path}')
        np.testing.assert_allclose(report['total_mass_kg'], entry['total_mass_kg'], atol=1e-8, rtol=0)
        records.append(dict(name=entry['name'], source_urdf_path=str(source_path), source_urdf_sha256=entry['source_urdf_sha256'],
            mujoco_path=str(folder/'robot.xml'), mujoco_sha256=report['artifact_sha256']['robot.xml'],
            validation_path=str(folder/'validation.json'), total_mass_kg=report['total_mass_kg'],
            default_com_world_xyz=report['default_com_world_xyz'], physics_steps=0))
    if manifest_path.read_bytes() != original_bytes:
        raise ValueError('Original family manifest changed during conversion')
    result = dict(schema_version=1, source_manifest_path=str(manifest_path),
                  source_manifest_sha256=hashlib.sha256(original_bytes).hexdigest(), variants=records,
                  evidence='Original AS2 MuJoCo ground/TCP/motor/keyframe model construction with supplied source trees; static checks only. Default runtime configuration not changed.')
    (output/'mujoco-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
