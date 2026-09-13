"""Fixed mount inertia and the real static MuJoCo assembly consumer."""
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import trimesh

from pawcerto.as2_pretraining_assets import inertial_properties, transform_variant
from pawcerto.mujoco.as2_piper_asset import CONFIG, assemble, build


@pytest.fixture
def mounting(tmp_path):
    config = json.loads(CONFIG.read_text())
    config['mount']['xyz'] = [0, 0, .092254550594]
    rails = []
    for name, y in [('rail_right', -.076), ('rail_left', .076)]:
        mesh = trimesh.creation.box([.14, .02, .018])
        mesh.apply_translation([-.005, y, .077254550594])
        path = tmp_path / (name + '.stl')
        mesh.export(path)
        rails.append(dict(name=name, mesh_path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    config['rail_mount'] = dict(rails=rails, plate=dict(size_m=[.16, .18, .006], density_kg_m3=2700,
                                                    com_base_m=[-.005, 0, .089254550594]))
    return config


def combined_expected(original, ratio, plate):
    m, c, tensor = inertial_properties(original)
    size = np.array(plate['size_m'])
    pm = float(np.prod(size) * plate['density_kg_m3'])
    pc = np.array(plate['com_base_m'])
    pi = pm / 12 * np.diag([size[1]**2 + size[2]**2, size[0]**2 + size[2]**2, size[0]**2 + size[1]**2])
    m *= ratio
    centre = (m*c + pm*pc)/(m+pm)
    def shift(mass, d):
        return mass * (np.eye(3)*np.dot(d,d)-np.outer(d,d))
    return m+pm, centre, tensor*ratio + shift(m,c-centre) + pi + shift(pm,pc-centre)


@pytest.mark.parametrize('parameters', [{}, {'as2_mass_ratio': 20/17.64}, {'base_mass_ratio': .8},
                                          {'as2_mass_ratio': 1.2, 'base_mass_ratio': .7, 'leg_scale': .95,
                                           'piper_mass_ratios': {'piper_link1': 1.1},
                                           'payload': {'mass_kg': 2, 'model': 'point'}}])
def test_fixed_plate_is_not_scaled_with_original_base(mounting, parameters):
    original = assemble()
    mounted = assemble(config=mounting)
    before = ET.tostring(mounted)
    variant = transform_variant(mounted, parameters, mounting)
    assert ET.tostring(mounted) == before
    if not parameters:
        assert ET.tostring(variant) == before
    ratio = parameters.get('as2_mass_ratio', 1)*parameters.get('base_mass_ratio', 1)
    expected = combined_expected(original.find("link[@name='base_link']"), ratio, mounting['rail_mount']['plate'])
    actual = inertial_properties(variant.find("link[@name='base_link']"))
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=1e-13, rtol=0)
    unmounted = transform_variant(original, parameters, json.loads(CONFIG.read_text()))
    for link in variant.findall('link'):
        if link.get('name') == 'base_link' or link.find('inertial') is None:
            continue
        other = unmounted.find(f"link[@name='{link.get('name')}']")
        for a, e in zip(inertial_properties(link), inertial_properties(other)):
            np.testing.assert_allclose(a, e, atol=1e-13, rtol=0)
    assert len(variant.findall('.//collision')) == 45


def test_static_mujoco_mount_inertia_topology_and_fk(mounting, tmp_path):
    original_bytes = CONFIG.read_bytes()
    root = assemble(config=mounting)
    config_path = tmp_path/'assembly.json'
    config_path.write_text(json.dumps(mounting))
    result = build(tmp_path/'mujoco', config_path=config_path, source_root=root, static_only=True)
    assert result['physics_steps'] == 0
    assert result['nu'] == 18
    np.testing.assert_allclose(result['total_mass_kg'], 22.347+.46656, atol=1e-12)
    model = mujoco.MjModel.from_xml_path(str(tmp_path/'mujoco/robot.xml'))
    assert np.count_nonzero(model.body_mass) == 28
    assert model.ngeom == 46  # 45 robot shapes plus original ground
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    np.testing.assert_allclose(data.body('piper_base_link').xpos-data.body('base_link').xpos,
                               mounting['mount']['xyz'], atol=1e-6, rtol=0)  # mj_saveLastXML pose precision
    base = root.find("link[@name='base_link']")
    for rail in mounting['rail_mount']['rails']:
        collision = base.find(f"collision[@name='{rail['name']}']")
        assert collision.find('origin').get('xyz') == '0 0 0'
        assert collision.find('origin').get('rpy') == '0 0 0'
    assert CONFIG.read_bytes() == original_bytes


def test_changed_rail_bytes_are_rejected(mounting):
    mounting = copy.deepcopy(mounting)
    mounting['rail_mount']['rails'][0]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='Changed registered rail mesh'):
        assemble(config=mounting)
