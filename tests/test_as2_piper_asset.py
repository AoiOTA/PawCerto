"""CPU integration checks require assets fetched by scripts/fetch_as2_piper.py."""
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
from pawcerto.mujoco.as2_piper_asset import CONFIG, SOURCES, assemble, build, urdf_fk, validate

pytestmark = pytest.mark.skipif(not (SOURCES / 'provenance.json').exists(), reason='Fetch pinned AS2/Piper assets first')


def test_source_composition_and_compiled_fk(tmp_path):
    report = build(tmp_path)
    assert report['component_masses_kg'] == pytest.approx({
        'as2_urdf_selected': 17.64, 'piper_h_bare_urdf_selected': 4.167,
        'as2_vendor_mjcf_not_selected': 20.7, 'stock_gripper_with_flange_selected': .54})
    assert report['total_mass_kg'] == pytest.approx(22.347)
    assert report['max_fk_position_error_m'] < 1e-5
    assert report['default_contacts'] == []
    assert report['cpu_probe']['direct_torque_mapping_and_effort_clipping'] == 'passed'
    assert report['cpu_probe']['unactuated_com_velocity_m_s'] == pytest.approx([0, 0, -.4905])
    model = mujoco.MjModel.from_xml_path(str(tmp_path / 'robot.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    before = data.qpos.copy()
    # Exercise actual CPU integration; not a balance or policy test.
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert not np.array_equal(before, data.qpos)
    assert all(w.number == 0 for w in data.warning)


def test_independent_fk_detects_wrong_axis(tmp_path):
    build(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / 'robot.xml'))
    root = ET.parse(tmp_path / 'robot.urdf').getroot()
    root.find("joint[@name='piper_joint2']/axis").set('xyz', '1 0 0')
    with pytest.raises((AssertionError, ValueError)):
        validate(model, root, json.loads(CONFIG.read_text()))


def test_changed_source_refused(tmp_path):
    shutil.copytree(SOURCES, tmp_path / 'sources')
    path = tmp_path / 'sources' / json.loads(CONFIG.read_text())['piper_urdf']
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Source content changed'):
        assemble(tmp_path / 'sources')


def test_frozen_gripper_matches_original_prismatic_mimic_fk(tmp_path):
    config = json.loads(CONFIG.read_text())
    build(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / 'robot.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    original = ET.parse(SOURCES / config['piper_urdf']).getroot()
    gripper = ET.parse(SOURCES / config['gripper']['source_xacro']).getroot()
    for element in gripper:
        if element.tag in ('link', 'joint'):
            original.append(element)
    positions = {name.removeprefix('piper_'): value for name, value in zip(
        config['controlled_joint_order'][12:], config['default_joint_positions'][12:])}
    width = config['gripper']['opening_m']
    positions.update(gripper=width, gripper_joint1=width * .5, gripper_joint2=width * -.5)
    original_poses = urdf_fk(original, positions, [0, 0, 0], [1, 0, 0, 0])
    base = np.eye(4)
    base[:3, :3] = data.body('piper_base_link').xmat.reshape(3, 3)
    base[:3, 3] = data.body('piper_base_link').xpos
    for name in ('flange_link', 'gripper_base', 'gripper_link', 'gripper_link1', 'gripper_link2'):
        expected = base @ original_poses[name]
        actual = data.body('piper_' + name)
        np.testing.assert_allclose(actual.xpos, expected[:3, 3], atol=1e-6)
        np.testing.assert_allclose(actual.xmat.reshape(3, 3), expected[:3, :3], atol=1e-6)
    assert model.nu == 18


def test_bare_arm_control_retained(tmp_path):
    report = build(tmp_path, config_path=CONFIG.with_name('as2_piper_bare.json'))
    assert report['total_mass_kg'] == pytest.approx(21.807)
    assert 'stock_gripper_with_flange_selected' not in report['component_masses_kg']
