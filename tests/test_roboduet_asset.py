"""Asset checks independent of Isaac startup and policy implementation."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from pawcerto.mujoco.roboduet_asset import ROOT, URDF, JOINT_NAMES, DEFAULT_POS, build_model


def test_original_asset_mass_mount_and_dofs(tmp_path):
    output = build_model(tmp_path / 'robot.xml')
    model = mujoco.MjModel.from_xml_path(str(output))
    source = ET.parse(URDF)
    expected_mass = sum(float(e.get('value')) for e in source.findall('link/inertial/mass')) + .1
    assert abs(model.body_mass.sum() - expected_mass) < 2e-5
    assert model.nq == 27 and model.nv == 26
    assert {model.joint(i).name for i in range(model.njnt)} == set(JOINT_NAMES) | {'root'}
    assert json.loads((ROOT / 'configs/roboduet_go1_arx5_joint_names.json').read_text()) == JOINT_NAMES
    data = mujoco.MjData(model)
    for name, angle in zip(JOINT_NAMES,DEFAULT_POS):
        data.qpos[model.joint(name).qposadr] = angle
    mujoco.mj_forward(model,data)
    np.testing.assert_allclose(data.body('base_link').xpos-data.body('base').xpos,[0,0,.057],atol=1e-8)
    np.testing.assert_allclose(data.body('base_link').xmat,data.body('base').xmat,atol=1e-8)
    assert all(model.body(leg+'_foot').id > 0 for leg in ('FL','FR','RL','RR'))
