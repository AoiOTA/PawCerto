"""CPU source-preparation checks; optional actual USD validation without Kit."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pawcerto.mujoco.as2_piper_asset import assemble, urdf_fk
from pawcerto.robots.as2_piper import prepare_urdf


def test_preparation_preserves_physics_and_frames(tmp_path):
    tree, manifest = prepare_urdf(tmp_path)
    source = assemble()
    assert len(manifest['body_names']) == 28
    assert manifest['mass_kg'] == pytest.approx(22.347)
    assert len(manifest['controlled_joint_order']) == 18
    for link in tree.findall('link'):
        original = source.find(f"link[@name='{link.get('name')}']")
        assert ET.tostring(link.find('inertial')) == ET.tostring(original.find('inertial'))
        assert [ET.tostring(c) for c in link.findall('collision')] == [ET.tostring(c) for c in original.findall('collision')]
    for joint in tree.findall('joint'):
        assert ET.tostring(joint) == ET.tostring(source.find(f"joint[@name='{joint.get('name')}']"))
    assert tree.find("link[@name='piper_gripper_link']") is None
    assert set(['FL_foot','FR_foot','RL_foot','RR_foot','piper_gripper_base']).issubset(manifest['body_names'])
    reconstructed = copy.deepcopy(tree.getroot())
    reconstructed.append(copy.deepcopy(source.find("link[@name='world']")))
    reconstructed.append(copy.deepcopy(source.find("joint[@name='floating_base_joint']")))
    for q in [dict.fromkeys(manifest['controlled_joint_order'], 0.), dict.fromkeys(manifest['controlled_joint_order'], .17)]:
        before = urdf_fk(source, q, [0,0,.55], [1,0,0,0])
        after = urdf_fk(reconstructed, q, [0,0,.55], [1,0,0,0])
        for name in manifest['body_names']:
            np.testing.assert_allclose(before[name], after[name], atol=1e-12)
        tcp = np.r_[manifest['tcp']['xyz'], 1.]
        np.testing.assert_allclose(before['piper_gripper_base'] @ tcp, after['piper_gripper_base'] @ tcp, atol=1e-12)


def test_generated_usd_source_contract(tmp_path):
    pointer = Path(__file__).resolve().parents[1] / 'reference/isaac/as2_piper/usd_path.txt'
    if not pointer.is_file():
        pytest.skip('Actual USD conversion has not run; CPU preparation is not conversion evidence')
    pytest.importorskip('pxr', reason='Use Lab Python --validate-only for standalone USD validation')
    from pxr import Usd
    usd = Path(pointer.read_text().strip())
    from scripts.convert_as2_piper_usd import restore_and_validate
    tree, manifest = prepare_urdf(tmp_path)
    stage = Usd.Stage.Open(str(usd))
    # Session layer keeps this validation read-only for the delivered artifact.
    stage.SetEditTarget(stage.GetSessionLayer())
    report = restore_and_validate(stage, tree, manifest)
    assert report['instance_proxy_collision_count'] > 0
    assert max(r['before_tensor_error'] for r in report['inertias']) < 1e-6
