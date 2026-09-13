"""Check saved family identity and nominal-evaluation selection without a simulator."""
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from pawcerto.methods.umi_on_legs.pretraining import (
    configure_asset_family, require_resume_family, resolve_training_asset, training_family,
)


def prepared_config(tmp_path):
    variants = []
    for name in ('nominal', 'payload'):
        folder = tmp_path / name
        folder.mkdir()
        urdf = folder / 'merged.urdf'
        usd = folder / 'robot.usd'
        # These test bytes exercise configuration identity, not USD validity.
        urdf.write_text(name + ' urdf fixture')
        usd.write_text(name + ' usd fixture')
        (folder / 'usd_path.txt').write_text(str(usd))
        variants.append(dict(name=name, merged_urdf_path=str(urdf)))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(variants=variants)))
    source = {'pawcerto_robot': {'name': 'as2_piper'}, 'runner': {'max_iterations': 4000}}
    before = deepcopy(source)
    config = configure_asset_family(source, manifest)
    assert source == before
    return config


def test_family_is_training_only_and_detects_changed_asset(tmp_path):
    config = prepared_config(tmp_path)
    assert config['pawcerto_asset']['usd_path'] == str(tmp_path/'nominal/robot.usd')
    assert len(training_family(config, True)['variants']) == 2
    assert training_family(config, False) is None
    (tmp_path/'payload/robot.usd').write_text('changed asset')
    with pytest.raises(ValueError, match='Changed usd'):
        training_family(config, True)
    # Frozen nominal evaluation does not need the old training-only payload file.
    assert training_family(config, False) is None


def test_resume_retains_family_but_initialization_can_be_a_separate_choice(tmp_path):
    config = prepared_config(tmp_path)
    require_resume_family(config, deepcopy(config))
    with pytest.raises(ValueError, match='Resume must retain'):
        require_resume_family(config, {'pawcerto_robot': config['pawcerto_robot']})


def test_weight_initialization_retains_destination_nominal(tmp_path):
    config = prepared_config(tmp_path)
    source_asset = dict(usd_path=str(tmp_path/'payload/robot.usd'),
                        urdf_path=str(tmp_path/'payload/merged.urdf'))
    assert resolve_training_asset(config, source_asset) == config['pawcerto_asset']
    with pytest.raises(ValueError, match='declared nominal'):
        resolve_training_asset(config, source_asset, usd_override=source_asset['usd_path'])
    # Preserve the existing source-asset default for ordinary single-asset runs.
    single = deepcopy(config)
    single.pop('pawcerto_asset_family')
    assert resolve_training_asset(single, source_asset)['usd_path'] == source_asset['usd_path']


def test_custom_assembly_rebinds_actual_shape_count_and_critic(tmp_path):
    from pawcerto.artifacts import file_identity
    from pawcerto.methods.umi_on_legs.robot_binding import as2_config
    from pawcerto.mujoco.as2_piper_asset import CONFIG
    root = Path(__file__).resolve().parents[1]
    source = as2_config(json.loads((root/'reference/checkpoints/tossing/ours/config.json').read_text()))
    before = deepcopy(source)
    nominal_config = tmp_path/'assembly.json'
    nominal_config.write_bytes(CONFIG.read_bytes())
    # A configuration fixture with one additional fixed-body collision shape.
    # The USD bytes test identity routing only; no converter or simulator is claimed.
    tree = ET.parse(root/'reference/as2_piper/robot.urdf')
    base = tree.getroot().find("link[@name='base_link']")
    collision = ET.SubElement(base, 'collision', name='test_mount_plate')
    geometry = ET.SubElement(collision, 'geometry')
    ET.SubElement(geometry, 'box', size='.16 .18 .006')
    urdf = tmp_path/'merged.urdf'
    tree.write(urdf)
    usd = tmp_path/'robot.usd'
    usd.write_text('identity fixture only')
    (tmp_path/'usd_path.txt').write_text(str(usd))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(
        source=dict(nominal_config_path=str(nominal_config), nominal_config_sha256=file_identity(nominal_config)['sha256']),
        variants=[dict(name='nominal', merged_urdf_path=str(urdf))])))
    actual = configure_asset_family(source, manifest)
    assert source == before
    assert actual['pawcerto_robot']['collision_shape_count'] == 43
    assert actual['env']['privileged_setup_obs']['rigid_shape_friction']['dim'] == 43
    assert actual['runner']['alg']['actor_critic']['num_critic_obs'] == 270
    assert actual['runner']['alg']['actor_critic']['critic']['_args_'][0]['in_features'] == 270
    assert actual['pawcerto_robot']['source_urdf'] == file_identity(urdf)
    assert actual['pawcerto_robot']['nominal_input'] == file_identity(nominal_config)
    assert actual['pawcerto_asset']['urdf_path'] == str(urdf)
