"""CPU input substitution; extracted statements avoid importing Isaac/starting Kit.

These checks execute the production XML/path blocks, not physics or learning.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from pawcerto.artifacts import file_identity
from pawcerto.robots.urdf import read_joint_limits
from pawcerto.methods.umi_on_legs.robot_binding import as2_config, robot_binding, joint_order

ROOT = Path(__file__).resolve().parents[1]
UMI_URDF = ROOT / 'reference/isaac/go2_arx5/go2_arx5_merged.urdf'


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<production input statements>', 'exec'), namespace)


def statement_block(nodes, first, last):
    start = next(i for i, node in enumerate(nodes) if ast.unparse(node).startswith(first))
    end = next(i for i, node in enumerate(nodes[start:], start) if ast.unparse(node).startswith(last))
    return nodes[start:end + 1]


@pytest.mark.parametrize('asset,names', [
    ('reference/isaac/go2_arx5/go2_arx5_merged.urdf', 'configs/umi_go2_arx5_joint_names.json'),
    ('reference/isaac/go1_arx5/merged.urdf', 'configs/roboduet_go1_arx5_joint_names.json'),
])
def test_real_sources_preserve_original_float_fields_and_order(asset, names):
    path = ROOT / asset
    joint_names = json.loads((ROOT / names).read_text())
    old = {joint.get('name'): joint.find('limit') for joint in ET.parse(path).findall('joint')}
    # Reversed order distinguishes caller order from XML order.
    joint_names.reverse()
    result = read_joint_limits(path, joint_names)
    assert list(result) == joint_names
    for name in joint_names:
        assert result[name] == {key: float(old[name].get(key)) for key in ('lower', 'upper', 'effort', 'velocity')}


def runtime_init_nodes():
    tree = ast.parse((ROOT / 'pawcerto/isaac/runtime.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Go2Arx5Isaac')
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__').body


def umi_input_blocks(path=None, config=None, names=None):
    init = runtime_init_nodes()
    config = config or {}
    namespace = dict(Path=Path, ET=ET, math=math, read_joint_limits=read_joint_limits,
                     robot_binding=robot_binding, joint_order=joint_order,
                     self=SimpleNamespace(), urdf_path=path, config=config or {},
                     joint_names=joint_order(config) if names is None else names)
    execute(statement_block(init, 'self.binding =', 'if list(joint_names)'), namespace)
    execute(statement_block(init, 'self.urdf_path =', 'for name in joint_names:'), namespace)
    execute(statement_block(init, 'root_body =', 'self.contact_names ='), namespace)
    return namespace


def write_urdf(path, velocity=7, parent='base'):
    path.write_text(f'''<robot name="input"><link name="base"/><link name="alternate_tool"><collision/></link>
      <joint name="motor" type="revolute"><parent link="{parent}"/><child link="alternate_tool"/>
      <limit lower="-1" upper="2" effort="3" velocity="{velocity}"/></joint></robot>''')
    return path


def write_runtime_urdf(path, velocity=7, parent=None, config=None):
    """Change a real bound asset, retaining its actual joint order and topology."""
    binding = robot_binding(config or {})
    asset = ET.parse(binding['urdf_path'])
    joint = asset.find(f"joint[@name='{binding['joint_names'][0]}']")
    joint.find('limit').set('velocity', str(velocity))
    if parent is not None:
        joint.find('parent').set('link', parent)
    link = ET.SubElement(asset.getroot(), 'link', name='alternate_tool')
    ET.SubElement(link, 'collision')
    mount = ET.SubElement(asset.getroot(), 'joint', name='alternate_mount', type='fixed')
    ET.SubElement(mount, 'parent', link=binding['root_body'])
    ET.SubElement(mount, 'child', link='alternate_tool')
    asset.write(path)
    return path


def test_umi_alternate_urdf_drives_limits_and_contact_topology(tmp_path):
    alternate = write_runtime_urdf(tmp_path / 'alternate.urdf')
    for path, config in [(alternate, {}), (None, {'pawcerto_asset': {'urdf_path': str(alternate)}}),
                         (alternate, {'pawcerto_asset': {'urdf_path': 'not-selected.urdf'}})]:
        result = umi_input_blocks(path, config)
        assert result['self'].urdf_path == alternate.resolve()
        assert result['velocity_limits'] == {name: values['velocity'] for name, values in
                                            read_joint_limits(alternate, joint_order(config)).items()}
        assert result['velocity_limits'][joint_order(config)[0]] == 7.0
        assert 'alternate_tool' in result['self'].contact_names
        assert result['paths']['alternate_tool'] == '{ENV_REGEX_NS}/Robot/Geometry/base/alternate_tool'
    default = umi_input_blocks()
    assert default['self'].urdf_path == UMI_URDF.resolve()
    assert default['velocity_limits'] == {name: values['velocity'] for name, values in
                                         read_joint_limits(UMI_URDF, joint_order({})).items()}


@pytest.mark.parametrize('velocity', [0, -1])
def test_umi_rejects_nonpositive_source_velocity_even_with_override(tmp_path, velocity):
    with pytest.raises(ValueError, match='Invalid original URDF velocity limit'):
        umi_input_blocks(write_runtime_urdf(tmp_path / 'invalid.urdf', velocity),
                         {'joint_velocity_limit_override_rad_s': 1000})


def test_umi_rejects_unreachable_topology(tmp_path):
    with pytest.raises(ValueError, match='not reachable from UMI base link'):
        umi_input_blocks(write_runtime_urdf(tmp_path / 'disconnected.urdf', parent='absent'))



def test_reader_errors_preserve_input_causes(tmp_path):
    path = write_urdf(tmp_path / 'robot.urdf')
    with pytest.raises(FileNotFoundError):
        read_joint_limits(tmp_path / 'missing.urdf', ['motor'])
    with pytest.raises(ValueError, match='no joint'):
        read_joint_limits(path, ['absent'])
    with pytest.raises(ValueError, match='duplicate'):
        read_joint_limits(path, ['motor', 'motor'])
    for broken, message in [('velocity="7"', 'Invalid URDF limits'),
                            ('<limit lower="-1" upper="2" effort="3" velocity="7"/>', 'has no limits')]:
        write_urdf(path)
        path.write_text(path.read_text().replace(broken, ''))
        with pytest.raises(ValueError, match=message):
            read_joint_limits(path, ['motor'])
    write_urdf(path, velocity='nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        read_joint_limits(path, ['motor'])


@pytest.mark.parametrize('script,record', [('train_umi.py', 'source_asset'),
                                         ('eval_umi.py', 'checkpoint_asset'),
                                         ('run_umi_isaac.py', 'saved_asset')])
@pytest.mark.parametrize('selection', ['explicit', 'recorded', 'default'])
@pytest.mark.parametrize('robot', ['go2_arx5', 'as2_piper'])
def test_cli_path_precedence_recording_and_runtime_argument(tmp_path, script, record, selection, robot):
    tree = ast.parse((ROOT / 'scripts' / script).read_text())
    # Execute the real argparse declaration and the contiguous input resolution
    # block, then the real runtime call with only its external consumer captured.
    declaration = next(node for node in ast.walk(tree) if isinstance(node, ast.Expr)
                       and ast.unparse(node).startswith("parser.add_argument('--urdf-path'"))
    parser = argparse.ArgumentParser()
    execute([declaration], dict(parser=parser, Path=Path))
    source = json.loads((ROOT / 'reference/checkpoints/tossing/ours/config.json').read_text())
    config = as2_config(source) if robot == 'as2_piper' else source
    explicit = write_runtime_urdf(tmp_path / 'explicit.urdf', velocity=8, config=config)
    recorded = write_runtime_urdf(tmp_path / 'recorded.urdf', velocity=9, config=config)
    args = parser.parse_args(['--urdf-path', str(explicit)] if selection == 'explicit' else [])
    usd = tmp_path / 'robot.usd'
    usd.write_text('input identity only; never opened by a simulator')
    pointer = tmp_path / 'usd_path.txt'
    pointer.write_text(str(usd))
    joint_file = tmp_path / 'joints.json'
    joint_file.write_text(json.dumps(joint_order(config)))
    args.__dict__.update(usd_path=pointer, joint_names=joint_file, num_envs=1, device='cpu',
                         domain_randomization=False, ground_contact_diagnostics=False, force_signal='normal-contact')
    captured = {}
    def capture(*positional, **keywords):
        captured.update(keywords)
        captured['joint_names'] = positional[1]
    namespace = dict(Path=Path, args=args, config=config, policy=SimpleNamespace(config=config),
                     DEFAULT_USD=usd, DEFAULT_URDF=UMI_URDF, file_identity=file_identity,
                     json=json, Go2Arx5Isaac=capture, robot_binding=robot_binding, joint_order=joint_order)
    namespace[record] = {'urdf_path': str(recorded)} if selection != 'default' else None
    body = next(node.body for node in ast.walk(tree) if isinstance(node, ast.Try)
                and any(ast.unparse(child).startswith('usd_path =') for child in node.body))
    execute([next(node for node in body if ast.unparse(node).startswith('binding ='))], namespace)
    last = 'asset_identity = dict(' if script == 'run_umi_isaac.py' else "config['pawcerto_asset'] ="
    execute(statement_block(body, 'usd_path =', last), namespace)
    runtime_call = next(node for node in body if isinstance(node, ast.Assign)
                        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'Go2Arx5Isaac')
    execute([runtime_call], namespace)
    selected = explicit if selection == 'explicit' else recorded if selection == 'recorded' else Path(robot_binding(config)['urdf_path'])
    identity = namespace['asset_identity'] if script == 'run_umi_isaac.py' else config['pawcerto_asset']
    assert captured['joint_names'] == joint_order(config)
    assert captured['urdf_path'] == selected
    assert captured['usd_path'] == usd.resolve()
    assert identity['urdf_path'] == str(selected.resolve())
    assert identity['urdf_sha256'] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert identity['usd_path'] == str(usd.resolve())
    joint_file.write_text(json.dumps(list(reversed(joint_order(config)))))
    with pytest.raises(ValueError, match='Joint order differs'):
        execute([runtime_call], namespace)


def test_as2_runtime_binding_drives_root_topology_and_limits(tmp_path):
    source = json.loads((ROOT / 'reference/checkpoints/tossing/ours/config.json').read_text())
    config = as2_config(source)
    initial = dict(config=config)
    execute(statement_block(runtime_init_nodes(), 'initial =', 'initial_rotation ='), initial)
    assert initial['initial_rotation'] == tuple(config['env']['cfg']['init_state']['rot']) == (0., 0., 0., 1.)
    alternate = write_runtime_urdf(tmp_path / 'as2.urdf', config=config)
    result = umi_input_blocks(alternate, config)
    assert result['self'].binding == robot_binding(config)
    assert result['root_body'] == 'base_link'
    assert result['paths']['alternate_tool'] == '{ENV_REGEX_NS}/Robot/Geometry/base_link/alternate_tool'
    assert 'piper_gripper_base' in result['paths']
    assert result['velocity_limits'] == {name: values['velocity'] for name, values in
                                        read_joint_limits(alternate, joint_order(config)).items()}
    default = umi_input_blocks(config=config)
    assert default['self'].urdf_path == Path(robot_binding(config)['urdf_path']).resolve()
    with pytest.raises(ValueError, match='Runtime joint order differs'):
        umi_input_blocks(alternate, config, names=joint_order({}))
