"""Real CPU checkpoint/export/reload checks; synthetic weights are not learning."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from pawcerto.methods.deepwbc.config import default_config
from pawcerto.methods.deepwbc.export import load_checkpoint, export_policy, compare_export
from pawcerto.methods.deepwbc.policy import build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    config = default_config()
    model = build_model(config).eval()
    path = tmp_path / 'checkpoint.pt'
    torch.save({'model_state_dict': model.state_dict(), 'config': config, 'iter': 42}, path)
    return load_checkpoint(path)


def test_fused_split_reload_and_original_actor(checkpoint, tmp_path):
    output = tmp_path / 'export'
    report = export_policy(checkpoint, output)
    assert report['actions']['policy']['passed']
    assert 'synthetic' in report['input_source']
    assert checkpoint.iteration == 42
    obs = torch.randn(4, 860)
    compare_export(checkpoint, output, obs)
    # Execute the actual pinned save_jit definitions only (no simulator imports).
    source = ROOT / 'third_party/deepwbc-reference/legged_gym/legged_gym/scripts/save_jit.py'
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
    cfg = checkpoint.config['policy']
    original = namespace['Actor'](76, cfg['actor_hidden_dims'], nn.ELU(),
        cfg['leg_control_head_hidden_dims'], cfg['arm_control_head_hidden_dims'],
        cfg['num_leg_actions'], cfg['num_arm_actions'], cfg['adaptive_arm_gains'],
        cfg['adaptive_arm_gains_scale'], 24, 10, 76, cfg['priv_encoder_dims'])
    original.load_state_dict(checkpoint.model.actor.state_dict(), strict=True)
    with torch.no_grad():
        expected = original(torch.cat((obs[:, :76], original.history_encoder(obs[:, 100:])), 1))
        actual = torch.jit.load(str(output / 'policy.pt'))(obs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Genuine standalone interpreter importing only torch, not PawCerto.
    subprocess.run([sys.executable, '-c',
        'import torch,sys; torch.set_num_threads(1); p=torch.jit.load(sys.argv[1],map_location="cpu"); assert p(torch.zeros(3,860)).shape == (3,18)',
        str(output / 'policy.pt')], cwd=tmp_path, check=True, capture_output=True)
    contract = json.loads((output / 'contract.json').read_text())
    assert contract['policy.pt']['input'] == ['batch', 860]
    assert contract['actor.pt']['input'] == ['batch', 96]
    assert contract['checkpoint_sha256'] == checkpoint.checkpoint_sha256


def test_history_only_and_shape_contract(checkpoint, tmp_path):
    output = tmp_path / 'export'
    export_policy(checkpoint, output)
    policy = torch.jit.load(str(output / 'policy.pt'))
    obs = torch.randn(2, 860)
    before = policy(obs)
    obs[:, 76:100] += 100
    torch.testing.assert_close(policy(obs), before, atol=0, rtol=0)
    obs[:, 100:] += 10
    assert not torch.equal(policy(obs), before)
    with pytest.raises((RuntimeError, torch.jit.Error), match='full_observation_width'):
        policy(torch.zeros(2, 861))
    with pytest.raises(ValueError, match='nonfinite'):
        compare_export(checkpoint, output, torch.full((1, 860), float('nan')))
    with pytest.raises(FileExistsError):
        export_policy(checkpoint, output)


def test_checkpoint_no_config_and_strict_state(checkpoint, tmp_path):
    raw = tmp_path / 'author.pt'
    torch.save({'model_state_dict': checkpoint.model.state_dict()}, raw)
    with pytest.raises(ValueError, match='lacks config'):
        load_checkpoint(raw)
    restored = load_checkpoint(raw, config=checkpoint.config)
    assert restored.iteration is None
    bad = default_config()
    bad['seed'] = 100
    with pytest.raises(ValueError, match='differs'):
        load_checkpoint(checkpoint.checkpoint_path, config=bad)
    state = checkpoint.model.state_dict()
    del state['actor.actor_backbone.0.weight']
    torch.save({'model_state_dict': state, 'config': checkpoint.config}, raw)
    with pytest.raises(RuntimeError, match='Missing key'):
        load_checkpoint(raw)


def test_cli(checkpoint, tmp_path):
    output = tmp_path / 'cli'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/export_deepwbc.py'),
        '--checkpoint', str(checkpoint.checkpoint_path), '--output', str(output)],
        check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report['samples'] == 7
    assert 'not physical evaluation' in report['evidence']
