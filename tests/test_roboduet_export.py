"""Five real JIT modules must retain the selected upstream arm output semantics."""
import json

import pytest
import torch

from pawcerto.methods.roboduet import build_models, default_config
from pawcerto.methods.roboduet.export import ExportedRoboDuet, MODULE_FILES, export_checkpoint


def test_five_modules_preserve_raw_and_training_mean(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    config = default_config()
    for side in ('DogAC_Args', 'ArmAC_Args'):
        config[side].update(actor_hidden_dims=[24, 16, 8], critic_hidden_dims=[24, 16, 8],
                            adaptation_module_branch_hidden_dims=[16])
    dog, arm = build_models(config)
    # Make the guidance distinction material rather than passing only near zero.
    with torch.no_grad():
        arm.actor_body[-1].bias[-2:] = torch.tensor([2., -3.])
    checkpoint = tmp_path / 'model.pt'
    torch.save({'format': 'pawcerto.roboduet.training.v1', 'config': config,
                'dog_model_state_dict': dog.state_dict(), 'arm_model_state_dict': arm.state_dict(),
                'stage': 1, 'next_iteration': 1}, checkpoint)
    inputs = {'obs': torch.randn(2, config['Cfg']['arm']['arm_num_observations']),
              'obs_history': torch.randn(2, config['Cfg']['arm']['arm_num_obs_history'])}
    modes = {}
    for mode in ('official_play', 'training_mean'):
        output = tmp_path / mode
        result = export_checkpoint(checkpoint, output, [f'joint_{i}' for i in range(20)], mode=mode)
        assert all((output / filename).exists() for filename in MODULE_FILES.values())
        assert result['next_rollout_stage'] == 1
        assert result['arm_update_count'] == 0
        assert 'has not received' in result['stage_note']
        restored = ExportedRoboDuet(output)
        modes[mode] = restored.arm_action(inputs)
        torch.testing.assert_close(modes[mode], arm.act_inference(inputs, mode=mode))
        assert json.loads((output / 'export.json').read_text())['arm_output_mode'] == mode
    latest_inputs = dict(inputs, obs=inputs['obs_history'][..., -config['Cfg']['arm']['arm_num_observations']:])
    torch.testing.assert_close(modes['training_mean'][..., -2:],
                               arm.act_inference(latest_inputs, mode='official_play')[..., -2:].tanh())
    assert (modes['official_play'][..., -2:].abs() > 1).all()
    assert (modes['training_mean'][..., -2:].abs() < 1).all()
    # Losing the history encoder is a broken package, not a two-actor fallback.
    (tmp_path / 'official_play' / MODULE_FILES['arm_history']).unlink()
    with pytest.raises((ValueError, RuntimeError)):
        ExportedRoboDuet(tmp_path / 'official_play')


def test_real_runner_stage_boundary_does_not_claim_arm_training(tmp_path):
    # Exercise the real runner's save boundary. This fixture only supplies
    # deterministic observations/rewards; it makes no physical learning claim.
    from test_roboduet_training import compact_config, ProtocolEnv
    from pawcerto.methods.roboduet.training.runner import RoboDuetRunner
    torch.set_num_threads(1)
    config = compact_config()
    runner = RoboDuetRunner(ProtocolEnv(config), config)
    runner.run_iteration()
    checkpoint = tmp_path / 'boundary.pt'
    runner.save(checkpoint)
    boundary = export_checkpoint(checkpoint, tmp_path / 'boundary', [f'joint_{i}' for i in range(20)])
    assert boundary['next_rollout_stage'] == 2
    assert boundary['arm_update_count'] == 0
    runner.run_iteration()
    trained = tmp_path / 'first_arm_update.pt'
    runner.save(trained)
    result = export_checkpoint(trained, tmp_path / 'updated', [f'joint_{i}' for i in range(20)])
    assert result['next_rollout_stage'] == 2
    assert result['arm_update_count'] == 1
