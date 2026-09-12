"""CPU contract and failure accounting; fixtures are not physics evidence."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pawcerto.methods.learning_force_control.config import build_model, network_config
from pawcerto.methods.learning_force_control.training.config import default_config, resolve_config
from pawcerto.methods.learning_force_control.export import load_training_checkpoint, export_training_checkpoint
from pawcerto.methods.learning_force_control.evaluation import evaluate_fixed_policy, position_target_world, fixed_task_config


@pytest.fixture(autouse=True)
def restore_original_torch_typo():
    normal = torch.distributions.Normal
    previous = normal.__dict__.get('set_default_validate_args')
    yield
    if previous is None:
        if 'set_default_validate_args' in normal.__dict__:
            delattr(normal, 'set_default_validate_args')
    else:
        normal.set_default_validate_args = previous


def test_full_checkpoint_original_jit_pair(tmp_path):
    torch.set_num_threads(1)
    config = default_config()
    model = build_model(network_config(config))
    checkpoint = tmp_path / 'model.pt'
    torch.save({'format': 'pawcerto.learning_force_control.training.v1',
        'config': config, 'model_state_dict': model.state_dict(),
        'next_iteration': 21, 'common_step_counter': 1009, 'global_steps': 16128}, checkpoint)
    loaded = load_training_checkpoint(checkpoint)
    result = export_training_checkpoint(loaded, tmp_path / 'export')
    assert result['actions']['max_abs_error'] == 0
    assert result['latents']['max_abs_error'] == 0
    contract = json.loads((tmp_path / 'export/contract.json').read_text())
    assert contract['next_iteration'] == 21
    assert contract['network']['num_actions'] == 19
    assert contract['network']['num_obs_history'] == 870


def test_original_position_target_fixed_height_and_negative_pitch():
    commands = torch.zeros(2, 23)
    commands[:, 15] = .6
    commands[1, 16] = torch.pi / 2
    base_pos = torch.tensor([[10., 20., 9.], [10., 20., -9.]])
    yaw90 = torch.tensor([[0., 0., 2**-.5, 2**-.5]]).expand(2, -1)
    target = position_target_world(commands, base_pos, yaw90)
    torch.testing.assert_close(target, torch.tensor([[10., 20.8, .7585], [10., 20.2, .1585]]),
                               atol=2e-6, rtol=1e-6)
    with pytest.raises(ValueError, match='ground clearance'):
        fixed_task_config(default_config(), {'commands': [.3, 0, 0],
            'ee_goal_lpy': [1., np.pi/2, 0.], 'ee_timing_s': 2., 'hybrid_mode': 'binary'})


def test_fixed_task_initializes_and_samples_actual_source_curriculum():
    from pawcerto.methods.learning_force_control.training.lifecycle import LFCLifecycle
    config = fixed_task_config(default_config(), {'commands': [.3, 0, 0],
        'ee_goal_lpy': [.6, 0, 0], 'ee_timing_s': 2., 'hybrid_mode': 'binary'})
    env = SimpleNamespace(cfg=resolve_config(config), num_envs=16, device='cpu')
    LFCLifecycle._init_command_distribution(env, torch.arange(16))
    for curriculum in env.curricula:
        samples, _ = curriculum.sample(16)
        assert np.isfinite(samples).all()
        np.testing.assert_allclose(samples[:, :3], np.tile([.3, 0., 0.], (16, 1)), rtol=0, atol=0)
        np.testing.assert_allclose(samples[:, 15:19], np.tile([.6, 0., 0., 2.], (16, 1)), rtol=0, atol=0)


def policies():
    model = SimpleNamespace(act_student=lambda history, **kw: torch.zeros(2, 19),
                            get_student_latent=lambda history: torch.zeros(2, 16))
    class Export:
        def __call__(self, history): return torch.zeros(2, 19)
        def adaptation(self, history): return torch.zeros(2, 16)
    return model, Export()


def test_finite_transition_mode_denominators_and_failure_consumption(tmp_path):
    obs = {'obs_history': torch.zeros(2, 870), 'privileged_obs': torch.zeros(2, 16)}
    env = SimpleNamespace(num_envs=2, transition_observer=None, reset=lambda: obs)
    def step(action):
        env.transition_observer(env)
        return obs, None, None, None
    env.step = step
    snapshots = []
    for errors, modes in [([1., 3.], [0, 1]), ([5., float('nan')], [0, 1])]:
        snapshots.append({'ee_position_error_m': np.array(errors),
            'base_up_dot': np.ones(2), 'velocity_error_xy': np.zeros((2, 2)),
            'force_error_xyz_N': np.full((2, 3), 2.),
            'force_or_position_control': np.array(modes), 'freed_envs': np.zeros(2, dtype=bool),
            'done': np.zeros(2, dtype=bool), 'reason_mask': np.zeros((2, 7), dtype=bool)})
    model, exported = policies()
    with patch('pawcerto.methods.learning_force_control.evaluation.transition_snapshot', side_effect=snapshots):
        with pytest.raises(FloatingPointError):
            evaluate_fixed_policy(env, model, steps=2, output=tmp_path / 'eval',
                                  metadata={}, exported_policy=exported)
    report = json.loads((tmp_path / 'eval/summary.json').read_text())
    assert report['finite_environment_transitions'] == 3
    assert report['position_transitions'] == 2
    assert report['force_transitions'] == 1
    assert report['position_mode_ee_error_mean_m'] == 3.
    assert report['force_mode_force_xyz_error_rms_N'] == 2.
    assert report['exported_policy_observed_steps'] == 2
    def fail_reset(): raise RuntimeError('reset failed')
    env.reset = fail_reset
    with pytest.raises(RuntimeError, match='reset failed'):
        evaluate_fixed_policy(env, model, steps=2, output=tmp_path / 'reset-failure',
                              metadata={}, exported_policy=exported)
    report = json.loads((tmp_path / 'reset-failure/summary.json').read_text())
    assert report['exported_policy_consumed'] is False
    assert report['exported_action_max_abs_error'] is None
