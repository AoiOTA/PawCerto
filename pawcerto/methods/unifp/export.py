"""UniFP deterministic student-policy export and CPU consumption.

History and PD control remain in the method's environment. The twelve decoded
values are estimates of training supervision, not external sensor readings.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn

from pawcerto.artifacts import file_identity
from .algorithm import ActorCritic


@dataclass
class LoadedCheckpoint:
    model: nn.Module
    config: dict
    iteration: int | None
    global_steps: int | None
    identity: dict


def load_checkpoint(path, *, config=None, device='cpu'):
    """Load a trusted runner checkpoint, requiring its actual network config."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or 'model_state_dict' not in saved:
        raise ValueError('Expected a runner checkpoint with model_state_dict')
    embedded = saved.get('config')
    if config is not None and embedded is not None and config != embedded:
        raise ValueError('Explicit config differs from checkpoint config')
    config = deepcopy(embedded if embedded is not None else config)
    if config is None:
        raise ValueError('Checkpoint lacks config; supply its complete resolved config')
    env = config['env']['env']
    if (env['num_observations'] != env['frame_stack'] * env['num_single_obs']
            or env['num_privileged_obs'] != env['c_frame_stack'] * env['single_num_privileged_obs']):
        raise ValueError('Observation dimensions disagree with configured history')
    model = ActorCritic(env['num_observations'], env['num_privileged_obs'],
                       env['num_pred_obs'], env['num_single_obs'], env['num_actions'],
                       **config['train']['policy'])
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.to(device).eval().requires_grad_(False)
    return LoadedCheckpoint(model, config, saved.get('iter'), saved.get('global_steps'),
                            file_identity(path))


class _StudentPolicy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.adaptation_encoder_module
        self.actor = model.actor_body
        self.width = model.num_obs
        self.frame_width = model.num_obs_now

    def forward(self, history):
        if history.dim() != 2 or history.size(0) == 0 or history.size(1) != self.width:
            raise RuntimeError('Expected nonempty [batch, actor_history_width]')
        return self.actor(torch.cat((history[:, -self.frame_width:], self.encoder(history)), -1))


class _Estimator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.adaptation_encoder_module
        self.decoder = model.adaptation_decoder_module
        self.width = model.num_obs

    def forward(self, history):
        if history.dim() != 2 or history.size(0) == 0 or history.size(1) != self.width:
            raise RuntimeError('Expected nonempty [batch, actor_history_width]')
        return self.decoder(self.encoder(history))


class ExportedUniFP:
    """Standalone inference on the same observation dictionary as ActorCritic."""
    def __init__(self, directory, *, device='cpu'):
        directory = Path(directory)
        self.contract = json.loads((directory / 'contract.json').read_text())
        if self.contract.get('format') != 'pawcerto.unifp.student.v1':
            raise ValueError('Expected a PawCerto UniFP student export')
        self.config = self.contract['config']
        self.policy = torch.jit.load(str(directory / 'policy.pt'), map_location=device).eval()
        self.estimator = torch.jit.load(str(directory / 'estimator.pt'), map_location=device).eval()

    @torch.inference_mode()
    def act_inference(self, observations):
        return self.policy(observations['obs'])

    @torch.inference_mode()
    def get_student_latent(self, history):
        return self.estimator(history)


def compare_export(loaded, directory, observations):
    """Compare reloaded CPU modules against the saved model, including estimates."""
    observations = observations.detach().to(device='cpu', dtype=torch.float32)
    if (observations.ndim != 2 or observations.shape[0] == 0
            or observations.shape[1] != loaded.model.num_obs
            or not torch.isfinite(observations).all()):
        raise ValueError('Expected finite, nonempty actor history observations')
    exported = ExportedUniFP(directory)
    device = next(loaded.model.parameters()).device
    with torch.inference_mode():
        expected_action = loaded.model.act_inference({'obs': observations.to(device)}, policy_info={}).cpu()
        expected_estimate = loaded.model.get_student_latent(observations.to(device)).cpu()
        actual_action = exported.act_inference({'obs': observations})
        actual_estimate = exported.get_student_latent(observations)
    result = {'samples': len(observations), 'evidence': 'CPU inference parity; no physical rollout'}
    for name, actual, expected in [('actions', actual_action, expected_action),
                                    ('estimates', actual_estimate, expected_estimate)]:
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        result[name] = {'max_abs_error': float((actual - expected).abs().max()), 'passed': True}
    return result


def export_policy(loaded, output, observations=None):
    output = Path(output)
    model = deepcopy(loaded.model).cpu().eval()
    policy = torch.jit.script(_StudentPolicy(model).eval())
    estimator = torch.jit.script(_Estimator(model).eval())
    output.mkdir(parents=True, exist_ok=False)
    policy.save(str(output / 'policy.pt'))
    estimator.save(str(output / 'estimator.pt'))
    env = loaded.config['env']['env']
    contract = {
        'format': 'pawcerto.unifp.student.v1', 'checkpoint': loaded.identity,
        'iteration': loaded.iteration, 'global_steps': loaded.global_steps,
        'config': loaded.config, 'dtype': 'float32',
        'input': ['batch', env['num_observations']],
        'policy.pt': ['batch', env['num_actions']],
        'estimator.pt': ['batch', env['num_pred_obs']],
        'history': 'oldest to newest; zero every history slot on reset, then append observed frame',
        'inference': 'deterministic student mean; current final frame plus encoded full history',
        'estimator': 'decoded base velocity, EE spherical position, injected EE force and injected base force; scaled training labels, not measured forces; decoded estimates do not directly enter actor',
        'action': 'raw 17-action output; method retains action clipping, 19-joint PD and effort saturation',
        'physical_evaluation': 'not performed by export',
    }
    (output / 'contract.json').write_text(json.dumps(contract, indent=2, allow_nan=False) + '\n')
    source = 'provided observations'
    if observations is None:
        observations = torch.randn(7, env['num_observations'], generator=torch.Generator().manual_seed(2027))
        observations[0].zero_()
        source = 'seeded synthetic history including zeros'
    comparison = compare_export(loaded, output, observations)
    comparison['input_source'] = source
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    return comparison
