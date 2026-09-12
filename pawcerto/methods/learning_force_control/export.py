"""Consume original raw ac_weights state dictionaries and original two-JIT export.

Only load trusted checkpoint/JIT files. Raw upstream weights contain no config;
callers must supply the resolved configuration belonging to those weights.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
import argparse
import json
import torch
from torch import nn
from pawcerto.artifacts import file_identity
from .config import build_model, network_config, SOURCE_REVISION


def load_checkpoint(path, *, config):
    model = build_model(config)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def export_policy(model, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, module in [('body_latest.jit', model.actor_body),
                         ('adaptation_module_latest.jit', model.adaptation_module)]:
        torch.jit.script(deepcopy(module).cpu().eval()).save(str(destination / name))
    return destination


class JitPolicy(nn.Module):
    def __init__(self, directory):
        super().__init__()
        directory = Path(directory)
        self.body = torch.jit.load(str(directory / 'body_latest.jit'), map_location='cpu').eval()
        self.adaptation = torch.jit.load(str(directory / 'adaptation_module_latest.jit'), map_location='cpu').eval()

    def forward(self, obs_history):
        return self.body(torch.cat((obs_history, self.adaptation(obs_history)), dim=-1))


@dataclass
class LoadedTrainingCheckpoint:
    model: nn.Module
    config: dict
    next_iteration: int
    common_step_counter: int
    global_steps: int
    identity: dict


def load_training_checkpoint(path, *, device='cpu'):
    """Load an explicitly identified trusted full Lab checkpoint, not raw weights."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format') != 'pawcerto.learning_force_control.training.v1':
        raise ValueError('Expected a full PawCerto Learning Force Control checkpoint')
    config = deepcopy(saved['config'])
    model = build_model(network_config(config))
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.to(device).eval().requires_grad_(False)
    return LoadedTrainingCheckpoint(model, config, saved['next_iteration'],
        saved['common_step_counter'], saved['global_steps'], file_identity(path))


def compare_training_export(loaded, directory, histories):
    """Reload original-format modules and compare student actions and latents."""
    histories = histories.detach().to(device='cpu', dtype=torch.float32)
    width = network_config(loaded.config)['num_obs_history']
    if (histories.ndim != 2 or histories.shape[0] == 0 or histories.shape[1] != width
            or not torch.isfinite(histories).all()):
        raise ValueError('Expected finite nonempty policy histories of the configured width')
    consumer = JitPolicy(directory)
    device = next(loaded.model.parameters()).device
    with torch.inference_mode():
        reference_action = loaded.model.act_student(histories.to(device), policy_info={}).cpu()
        reference_latent = loaded.model.get_student_latent(histories.to(device)).cpu()
        action = consumer(histories)
        latent = consumer.adaptation(histories)
    report = {'samples': len(histories), 'atol': 1e-6, 'rtol': 1e-5,
              'evidence': 'CPU student inference parity; no physical execution'}
    for name, actual, expected in [('actions', action, reference_action),
                                    ('latents', latent, reference_latent)]:
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        report[name] = {'passed': True, 'max_abs_error': float((actual - expected).abs().max())}
    return report


def export_training_checkpoint(loaded, directory, histories=None):
    """Export the original JIT pair with this full checkpoint's actual contract."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    export_policy(loaded.model, directory)
    network = network_config(loaded.config)
    contract = {
        'format': 'pawcerto.learning_force_control.student.v1', 'source_revision': SOURCE_REVISION,
        'checkpoint': loaded.identity, 'next_iteration': loaded.next_iteration,
        'common_step_counter': loaded.common_step_counter, 'global_steps': loaded.global_steps,
        'config': loaded.config, 'network': network, 'dtype': 'float32',
        'adaptation_module_latest.jit': {'input': ['batch', network['num_obs_history']],
                                         'output': ['batch', network['num_privileged_obs']]},
        'body_latest.jit': {'input': ['batch', network['num_obs_history'] + network['num_privileged_obs']],
                            'output': ['batch', network['num_actions']]},
        'consumer': 'deterministic student: concatenate raw history and adaptation prediction, then body; no teacher privileged input',
        'actions': 'raw 19-output policy, not torque; original environment owns clipping, action18 overwrite, delay, scaling and position-drive targets',
        'force': 'actor consumes commands; applied force is training supervision, not a required deployed force sensor',
        'history': 'original ObservationHistory ordering and reset behavior; no additional normalization',
        'physical_evaluation': 'not performed by export',
    }
    (directory / 'contract.json').write_text(json.dumps(contract, indent=2, allow_nan=False) + '\n')
    source = 'provided histories'
    if histories is None:
        histories = torch.randn(7, network['num_obs_history'], generator=torch.Generator().manual_seed(2027))
        histories[0].zero_()
        source = 'seeded synthetic histories including zeros'
    result = compare_training_export(loaded, directory, histories)
    result['input_source'] = source
    (directory / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, config=json.loads(args.config.read_text()))
    export_policy(model, args.output)


if __name__ == '__main__':
    main()
