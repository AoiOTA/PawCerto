"""Five-module RoboDuet CPU exports, with the arm playback semantics explicit."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import torch

from .policy import build_models


MODULE_FILES = {
    'dog_body': 'body_latest_dog.jit',
    'dog_adaptation': 'adaptation_module_latest_dog.jit',
    'arm_body': 'body_latest_arm.jit',
    'arm_adaptation': 'adaptation_module_latest_arm.jit',
    'arm_history': 'history_latest_arm.jit',
}
MODES = ('official_play', 'training_mean')


class ExportedRoboDuet:
    """Consume all five modules; the environment still owns history and physics.

    Both methods use the same observation dictionaries as the original loader.
    The caller must run arm -> plan -> dog, rather than collecting the two
    observation dictionaries before writing the arm's body guidance.
    """
    def __init__(self, directory):
        directory = Path(directory)
        self.config = json.loads((directory / 'config.json').read_text())
        self.info = json.loads((directory / 'export.json').read_text())
        self.joint_names = json.loads((directory / 'joint_names.json').read_text())
        self.mode = self.info['arm_output_mode']
        if self.mode not in MODES:
            raise ValueError(f'Unknown arm output mode: {self.mode}')
        self.modules = {key: torch.jit.load(str(directory / name), map_location='cpu').eval()
                        for key, name in MODULE_FILES.items()}
        self.arm_num_obs = self.config['Cfg']['arm']['arm_num_observations']

    @torch.inference_mode()
    def dog_action(self, observations):
        history = observations['obs_history']
        latent = self.modules['dog_adaptation'](history)
        return self.modules['dog_body'](torch.cat((history, latent), dim=-1))

    @torch.inference_mode()
    def arm_action(self, observations):
        history = observations['obs_history']
        latent = self.modules['arm_adaptation'](history)
        encoded = self.modules['arm_history'](history[..., :-self.arm_num_obs])
        current = history[..., -self.arm_num_obs:] if self.mode == 'training_mean' else observations['obs']
        raw = self.modules['arm_body'](torch.cat((current, latent, encoded), dim=-1))
        if self.mode == 'training_mean':
            return torch.cat((raw[..., :-2], torch.tanh(raw[..., -2:])), dim=-1)
        return raw


def export_checkpoint(checkpoint_path, output, joint_names, *, mode='official_play'):
    """Export a PawCerto full training checkpoint without changing its weights.

    official_play reproduces upstream load_policy.py, including raw body-guidance
    outputs. training_mean uses history's latest frame and applies the training
    Gaussian mean's tanh to the last two arm outputs; it does not sample actions
    or reproduce a stochastic rollout.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown arm output mode: {mode}')
    if len(joint_names) != 20 or len(set(joint_names)) != 20:
        raise ValueError('Expected 20 distinct original Go1/ARX5 DOF names, including the two fixed gripper DOFs')
    checkpoint_path, output = Path(checkpoint_path), Path(output)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 'pawcerto.roboduet.training.v1':
        raise ValueError('Expected a complete PawCerto RoboDuet training checkpoint')
    from .training.runner import checkpoint_arm_update_count
    arm_updates = checkpoint_arm_update_count(checkpoint)
    config = checkpoint['config']
    dog, arm = build_models(config, device='cpu')
    dog.load_state_dict(checkpoint['dog_model_state_dict'])
    arm.load_state_dict(checkpoint['arm_model_state_dict'])
    dog.eval()
    arm.eval()
    originals = {
        'dog_body': dog.actor_body, 'dog_adaptation': dog.adaptation_module,
        'arm_body': arm.actor_body, 'arm_adaptation': arm.adaptation_module,
        'arm_history': arm.actor_history_encoder,
    }
    blobs = {}
    for key, module in originals.items():
        stream = io.BytesIO()
        torch.jit.save(torch.jit.script(deepcopy(module).cpu().eval()), stream)
        blobs[key] = stream.getvalue()
    info = {
        'format': 'pawcerto.roboduet.export.v1',
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'next_iteration': checkpoint['next_iteration'],
        'next_rollout_stage': checkpoint['stage'],
        'arm_update_count': arm_updates,
        'arm_output_mode': mode,
        'module_files': MODULE_FILES,
        'scope': 'CPU deterministic five-module consumption; no learning or physical behavior qualification',
        'stage_note': ('The arm has not received any PPO updates, even if the next rollout is Stage 2.'
                       if arm_updates == 0 else
                       f'The arm has received {arm_updates} PPO updates; this does not establish learned cooperation.'),
    }
    output.mkdir(parents=True, exist_ok=False)
    for key, blob in blobs.items():
        (output / MODULE_FILES[key]).write_bytes(blob)
    (output / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    (output / 'joint_names.json').write_text(json.dumps(joint_names, indent=2) + '\n')
    (output / 'export.json').write_text(json.dumps(info, indent=2) + '\n')
    restored = ExportedRoboDuet(output)
    generator = torch.Generator().manual_seed(2027)
    dog_history = torch.randn(4, config['Cfg']['dog']['dog_num_obs_history'], generator=generator)
    arm_history = torch.randn(4, config['Cfg']['arm']['arm_num_obs_history'], generator=generator)
    # Deliberately do not alias obs to history's latest slice. The official
    # deployment consumes the supplied current observation separately.
    arm_obs = torch.randn(4, config['Cfg']['arm']['arm_num_observations'], generator=generator)
    dog_input = {'obs_history': dog_history}
    arm_input = {'obs': arm_obs, 'obs_history': arm_history}
    with torch.inference_mode():
        dog_expected = dog.act_inference(dog_input)
        arm_expected = arm.act_inference(arm_input, mode=mode)
        dog_actual = restored.dog_action(dog_input)
        arm_actual = restored.arm_action(arm_input)
    torch.testing.assert_close(dog_actual, dog_expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(arm_actual, arm_expected, rtol=1e-6, atol=1e-6)
    info['reload_max_abs_error'] = {
        'dog': (dog_actual - dog_expected).abs().max().item(),
        'arm': (arm_actual - arm_expected).abs().max().item(),
    }
    (output / 'export.json').write_text(json.dumps(info, indent=2) + '\n')
    return info
