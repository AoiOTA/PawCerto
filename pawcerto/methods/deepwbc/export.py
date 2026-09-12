"""CPU-safe, history-conditioned DeepWBC checkpoint and TorchScript consumption.

Only load trusted checkpoints: runner checkpoints use Python pickle. Exported
modules consume observations, not raw sensors, and do not own history or control.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .config import UPSTREAM_SHA
from .controller import POLICY_DOF_NAMES
from .policy import build_model


@dataclass
class LoadedCheckpoint:
    model: nn.Module
    config: dict
    iteration: int | None
    checkpoint_path: Path
    checkpoint_sha256: str


def load_checkpoint(path, *, config=None, device='cpu'):
    """Strictly restore a trusted full runner checkpoint without PPO/environment.

    An original-author checkpoint lacks configuration; require its complete
    resolved configuration rather than guessing a network or training recipe.
    """
    path = Path(path).resolve()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or 'model_state_dict' not in saved:
        raise ValueError('Expected a full runner checkpoint with model_state_dict')
    embedded = saved.get('config')
    if embedded is not None and config is not None and embedded != config:
        raise ValueError('Explicit config differs from checkpoint config')
    config = deepcopy(embedded if embedded is not None else config)
    if config is None:
        raise ValueError('Checkpoint lacks config; provide its complete resolved config')
    env = config['env']
    expected = env['num_proprio'] * (env['history_len'] + 1) + env['num_priv']
    if expected != env['num_observations']:
        raise ValueError('Observation width disagrees with proprio/priv/history layout')
    model = build_model(config, device='cpu')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.to(device).eval().requires_grad_(False)
    return LoadedCheckpoint(model, config, saved.get('iter'), path,
                            hashlib.sha256(path.read_bytes()).hexdigest())


class _ActorFromLatent(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.backbone = actor.actor_backbone
        self.leg = actor.actor_leg_control_head
        self.arm = actor.actor_arm_control_head
        self.adaptive_arm_gains = actor.adaptive_arm_gains
        self.adaptive_arm_gains_scale = actor.adaptive_arm_gains_scale
        self.num_arm_actions = actor.num_arm_actions

    def forward(self, proprio_and_latent):
        features = self.backbone(proprio_and_latent)
        arm = self.arm(features)
        if self.adaptive_arm_gains:
            arm = torch.cat((arm[:, :self.num_arm_actions // 2],
                             arm[:, self.num_arm_actions // 2:] * self.adaptive_arm_gains_scale), -1)
        return torch.cat((self.leg(features), arm), -1)


class _HistoryPolicy(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.encoder = actor.history_encoder
        self.action = _ActorFromLatent(actor)
        self.num_prop = actor.num_prop
        self.history_width = actor.num_hist * actor.num_prop
        self.observation_width = actor.num_prop + actor.num_priv + self.history_width

    def forward(self, observations):
        if observations.dim() != 2 or observations.size(1) != self.observation_width:
            raise RuntimeError('Expected [batch, full_observation_width]')
        latent = self.encoder(observations[:, -self.history_width:])
        return self.action(torch.cat((observations[:, :self.num_prop], latent), 1))


def _observations(loaded, observations):
    if observations is None:
        generator = torch.Generator(device='cpu').manual_seed(8159)
        observations = torch.randn(7, loaded.config['env']['num_observations'], generator=generator)
        observations[0].zero_()
    observations = observations.detach().to(device='cpu', dtype=torch.float32)
    if observations.ndim != 2 or observations.shape[0] == 0 or observations.shape[1] != loaded.config['env']['num_observations']:
        raise ValueError('Comparison observations must be nonempty [batch, num_observations]')
    if not torch.isfinite(observations).all():
        raise ValueError('Comparison observations contain nonfinite values')
    return observations


def compare_export(loaded, export_dir, observations, *, atol=1e-6, rtol=1e-5):
    """Reload standalone modules and compare raw actions, without a simulator."""
    directory = Path(export_dir)
    obs = _observations(loaded, observations)
    policy = torch.jit.load(str(directory / 'policy.pt'), map_location='cpu').eval()
    encoder = torch.jit.load(str(directory / 'history_encoder.pt'), map_location='cpu').eval()
    actor = torch.jit.load(str(directory / 'actor.pt'), map_location='cpu').eval()
    cfg = loaded.config['env']
    device = next(loaded.model.parameters()).device
    with torch.inference_mode():
        reference = loaded.model.act_inference(obs.to(device), hist_encoding=True).cpu()
        fused = policy(obs)
        history = obs[:, -cfg['history_len'] * cfg['num_proprio']:]
        split = actor(torch.cat((obs[:, :cfg['num_proprio']], encoder(history)), 1))
    results = {}
    for name, actual in [('policy', fused), ('split', split)]:
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        results[name] = {'max_abs_error': float((actual-reference).abs().max()), 'passed': True}
    return {'samples': len(obs), 'atol': atol, 'rtol': rtol, 'actions': results,
            'evidence': 'numerical action parity only; not physical evaluation'}


def export_policy(loaded, output_dir, observations=None):
    """Export fused and upstream-style split modules, then independently reload."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    actor = deepcopy(loaded.model.actor).cpu().eval()
    # Script the fused consumer so width validation and dynamic batch survive export.
    fused = torch.jit.script(_HistoryPolicy(actor).eval())
    fused.save(str(directory / 'policy.pt'))
    torch.jit.script(_ActorFromLatent(actor).eval()).save(str(directory / 'actor.pt'))
    torch.jit.script(actor.history_encoder).save(str(directory / 'history_encoder.pt'))
    cfg = loaded.config['env']
    latent_width = loaded.config['policy']['priv_encoder_dims'][-1] if loaded.config['policy']['priv_encoder_dims'] else cfg['num_priv']
    contract = {
        'format': 'deepwbc-history-policy-v1', 'upstream_sha': UPSTREAM_SHA,
        'checkpoint': str(loaded.checkpoint_path), 'checkpoint_sha256': loaded.checkpoint_sha256,
        'iteration': loaded.iteration, 'dtype': 'float32', 'batch': 'dynamic, nonempty',
        'policy.pt': {'input': ['batch', cfg['num_observations']], 'output': ['batch', cfg['num_actions']]},
        'actor.pt': {'input': ['batch', cfg['num_proprio'] + latent_width], 'output': ['batch', cfg['num_actions']]},
        'history_encoder.pt': {'input': ['batch', cfg['num_proprio'] * cfg['history_len']], 'output': ['batch', latent_width]},
        'observation_layout': {'proprio': [0, cfg['num_proprio']],
            'privileged': [cfg['num_proprio'], cfg['num_proprio'] + cfg['num_priv']],
            'history': [cfg['num_proprio'] + cfg['num_priv'], cfg['num_observations']]},
        'history': 'oldest to newest, flattened; observer returns observation BEFORE history update; reset history is zero, then repeated proprio when episode_length <= 1',
        'inference': 'deterministic act_inference(hist_encoding=True); privileged slots ignored',
        'observations': 'Use DeepWBCObserver output including its scaling, ordering and clipping; no extra normalization. Proprio: roll/pitch 2, angular velocity 3, joint positions 20, joint velocities 20, latest actions 18, foot contacts 4, commands 3, EE target 3, EE orientation delta 3.',
        'actions': 'Raw actor output, NOT torque or joint targets. Environment/controller retains clipping, delay, reordering, action scale and PD torque computation.',
        'action_joint_order': list(POLICY_DOF_NAMES[:18]),
        'config': loaded.config,
        'physical_evaluation': 'not performed by export',
    }
    (directory / 'contract.json').write_text(json.dumps(contract, indent=2, allow_nan=False) + '\n')
    source = 'provided observations' if observations is not None else 'seeded synthetic observations, including zeros'
    comparison = compare_export(loaded, directory, _observations(loaded, observations))
    comparison['input_source'] = source
    (directory / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    return comparison
