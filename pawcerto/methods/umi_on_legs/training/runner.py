"""Thin runner for the vendored official UMI-on-Legs PPO.

An environment owns physics, observations, rewards, domain randomization and resets.
reset() returns (actor_obs, privileged_obs); step(action) returns those tensors plus
reward[N], done[N], info. info['time_outs'] includes timeout and safe-bound exits.
Observations returned by step must already include resets, as in the original env.
The original PPO bootstraps timeouts with the pre-step V(s), intentionally preserved.
"""
from pathlib import Path
from copy import deepcopy
import json
import random
import time
import numpy as np
import torch
from torch import nn
from .actor_critic import ActorCritic
from .ppo import PPO
from .semantics import require_resume_contract

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / 'reference/checkpoints/tossing/ours/config.json'


def load_config(path=DEFAULT_CONFIG):
    with open(path) as stream:
        return json.load(stream)


def _network(spec):
    if spec['_target_'] != 'torch.nn.Sequential':
        raise ValueError('This UMI adapter supports the official feedforward Sequential network')
    constructors = {'torch.nn.Linear': nn.Linear, 'torch.nn.ELU': nn.ELU}
    return nn.Sequential(*(constructors[item['_target_']](**{
        k: v for k, v in item.items() if not k.startswith('_')
    }) for item in spec['_args_']))


def build_algorithm(config, device='cpu'):
    cfg = config['runner']['alg']
    ac = cfg['actor_critic']
    model = ActorCritic(actor=_network(ac['actor']), critic=_network(ac['critic']),
                        num_actions=ac['num_actions'], init_noise_std=ac['init_noise_std'])
    return PPO(actor_critic=model, device=device, **{
        k: v for k, v in cfg.items() if not k.startswith('_') and k not in {'actor_critic', 'device'}
    })


class UmiTrainer:
    def __init__(self, env, config=None, device='cpu', rollout_steps=None):
        self.env = env
        source_config = getattr(env, 'config', None) if config is None else config
        self.config = deepcopy(load_config() if source_config is None else source_config)
        actual_contract = getattr(env, 'runtime_metadata', None)
        configured_contract = self.config.get('pawcerto_runtime')
        if actual_contract is not None:
            if configured_contract is not None and configured_contract != actual_contract:
                raise ValueError('Trainer config and actual environment runtime contracts differ')
            self.config['pawcerto_runtime'] = dict(actual_contract)
        self.runtime_metadata = deepcopy(self.config.get('pawcerto_runtime'))
        self.device = torch.device(device)
        self.alg = build_algorithm(self.config, device)
        cfg = self.config['runner']
        self.rollout_steps = cfg['num_transitions_per_env'] if rollout_steps is None else rollout_steps
        ac = cfg['alg']['actor_critic']
        batch = env.num_envs * self.rollout_steps
        if batch < 2 or batch < self.alg.num_mini_batches:
            raise ValueError('Rollout must contain at least two samples and one sample per minibatch')
        self.alg.init_storage(env.num_envs, self.rollout_steps, ac['num_actor_obs'],
                              ac['num_critic_obs'], ac['num_actions'], 1)
        self.iteration = 0
        self.total_transitions = 0
        self.obs = None
        self.critic_obs = None

    def reset(self):
        self.obs, self.critic_obs = self.env.reset()
        if self.config['runner']['init_at_random_ep_len']:
            # Changes the trajectory clock after reset, matching the source runner.
            self.env.randomize_episode_progress()
        self.obs = self.obs.to(self.device)
        self.critic_obs = self.critic_obs.to(self.device)

    def train_iteration(self):
        if self.obs is None:
            self.reset()
        start = time.monotonic()
        self.alg.actor_critic.train()
        totals = {}
        with torch.inference_mode():
            for _ in range(self.rollout_steps):
                # Clone snapshots: simulator adapters may update their observation buffers in-place.
                action = self.alg.act(self.obs.clone(), self.critic_obs.clone())
                obs, critic_obs, reward, done, info = self.env.step(action)
                self.obs, self.critic_obs = obs.to(self.device), critic_obs.to(self.device)
                self.alg.process_env_step(reward.to(self.device), done.to(self.device), info)
                totals['reward'] = totals.get('reward', 0.) + reward.mean().item()
                totals['terminated'] = totals.get('terminated', 0.) + done.sum().item()
                for key, value in info.items():
                    if isinstance(value, torch.Tensor) and value.numel() and value.is_floating_point():
                        totals[key] = totals.get(key, 0.) + value.mean().item()
            self.alg.compute_returns(self.critic_obs)
        collection_time = time.monotonic() - start
        stats = self.alg.update(self.iteration)
        # Observe the final update against the collected behavior policy without
        # sampling actions or changing PPO's update schedule/RNG consumption.
        with torch.no_grad():
            storage = self.alg.storage
            mean = self.alg.actor_critic.actor(storage.observations.flatten(0,1))
            old_mean = storage.mu.flatten(0,1)
            old_std = storage.sigma.flatten(0,1)
            std = self.alg.actor_critic.std
            kl = (torch.log(std/old_std+1e-5) +
                  (old_std.square()+(old_mean-mean).square())/(2*std.square())-.5).sum(-1)
            stats['final_policy_kl'] = kl.mean().item()
        if not all(np.isfinite(value) for value in stats.values()):
            raise FloatingPointError(f'Nonfinite PPO update: {stats}')
        if not all(torch.isfinite(p).all() for p in self.alg.actor_critic.parameters()):
            raise FloatingPointError('Nonfinite actor/critic parameters')
        self.alg.storage.clear()
        self.iteration += 1
        self.total_transitions += self.rollout_steps * self.env.num_envs
        return dict(iteration=self.iteration, total_transitions=self.total_transitions,
                    collection_seconds=collection_time, iteration_seconds=time.monotonic()-start,
                    **stats, **{k: v / self.rollout_steps for k, v in totals.items()})

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(model_state_dict=self.alg.get_model_state_dict(),
                     optimizer_state_dict=self.alg.get_optimizer_state_dict(), iter=self.iteration,
                     total_transitions=self.total_transitions, learning_rate=self.alg.learning_rate,
                     config=self.config, infos=None, torch_rng=torch.get_rng_state(),
                     numpy_rng=np.random.get_state(), python_rng=random.getstate())
        if self.device.type == 'cuda':
            state['cuda_rng'] = torch.cuda.get_rng_state_all()
        if hasattr(self.env, 'training_state_dict'):
            state['env_training_state'] = self.env.training_state_dict()
        torch.save(state, path)

    def load_actor(self, path):
        """Initialize an AS2 actor across assembly changes, retaining a fresh critic."""
        from ..robot_binding import require_same_as2_actor
        if self.iteration or self.total_transitions or self.alg.optimizer.state:
            raise ValueError('Actor initialization requires a fresh trainer')
        state = torch.load(path, map_location='cpu', weights_only=False)
        require_same_as2_actor(self.config, state.get('config', {}))
        current = self.alg.actor_critic.state_dict()
        source = {k: v for k, v in state['model_state_dict'].items()
                  if k == 'std' or k.startswith('actor.')}
        expected = {k for k in current if k == 'std' or k.startswith('actor.')}
        if set(source) != expected or any(source[k].shape != current[k].shape for k in expected):
            raise ValueError('AS2 actor checkpoint tensors differ from the destination network')
        current.update(source)
        self.alg.actor_critic.load_state_dict(current, strict=True)
        self.obs = self.critic_obs = None
        return state.get('infos')

    def load(self, path, load_optimizer=True):
        # Load RNG ByteTensors on CPU even when training is on CUDA.
        state = torch.load(path, map_location='cpu', weights_only=False)
        if load_optimizer and self.runtime_metadata is not None:
            require_resume_contract(state.get('config',{}).get('pawcerto_runtime'),
                                    self.runtime_metadata)
        self.alg.actor_critic.load_state_dict(state['model_state_dict'], strict=True)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.alg.learning_rate = state.get('learning_rate', self.alg.optimizer.param_groups[0]['lr'])
            self.iteration = state.get('iter', 0)
            self.total_transitions = state.get('total_transitions', 0)
            if 'torch_rng' in state:
                torch.set_rng_state(state['torch_rng'].cpu())
                np.random.set_state(state['numpy_rng'])
                random.setstate(state['python_rng'])
            if 'cuda_rng' in state and self.device.type == 'cuda':
                torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
            if 'env_training_state' in state:
                self.env.load_training_state_dict(state['env_training_state'])
        # Physics is reset on continuation; checkpoints do not claim exact rollout replay.
        self.obs = self.critic_obs = None
        return state.get('infos')
