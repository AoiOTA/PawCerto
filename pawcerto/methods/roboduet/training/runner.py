"""Automatic RoboDuet rollout order and iteration-boundary full checkpoints."""
from copy import deepcopy
from pathlib import Path
import os
import random

import numpy as np
import torch

from ..policy import build_models
from .ppo import PPO

FORMAT = 'pawcerto.roboduet.training.v1'


def checkpoint_arm_update_count(payload):
    """Migrate early full-run v1 checkpoints, never upstream weights-only files.

    This runner starts at iteration zero, updates the arm every Stage 2
    iteration, and opens Stage 2 after the configured switch iteration.
    """
    if payload['format'] != FORMAT:
        raise ValueError('Expected a complete RoboDuet training checkpoint')
    if 'arm_update_count' in payload:
        return int(payload['arm_update_count'])
    start = payload['config']['stage']['pretrained_to_hybrid_start']
    return max(0, int(payload['next_iteration']) - int(start) - 1)


class RoboDuetRunner:
    """One complete update per call; environment implements the automatic wrapper API.

    Checkpoints include the stale arm observation retained across Stage 1. Physical
    restore precision is the environment's contract, not a bitwise PhysX guarantee.
    """
    def __init__(self, env, config, device='cpu', *, init_at_random_ep_len=True):
        self.env, self.config, self.device = env, deepcopy(config), torch.device(device)
        self.dog_model, self.arm_model = build_models(config, device)
        self.alg_dog = PPO(self.dog_model, device, config['PPO_Args'])
        self.alg_arm = PPO(self.arm_model, device, config['PPO_Args'])
        self.num_steps_per_env = config['RunnerArgs']['num_steps_per_env']
        if self.num_steps_per_env < 1:
            raise ValueError('num_steps_per_env must be positive')
        for side, alg in [('dog', self.alg_dog), ('arm', self.alg_arm)]:
            cfg = config['Cfg'][side]
            action_dim = cfg['dog_actions'] if side == 'dog' else cfg['num_actions_arm_cd']
            alg.init_storage(env.num_envs, self.num_steps_per_env,
                             [cfg[f'{side}_num_observations']], [cfg[f'{side}_num_privileged_obs']],
                             [cfg[f'{side}_num_obs_history']], [action_dim], [action_dim])
        self.next_iteration = 0
        self.total_transitions = 0
        self.arm_update_count = 0
        self.stage = 1
        self._at_boundary = True
        env.set_stage(1)
        env.reset()
        if init_at_random_ep_len:
            env.randomize_episode_lengths()
        self.arm_obs = env.get_arm_observations()

    def run_iteration(self):
        if not self._at_boundary:
            raise RuntimeError('An incomplete update cannot be resumed as a new rollout')
        self._at_boundary = False
        iteration, rollout_stage = self.next_iteration, self.stage
        self.env.set_iteration(iteration)
        self.dog_model.train()
        self.arm_model.train()
        arm_actions = torch.zeros(self.env.num_envs, 8, device=self.device)
        dog_reward_sum = arm_reward_sum = 0.0
        done_count = 0
        with torch.inference_mode():
            # The final arm sample/plan and dog getter are deliberate upstream RNG
            # and history operations, even though there is no final physics step.
            for step in range(self.num_steps_per_env + 1):
                if rollout_stage == 2:
                    arm_actions = self.alg_arm.act(self.arm_obs['obs'], self.arm_obs['privileged_obs'],
                                                   self.arm_obs['obs_history'])
                    self.env.plan(arm_actions[..., -2:])
                dog_obs = self.env.get_dog_observations()
                if step:
                    self.alg_dog.process_env_step(rewards_dog, dones, infos)
                    if step == self.num_steps_per_env:
                        break
                dog_actions = self.alg_dog.act(dog_obs['obs'], dog_obs['privileged_obs'], dog_obs['obs_history'])
                rewards_dog, rewards_arm, dones, infos = self.env.step(dog_actions, arm_actions[..., :-2])
                if rollout_stage == 2:
                    self.arm_obs = self.env.get_arm_observations()
                    self.alg_arm.process_env_step(rewards_arm, dones, infos)
                # In-place history clearing preserves the upstream alias behavior.
                self.env.clear_cached(dones.nonzero(as_tuple=False).flatten())
                dog_reward_sum += rewards_dog.mean().item()
                arm_reward_sum += rewards_arm.mean().item()
                done_count += dones.count_nonzero().item()
            if rollout_stage == 2:
                self.alg_arm.compute_returns(self.arm_obs['obs_history'], self.arm_obs['privileged_obs'])
            self.alg_dog.compute_returns(dog_obs['obs_history'], dog_obs['privileged_obs'])
        arm_losses = self.alg_arm.update() if rollout_stage == 2 else None
        if rollout_stage == 2:
            self.arm_update_count += 1
        dog_losses = self.alg_dog.update()
        self.next_iteration += 1
        self.total_transitions += self.num_steps_per_env * self.env.num_envs
        if iteration == self.config['stage']['pretrained_to_hybrid_start']:
            self.stage = 2
            self.env.set_stage(2)
        # Upstream leaves a bootstrap-only arm transition; it is overwritten by
        # next act and is never inserted into storage.
        self.alg_arm.transition.clear()
        self._at_boundary = True
        return dict(iteration=iteration, next_iteration=self.next_iteration, rollout_stage=rollout_stage,
                    stage=self.stage, total_transitions=self.total_transitions, arm_update_count=self.arm_update_count,
                    dog_losses=dog_losses, arm_losses=arm_losses,
                    dog_reward_mean=dog_reward_sum/self.num_steps_per_env,
                    arm_reward_mean=arm_reward_sum/self.num_steps_per_env, done_count=done_count,
                    dog_learning_rate=self.alg_dog.learning_rate, arm_learning_rate=self.alg_arm.learning_rate)

    def checkpoint(self):
        if not self._at_boundary:
            raise RuntimeError('Save only at a completed iteration boundary')
        payload = dict(format=FORMAT, config=deepcopy(self.config),
                       next_iteration=self.next_iteration, stage=self.stage,
                       total_transitions=self.total_transitions, arm_update_count=self.arm_update_count,
                       env_training_state=self.env.training_state(), arm_observation_cache=self.arm_obs,
                       torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                       python_rng=random.getstate(),
                       cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None)
        for side, alg in [('dog', self.alg_dog), ('arm', self.alg_arm)]:
            payload[f'{side}_model_state_dict'] = alg.actor_critic.state_dict()
            payload[f'{side}_optimizer'] = alg.optimizer.state_dict()
            payload[f'{side}_adaptation_optimizer'] = alg.adaptation_module_optimizer.state_dict()
            payload[f'{side}_learning_rate'] = alg.learning_rate
        return payload

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        try:
            torch.save(self.checkpoint(), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, path):
        # Keep CPU RNG a CPU uint8 tensor; optimizer loading moves its own state.
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['format'] != FORMAT or payload['config'] != self.config:
            raise ValueError('Checkpoint format/configuration mismatch')
        next_iteration = payload['next_iteration']
        expected_stage = 2 if next_iteration > self.config['stage']['pretrained_to_hybrid_start'] else 1
        if payload['stage'] != expected_stage:
            raise ValueError('Checkpoint stage does not match its next iteration')
        for side, alg in [('dog', self.alg_dog), ('arm', self.alg_arm)]:
            alg.actor_critic.load_state_dict(payload[f'{side}_model_state_dict'])
            alg.optimizer.load_state_dict(payload[f'{side}_optimizer'])
            alg.adaptation_module_optimizer.load_state_dict(payload[f'{side}_adaptation_optimizer'])
            alg.learning_rate = payload[f'{side}_learning_rate']
            alg.storage.clear()
            alg.transition.clear()
        self.env.load_training_state(payload['env_training_state'])
        self.arm_obs = self.env.restore_arm_observation_cache(payload['arm_observation_cache'])
        self.next_iteration, self.stage = next_iteration, payload['stage']
        self.total_transitions = payload['total_transitions']
        self.arm_update_count = checkpoint_arm_update_count(payload)
        self.env.set_stage(self.stage)
        torch.set_rng_state(payload['torch_rng'].cpu())
        np.random.set_state(payload['numpy_rng'])
        random.setstate(payload['python_rng'])
        if payload['cuda_rng'] is not None:
            if self.device.type != 'cuda':
                raise ValueError('CUDA training resume requires a CUDA runtime')
            torch.cuda.set_rng_state_all([state.cpu() for state in payload['cuda_rng']])
        self._at_boundary = True
        return payload
