# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""DeepWBC rollout orchestration, retaining the upstream alternating update order.

Environment contract: num_envs/num_obs/num_privileged_obs/num_actions;
reset(), get_observations(), get_privileged_observations(),
update_command_curriculum(), step(actions) -> (obs, privileged_obs,
leg_rewards, arm_rewards, dones, infos). Observations use policy.py layout;
rewards/dones/time_outs are vectors with num_envs entries.
"""
from copy import deepcopy
from pathlib import Path
import warnings
import torch
from ..policy import build_model
from .ppo import PPO


class _HistoryPolicy(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observations):
        return self.actor(observations, hist_encoding=True)


class OnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.env = env
        self.config = deepcopy(train_cfg)
        self.cfg = self.config['runner']
        self.alg_cfg = self.config['algorithm']
        self.device = torch.device(device)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.alg = PPO(build_model(self.config, device=device), device=device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg['num_steps_per_env']
        self.save_interval = self.cfg['save_interval']
        self.dagger_update_freq = self.alg_cfg['dagger_update_freq']
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.alg.init_storage(env.num_envs, self.num_steps_per_env, [env.num_obs],
                              [env.num_privileged_obs], [env.num_actions])
        env.reset()
        if self.alg.torque_supervision:
            self.alg.set_arm_default_coeffs(env.default_arm_p_gains,
                                            env.default_arm_d_gains,
                                            env.default_arm_dof_pos)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, *, save_final=True):
        """Run updates; callers streaming one iteration may defer the final save."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations().to(self.device)
        priv = self.env.get_privileged_observations()
        critic_obs = obs if priv is None else priv.to(self.device)
        self.alg.actor_critic.train()
        results = []
        stop = self.current_learning_iteration + num_learning_iterations
        for iteration in range(self.current_learning_iteration, stop):
            self.env.update_command_curriculum()
            hist_encoding = iteration % self.dagger_update_freq == 0
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, hist_encoding)
                    obs, priv, rewards, arm_rewards, dones, infos = self.env.step(actions)
                    obs = obs.to(self.device)
                    critic_obs = obs if priv is None else priv.to(self.device)
                    self.alg.process_env_step(rewards.to(self.device), arm_rewards.to(self.device),
                                              dones.to(self.device), infos)
                self.alg.compute_returns(critic_obs)
            losses = self.alg.update_dagger() if hist_encoding else self.alg.update()
            self.current_learning_iteration = iteration + 1
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            results.append({'iteration': iteration, 'stage': 'dagger' if hist_encoding else 'ppo',
                            'losses': losses})
            if self.log_dir is not None and self.current_learning_iteration % self.save_interval == 0:
                self.save(self.log_dir / f'model_{self.current_learning_iteration}.pt')
        if self.log_dir is not None and save_final and self.current_learning_iteration % self.save_interval != 0:
            self.save(self.log_dir / f'model_{self.current_learning_iteration}.pt')
        return results

    def save(self, path, infos=None):
        """Keep upstream keys; add omitted adaptation/schedule state for continuation."""
        torch.save({'model_state_dict': self.alg.actor_critic.state_dict(),
                    'optimizer_state_dict': self.alg.optimizer.state_dict(),
                    'hist_encoder_optimizer_state_dict': self.alg.hist_encoder_optimizer.state_dict(),
                    'iter': self.current_learning_iteration, 'algorithm_counter': self.alg.counter,
                    'learning_rate': self.alg.learning_rate, 'infos': infos,
                    'rng_state': torch.get_rng_state(), 'tot_timesteps': self.tot_timesteps,
                    'config': self.config}, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'hist_encoder_optimizer_state_dict' in checkpoint:
                self.alg.hist_encoder_optimizer.load_state_dict(checkpoint['hist_encoder_optimizer_state_dict'])
            else:
                warnings.warn('Upstream checkpoint has no history optimizer state; adaptation Adam starts fresh.', stacklevel=2)
        self.current_learning_iteration = checkpoint['iter']
        self.alg.counter = checkpoint.get('algorithm_counter', self.current_learning_iteration)
        self.alg.learning_rate = checkpoint.get('learning_rate', self.alg.optimizer.param_groups[0]['lr'])
        self.tot_timesteps = checkpoint.get('tot_timesteps', 0)
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].cpu())
        return checkpoint.get('infos')

    def get_inference_policy(self, device=None, stochastic=False):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act if stochastic else self.alg.actor_critic.act_inference

    def export(self, path):
        """Trace the original history-conditioned actor; input is full observation."""
        actor = deepcopy(self.alg.actor_critic.actor).cpu().eval()
        policy = _HistoryPolicy(actor).eval()
        example = torch.zeros(1, self.config['env']['num_observations'])
        traced = torch.jit.trace(policy, example)
        traced.save(str(path))
        return traced
