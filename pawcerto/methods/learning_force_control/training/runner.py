"""Original PPO-CSE collection/update loop with local recoverable checkpoints.

Logging/W&B orchestration is replaced by local JSONL. The original algorithm
and rollout length remain unchanged. A finite update is not learning success.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import random
import time
import numpy as np
import torch
from ..config import build_model, network_config
from ..ppo import PPO
from ..export import export_policy

FORMAT = 'pawcerto.learning_force_control.training.v1'


class OnPolicyRunner:
    def __init__(self, env, config, output, device='cpu', *, initialize=True):
        self.env,self.config,self.output,self.device = env,deepcopy(config),Path(output),device
        self.output.mkdir(parents=True,exist_ok=True)
        self.alg = PPO(build_model(network_config(config)).to(device),device=device,
                       config=SimpleNamespace(**deepcopy(config['ppo'])))
        self.num_steps_per_env = config['runner']['num_steps_per_env']
        self.alg.init_storage(env.num_envs,self.num_steps_per_env,[env.num_obs],[env.num_privileged_obs],[env.num_obs_history],[env.num_actions])
        self.next_iteration = 0
        self.global_steps = 0
        self.last_observations = None
        if initialize:
            with torch.no_grad():env.reset()

    def learn(self, iterations, *, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            if self.next_iteration:
                raise ValueError('Resume must retain saved episode counters')
            self.env.episode_length_buf[:] = torch.randint_like(self.env.episode_length_buf,high=int(self.env.max_episode_length))
        if self.last_observations is None:
            self.last_observations = self.env.get_observations()
        self.alg.actor_critic.train()
        for _ in range(iterations):
            start = time.monotonic()
            resets = 0
            reward_sum = 0.
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    obs = self.last_observations
                    actions = self.alg.act(obs['obs'],obs['privileged_obs'],obs['obs_history'])
                    self.last_observations,reward,done,info = self.env.step(actions)
                    self.alg.process_env_step(reward,done,info)
                    self.global_steps += self.env.num_envs
                    resets += int(done.sum())
                    reward_sum += float(reward.mean())
                    if not torch.isfinite(reward).all() or any(not torch.isfinite(v).all() for v in self.last_observations.values()):
                        raise FloatingPointError('Non-finite real rollout tensor; original transition retained in storage')
                obs = self.last_observations
                self.alg.compute_returns(obs['obs_history'],obs['privileged_obs'])
            losses = self.alg.update()
            if any(not torch.isfinite(p).all() for p in self.alg.actor_critic.parameters()):
                raise FloatingPointError('Non-finite policy after original PPO-CSE update')
            self.next_iteration += 1
            record = {'next_iteration':self.next_iteration,'global_steps':self.global_steps,
                      'common_step_counter':self.env.common_step_counter,
                      'mean_reward':reward_sum/self.num_steps_per_env,'resets':resets,
                      'value_loss':losses[0],'surrogate_loss':losses[1],
                      'adaptation_loss':losses[2],'adaptation_losses':losses[-1],
                      'learning_rate':self.alg.learning_rate,'seconds':time.monotonic()-start,
                      'force_envs':int((self.env.force_or_position_control==1).sum()),
                      'applied_force_max_N':float(self.env.forces[:,self.env.gripper_stator_index].abs().max()),
                      'learning_success':False}
            with (self.output/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            print('LFC_UPDATE '+json.dumps(record),flush=True)
            if self.next_iteration%self.config['runner']['save_interval']==0:
                self.save(self.output/f'model_{self.next_iteration}.pt')
        self.save(self.output/f'model_{self.next_iteration}.pt')
        torch.save(self.alg.actor_critic.state_dict(),self.output/'ac_weights_latest.pt')
        export_policy(self.alg.actor_critic,self.output/'export')

    def save(self,path):
        if self.alg.storage.step:
            raise ValueError('Save at a complete PPO rollout/update boundary')
        torch.save({'format':FORMAT,'model_state_dict':self.alg.actor_critic.state_dict(),
                    'optimizer_state_dict':self.alg.optimizer.state_dict(),
                    'adaptation_optimizer_state_dict':self.alg.adaptation_module_optimizer.state_dict(),
                    'learning_rate':self.alg.learning_rate,'next_iteration':self.next_iteration,
                    'global_steps':self.global_steps,'common_step_counter':self.env.common_step_counter,
                    'config':self.config,'env_state':self.env.training_state(),
                    'last_observations':None if self.last_observations is None else {k:v.detach().cpu().clone() for k,v in self.last_observations.items()},
                    'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate(),
                           'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                    'resume_semantics':'Restores exposed task and physics state; PhysX solver warm-start/contact caches unavailable'},path)

    def load(self,path):
        saved = torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('format') != FORMAT:raise ValueError('Expected full LFC training checkpoint')
        if saved['config'] != self.config:raise ValueError('Resume configuration differs')
        self.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
        self.alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
        self.alg.adaptation_module_optimizer.load_state_dict(saved['adaptation_optimizer_state_dict'])
        self.alg.learning_rate = saved['learning_rate']
        self.next_iteration,self.global_steps = saved['next_iteration'],saved['global_steps']
        self.env.load_training_state(saved['env_state'])
        self.last_observations = None if saved['last_observations'] is None else {k:v.to(self.device) for k,v in saved['last_observations'].items()}
        # torch CPU RNG must remain a CPU ByteTensor even for CUDA training.
        torch.set_rng_state(saved['rng']['torch'].cpu())
        np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
        if saved['rng']['cuda'] is not None:torch.cuda.set_rng_state_all([v.cpu() for v in saved['rng']['cuda']])
        return saved
