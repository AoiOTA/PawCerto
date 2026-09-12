"""RoboDuet automatic dog/arm networks with original state-dict names.

Adapted from locomanip-duet/RoboDuet at a7e1528215c048199f90cb69ceb7749a1d745f28
(MIT; PPO-related upstream attribution also includes LICENSES/rsl_rl/LICENSE).
The broken legacy arm act_student/act_teacher helpers are not a reference path:
training uses update_distribution, official playback uses scripts/load_policy.py.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal

from .config import default_config


def _activation(name):
    choices = {'elu': nn.ELU, 'selu': nn.SELU, 'relu': nn.ReLU, 'crelu': nn.ReLU,
               'lrelu': nn.LeakyReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
    if name not in choices:
        raise ValueError(f'Unknown RoboDuet activation: {name}')
    return choices[name]()


def _mlp(inputs, hidden, outputs, activation):
    layers = []
    for size in hidden:
        layers += [nn.Linear(inputs, size), _activation(activation)]
        inputs = size
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


class _ActorCritic(nn.Module):
    is_recurrent = False

    def reset(self, dones=None):
        # History belongs to the environment wrapper; the networks are stateless.
        pass

    def forward(self):
        raise NotImplementedError('Use act/evaluate or act_inference with explicit observations')

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, observation_history, **kwargs):
        self.update_distribution(observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)


class DogActorCritic(_ActorCritic):
    def __init__(self, num_obs, num_privileged_obs, num_obs_history, num_actions,
                 network_config=None, device=None):
        super().__init__()
        cfg = deepcopy(default_config()['DogAC_Args'] if network_config is None else network_config)
        if cfg['use_decoder']:
            raise ValueError('The default automatic path has no decoder implementation')
        self.decoder = False
        self.num_obs, self.num_obs_history = num_obs, num_obs_history
        self.num_privileged_obs = num_privileged_obs
        self.adaptation_module = _mlp(num_obs_history, cfg['adaptation_module_branch_hidden_dims'],
                                      num_privileged_obs, cfg['activation'])
        self.actor_body = _mlp(num_obs_history + num_privileged_obs, cfg['actor_hidden_dims'],
                               num_actions, cfg['activation'])
        self.critic_body = _mlp(num_obs_history + num_privileged_obs, cfg['critic_hidden_dims'],
                                1, cfg['activation'])
        self.std = nn.Parameter(cfg['init_noise_std'] * torch.ones(num_actions))
        self.distribution = None
        if device is not None:
            self.to(device)

    def update_distribution(self, observation_history):
        latent = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act_inference(self, ob, policy_info=None):
        history = ob['obs_history']
        latent = self.adaptation_module(history)
        if policy_info is not None:
            policy_info['latent'] = latent
        return self.actor_body(torch.cat((history, latent), dim=-1))

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        return self.critic_body(torch.cat((observation_history, privileged_observations), dim=-1))


class ArmActorCritic(_ActorCritic):
    def __init__(self, num_obs, num_privileged_obs, num_obs_history, num_actions,
                 network_config=None, device=None):
        super().__init__()
        cfg = deepcopy(default_config()['ArmAC_Args'] if network_config is None else network_config)
        if cfg['use_decoder']:
            raise ValueError('The default automatic path has no decoder implementation')
        if len(cfg['actor_hidden_dims']) != 3 or len(cfg['critic_hidden_dims']) != 3:
            raise ValueError('The upstream arm history encoders require three hidden dimensions')
        if num_obs_history <= num_obs:
            raise ValueError('Arm history must include frames before the current observation')
        self.decoder = False
        self.num_obs, self.num_obs_history = num_obs, num_obs_history
        self.num_privileged_obs = num_privileged_obs
        self.adaptation_module = _mlp(num_obs_history, cfg['adaptation_module_branch_hidden_dims'],
                                      num_privileged_obs, cfg['activation'])
        actor_dims, critic_dims = cfg['actor_hidden_dims'], cfg['critic_hidden_dims']
        self.actor_history_encoder = _mlp(num_obs_history - num_obs, actor_dims[:2],
                                          actor_dims[2], cfg['activation'])
        self.actor_body = _mlp(num_obs + num_privileged_obs + actor_dims[2], actor_dims,
                               num_actions, cfg['activation'])
        self.critic_history_encoder = _mlp(num_obs_history - num_obs, critic_dims[:2],
                                           critic_dims[2], cfg['activation'])
        self.critic_body = _mlp(num_obs + num_privileged_obs + critic_dims[2], critic_dims,
                                1, cfg['activation'])
        self.std = nn.Parameter(cfg['init_noise_std'] * torch.ones(num_actions))
        self.distribution = None
        if device is not None:
            self.to(device)

    def _body_output(self, observations, history):
        latent = self.adaptation_module(history)
        encoded = self.actor_history_encoder(history[..., :-self.num_obs])
        return self.actor_body(torch.cat((observations, latent, encoded), dim=-1)), latent

    def update_distribution(self, observation_history):
        raw, _ = self._body_output(observation_history[..., -self.num_obs:], observation_history)
        # Only the Gaussian mean's final two dimensions are squashed. Samples
        # remain unbounded, matching the actual automatic PPO collection path.
        mean = torch.cat((raw[..., :-2], torch.tanh(raw[..., -2:])), dim=-1)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act_inference(self, ob, policy_info=None, *, mode='official_play'):
        if mode not in ('official_play', 'training_mean'):
            raise ValueError(f'Unknown RoboDuet inference mode: {mode}')
        # The official loader reads obs separately; training uses history's
        # latest frame. These can differ when a caller has cleared history.
        current = ob['obs'] if mode == 'official_play' else ob['obs_history'][..., -self.num_obs:]
        raw, latent = self._body_output(current, ob['obs_history'])
        if policy_info is not None:
            policy_info['latent'] = latent
        if mode == 'training_mean':
            return torch.cat((raw[..., :-2], torch.tanh(raw[..., -2:])), dim=-1)
        return raw

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        obs = observation_history[..., -self.num_obs:]
        encoded = self.critic_history_encoder(observation_history[..., :-self.num_obs])
        return self.critic_body(torch.cat((obs, privileged_observations, encoded), dim=-1))


def build_models(config=None, device='cpu'):
    config = default_config() if config is None else config
    cfg = config['Cfg']
    dog, arm = cfg['dog'], cfg['arm']
    return (DogActorCritic(dog['dog_num_observations'], dog['dog_num_privileged_obs'],
                           dog['dog_num_obs_history'], dog['dog_actions'],
                           config['DogAC_Args'], device=device),
            ArmActorCritic(arm['arm_num_observations'], arm['arm_num_privileged_obs'],
                           arm['arm_num_obs_history'], arm['num_actions_arm_cd'],
                           config['ArmAC_Args'], device=device))


@torch.no_grad()
def coordinated_actions(dog, arm, env, *, stage2=True, mode='official_play'):
    """Read arm, write guidance, then read dog; return dog and six arm actions.

    The environment owns history and the subsequent physical step. Stage 1
    does not run the arm network or write guidance, as in the original runner.
    """
    if stage2:
        arm_action = arm.act_inference(env.get_arm_observations(), mode=mode)
        env.plan(arm_action[..., -2:])
        physical_arm = arm_action[..., :-2]
    else:
        physical_arm = torch.zeros((env.num_envs, 6), device=next(dog.parameters()).device)
    dog_action = dog.act_inference(env.get_dog_observations())
    return dog_action, physical_arm
