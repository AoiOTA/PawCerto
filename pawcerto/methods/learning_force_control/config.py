"""Resolved network dimensions from the fixed official scripts/train.py.

The paper's 17 effective controls are not this revision's 19-output policy ABI.
"""
from copy import deepcopy
from .actor_critic import AC_Args

SOURCE_REVISION = 'c760e1d74ad165d3c069d4f57ab5d066f6a41eb6'


def network_config(full_config):
    """Resolve the network actually consumed by the training and export paths."""
    env = full_config['env']['env']
    return {'num_obs': env['num_observations'],
            'num_privileged_obs': env['num_privileged_obs'],
            'num_obs_history': env['num_observations'] * env['num_observation_history'],
            'num_actions': env['num_actions'],
            'policy': deepcopy(full_config['policy'])}


def training_config():
    return dict(num_obs=87, num_privileged_obs=16, num_obs_history=870,
                num_actions=19, policy=dict(
                    init_noise_std=1.0, actor_hidden_dims=[512, 256, 128],
                    critic_hidden_dims=[512, 256, 128], activation='elu',
                    adaptation_module_branch_hidden_dims=[256, 128],
                    adaptation_labels=['motion_loss', 'dynamics_loss', 'force_loss',
                                       'friction_loss', 'gripper_pos_loss', 'gripper_target_pos_loss'],
                    adaptation_dims=[3, 3, 3, 1, 3, 3],
                    adaptation_weights=[1, 1, 0.05, 1, 10, 1], use_decoder=False))


def build_model(config):
    from .actor_critic import ActorCritic
    args = AC_Args()
    for key, value in config['policy'].items():
        if not hasattr(args, key):
            raise ValueError(f'Unknown policy setting: {key}')
        setattr(args, key, value)
    if sum(args.adaptation_dims) != config['num_privileged_obs']:
        raise ValueError('Adaptation dimensions must match privileged observations')
    if args.use_decoder:
        raise ValueError('Official network does not implement a decoder')
    return ActorCritic(**{k: config[k] for k in ('num_obs', 'num_privileged_obs', 'num_obs_history', 'num_actions')}, config=args)
