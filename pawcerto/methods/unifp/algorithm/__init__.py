"""UniFP original feed-forward actor, concurrent estimator, PPO, and runner.

Environment protocol for OnPolicyRunner: integer num_envs, num_obs (history),
num_privileged_obs, num_pred_obs (12), num_single_obs and num_actions;
reset() returns a two-tuple; get_observations() returns a dict containing obs,
privileged_obs and obs_pred tensors; step(actions) returns that dict, rewards,
dones, infos. Rewards/dones/time_outs are vectors of num_envs. History is ordered
oldest to newest, current observation last. Optional random episode initialization
uses episode_length_buf and max_episode_length. The environment supplies original
observations, force labels, rewards and timing; this module supplies no simulator.

log_dir=None requires only torch/numpy and performs no automatic checkpoint I/O.
A log_dir explicitly enables TensorBoard (optional dependency); W&B is not used.
Checkpoints count completed updates and include both Adam states and adaptive LR.
Environment/RNG state is not checkpointed, so load is optimizer continuation,
not bitwise trajectory replay. Legacy upstream model/optimizer keys are accepted;
missing adaptation optimizer state is reported with a warning.

Equations and state_dict names retain the pinned UniFP reference. Integration
repairs: plain AC_Args instead of params_proto; internal imports; scoped Normal
validation; eval() in test_mode; non-mutating storage statistics; explicit class
lookup; optional logging/checkpoint paths; completed iteration and both optimizers
on resume; logging tolerates no completed episodes. Original adaptation loss scales labels before MSE (squared weights),
uses two Adam optimizers over all parameters with gradients determining which
parameters update, and divides reported adaptation loss by adaptation_batch_size.
"""
from .actor_critic import AC_Args, ActorCritic
from .ppo import Adaptation_Args, PPO
from .rollout_storage import RolloutStorage
from .on_policy_runner import OnPolicyRunner

__all__ = ["AC_Args", "ActorCritic", "Adaptation_Args", "PPO", "RolloutStorage", "OnPolicyRunner"]
