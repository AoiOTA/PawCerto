"""UniFP history buffers, with zeroed reset slots and oldest-to-newest order."""
from collections import deque
import torch


def initialize_history(state):
    """Allocate the two buffers consumed by UniFPEquations.compute_observations."""
    cfg = state.cfg.env
    state.obs_history = deque((torch.zeros(state.num_envs, cfg.num_single_obs,
                                          device=state.device)
                               for _ in range(cfg.frame_stack)), maxlen=cfg.frame_stack)
    state.critic_history = deque((torch.zeros(state.num_envs, cfg.single_num_privileged_obs,
                                             device=state.device)
                                  for _ in range(cfg.c_frame_stack)), maxlen=cfg.c_frame_stack)


def reset_history(state, env_ids):
    """Run before the post-reset observation, as in upstream reset_idx."""
    for frame in state.obs_history:
        frame[env_ids] *= 0
    for frame in state.critic_history:
        frame[env_ids] *= 0


def force_stage_active(global_steps, cfg):
    """Strict upstream threshold in policy steps, evaluated before each substep."""
    return global_steps > cfg.commands.force_start_step * 24
