# Derived from b1_gym/envs/wrappers/history_wrapper.py at fixed source revision.
# Copyright (c) 2022 MIT Improbable AI Lab. See LICENSE.
"""Tensor history portion of the upstream wrapper, without gym or simulator.

append matches step/get_observations ordering; reset_rows matches reset_idx.
This is a buffer only, not a synthetic environment.
"""
import torch


class ObservationHistory:
    def __init__(self, num_envs, num_obs=87, history_length=10, frame_skip=1, device='cpu'):
        if min(num_envs, num_obs, history_length, frame_skip) <= 0:
            raise ValueError('History dimensions must be positive')
        self.frame_skip = frame_skip
        self.buffer = torch.zeros(num_envs, history_length * frame_skip, num_obs, device=device)
        self.history = torch.zeros(num_envs, history_length * num_obs, device=device)

    def append(self, obs, reset_ids=None):
        self.buffer = torch.cat((self.buffer[:, 1:, :], obs.unsqueeze(1)), dim=1)
        self.history = self.buffer[:, self.frame_skip-1::self.frame_skip, :].reshape(obs.shape[0], -1)
        if reset_ids is not None:
            self.reset_rows(reset_ids)
        return self.history

    def reset_rows(self, env_ids):
        self.buffer[env_ids] = 0
        self.history[env_ids] = 0
