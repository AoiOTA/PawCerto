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


"""DeepWBC public position PD path (BSD-3-Clause; see LICENSE).

Inputs/outputs use simulator DOF names, so physical ordering must be verified
by the environment. The upstream q18[-8] wrapping quirk is preserved.
"""
import torch
from .config import default_config

POLICY_DOF_NAMES = tuple(f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL')
                         for joint in ('hip', 'thigh', 'calf')) + (
    'widow_waist', 'widow_shoulder', 'widow_elbow', 'widow_forearm_roll',
    'widow_wrist_angle', 'widow_wrist_rotate', 'widow_left_finger', 'widow_right_finger')


def wrap_to_pi(value):
    """Signed principal angle, with +pi retained (upstream helper unavailable)."""
    value = torch.remainder(value, 2 * torch.pi)
    return value - 2 * torch.pi * (value > torch.pi)


class DeepWBCController:
    def __init__(self, num_envs, dof_names, torque_limits, config=None, device='cpu'):
        self.config = default_config() if config is None else config
        if set(dof_names) != set(POLICY_DOF_NAMES) or len(dof_names) != 20:
            raise ValueError('Expected all 20 named Go1/WidowX DOFs')
        if tuple(dof_names[-2:]) != POLICY_DOF_NAMES[-2:]:
            raise ValueError('Upstream control requires fingers last')
        self.dof_names = tuple(dof_names)
        self.policy_to_sim = [POLICY_DOF_NAMES[:18].index(n) for n in dof_names[:18]]
        ctl = self.config['control']
        if ctl['adaptive_arm_gains'] or ctl['torque_supervision']:
            raise ValueError('Core controller implements public fixed-gain unsupervised torque path')
        tensor = lambda value: torch.tensor(value, device=device, dtype=torch.float32)
        self.default = tensor([self.config['init_state']['default_joint_angles'][n] for n in dof_names])
        self.kp = tensor([ctl['stiffness']['widow' if 'widow' in n else 'joint'] for n in dof_names[:18]])
        self.kd = tensor([ctl['damping']['widow' if 'widow' in n else 'joint'] for n in dof_names[:18]])
        # Upstream applies these scales in simulator order, after action reordering.
        self.scale = tensor(ctl['action_scale'])
        self.limits = torch.as_tensor(torque_limits, device=device, dtype=torch.float32)
        if self.limits.shape != (20,) or torch.any(self.limits < 0):
            raise ValueError('Expected 20 nonnegative simulator effort limits')
        self.delay = self.config['env']['action_delay']
        self.history = torch.zeros(num_envs, self.delay + 2, 18, device=device)
        self.actions = torch.zeros(num_envs, 18, device=device)

    def reset(self, env_ids):
        self.history[env_ids] = 0
        self.actions[env_ids] = 0

    def push(self, policy_actions):
        """Once per policy step; newest history is undelayed simulator action."""
        actions = policy_actions[:, self.policy_to_sim].clamp(
            -self.config['normalization']['clip_actions'], self.config['normalization']['clip_actions'])
        if self.delay != -1:
            self.history = torch.cat((self.history[:, 1:], actions[:, None]), dim=1)
            actions = self.history[:, -self.delay - 1]
        self.actions = actions.clone()
        return self.actions

    def torques(self, dof_pos, dof_vel, motor_strength):
        """Once per physics substep; motor strength scales target displacement."""
        q = dof_pos[:, :18].clone()
        q[:, -8] = wrap_to_pi(q[:, -8])  # exact index from upstream 18-DOF torque path
        tau = self.kp * (self.actions * motor_strength * self.scale + self.default[:18] - q)
        tau -= self.kd * dof_vel[:, :18]
        tau = torch.cat((tau, tau.new_zeros(tau.shape[0], 2)), dim=-1)
        return torch.maximum(torch.minimum(tau, self.limits), -self.limits)
