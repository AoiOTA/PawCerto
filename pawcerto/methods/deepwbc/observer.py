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


"""DeepWBC observation assembly and target-frame conventions; see LICENSE.

Physics caller supplies signed roll/pitch and original six-axis foot sensor
wrenches. A contact-force-only signal is not equivalent without validation.
"""
import torch
from .config import default_config
from .controller import POLICY_DOF_NAMES, wrap_to_pi


def sphere_to_cartesian(lpy):
    """Explicit reconstruction: pitch positive downward, yaw positive left.

    Upstream imports sphere2cart from an unpublished IsaacGym torch_utils
    extension; this equation is not claimed numerically verified against it.
    """
    length, pitch, yaw = lpy.unbind(-1)
    return torch.stack((length * torch.cos(pitch) * torch.cos(yaw),
                        length * torch.cos(pitch) * torch.sin(yaw),
                        -length * torch.sin(pitch)), dim=-1)


def cartesian_to_sphere(xyz):
    x, y, z = xyz.unbind(-1)
    return torch.stack((xyz.norm(dim=-1), torch.atan2(-z, torch.sqrt(x*x + y*y)),
                        torch.atan2(y, x)), dim=-1)


def goal_world(lpy, base_xy, base_yaw):
    """Yaw rotation + [base_x, base_y, 0.53], invariant to actual base z/r/p."""
    local = sphere_to_cartesian(lpy)
    x, y, z = local.unbind(-1)
    c, s = torch.cos(base_yaw), torch.sin(base_yaw)
    return torch.stack((c*x-s*y+base_xy[:, 0], s*x+c*y+base_xy[:, 1], z+0.53), dim=-1)


class DeepWBCObserver:
    def __init__(self, num_envs, dof_names, config=None, device='cpu'):
        self.config = default_config() if config is None else config
        if set(dof_names) != set(POLICY_DOF_NAMES) or len(dof_names) != 20:
            raise ValueError('Expected all 20 named Go1/WidowX DOFs')
        if tuple(dof_names[-2:]) != POLICY_DOF_NAMES[-2:]:
            raise ValueError('Upstream observation requires fingers last')
        self.reorder = [dof_names.index(n) for n in POLICY_DOF_NAMES]
        self.default = torch.tensor([self.config['init_state']['default_joint_angles'][n] for n in dof_names], device=device)
        env = self.config['env']
        if not self.config['domain_rand']['observe_priv']:
            raise ValueError('Public path requires privileged observation slots')
        self.history = torch.zeros(num_envs, env['history_len'], env['num_proprio'], device=device)

    def reset(self, env_ids):
        self.history[env_ids] = 0

    def observe(self, *, roll_pitch, base_ang_vel, dof_pos, dof_vel,
                latest_sim_actions, foot_wrenches, commands, ee_goal_lpy,
                ee_delta_orn, mass_params, friction, motor_strength, episode_length):
        """Return 860 columns before advancing history (same reset semantics).

        foot_wrenches: [N,4,6] in FL,FR,RL,RR order; mass_params: [N,5]
        (base added mass, added COM xyz, gripper added mass); friction [N,1].
        Motor strength [N,18] stays simulator ordered in privileged slots.
        """
        if foot_wrenches.shape[1:] != (4, 6):
            raise ValueError('DeepWBC uses the norm of all six sensor channels')
        scales = self.config['normalization']['obs_scales']
        q = dof_pos.clone()
        q[:, -8] = wrap_to_pi(q[:, -8])
        contacts = (foot_wrenches.norm(dim=-1) > 1.5)[:, [1, 0, 3, 2]]
        target = ee_goal_lpy if self.config['goal_ee']['command_mode'] == 'sphere' else sphere_to_cartesian(ee_goal_lpy)
        proprio = torch.cat((roll_pitch, base_ang_vel * scales['ang_vel'],
            ((q-self.default) * scales['dof_pos'])[:, self.reorder],
            (dof_vel * scales['dof_vel'])[:, self.reorder],
            latest_sim_actions[:, self.reorder[:18]], contacts,
            commands * commands.new_tensor([scales['lin_vel'], scales['lin_vel'], scales['ang_vel']]),
            target, ee_delta_orn), dim=-1)
        priv = torch.cat((mass_params, friction, motor_strength-1), dim=-1)
        if proprio.shape[-1] != 76 or priv.shape[-1] != 24:
            raise ValueError('Expected 76 proprioceptive and 24 privileged columns')
        result = torch.cat((proprio, priv, self.history.flatten(1)), dim=-1)
        self.history = torch.where((episode_length <= 1)[:, None, None],
            proprio[:, None].expand_as(self.history),
            torch.cat((self.history[:, 1:], proprio[:, None]), dim=1))
        clip = self.config['normalization']['clip_observations']
        return result.clamp(-clip, clip)
