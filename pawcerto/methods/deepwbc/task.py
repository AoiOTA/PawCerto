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


"""Original DeepWBC task equations with an explicit external-physics boundary.

Copied methods retain upstream behavior; quaternion and LPY helpers are explicit
local reconstructions because the upstream IsaacGym helper extension is absent.
"""
from types import SimpleNamespace
from copy import deepcopy
import math
import numpy as np
import torch
from .config import default_config
from .controller import wrap_to_pi as torch_wrap_to_pi_minuspi
from .observer import sphere_to_cartesian as sphere2cart, cartesian_to_sphere as cart2sphere


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value


def torch_rand_float(lower, upper, shape, device):
    return (upper-lower)*torch.rand(*shape, device=device)+lower


def euler_from_quat(q):
    x, y, z, w = q.unbind(-1)
    return (torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            torch.asin((2*(w*y-z*x)).clamp(-1, 1)),
            torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def quat_apply(q, v):
    uv = torch.cross(q[..., :3], v, dim=-1)
    return v + 2*(q[..., 3:]*uv + torch.cross(q[..., :3], uv, dim=-1))


def quat_rotate_inverse(q, v):
    return quat_apply(torch.cat((-q[..., :3], q[..., 3:]), dim=-1), v)


def evaluation_config(config=None):
    """Relevant original play.py overrides; randomization remains enabled."""
    c = deepcopy(default_config() if config is None else config)
    c['termination'].update(r_threshold=1., p_threshold=1., z_threshold=0.)
    for key in ('randomize_friction', 'randomize_base_mass', 'randomize_base_com', 'randomize_motor', 'push_robots'):
        c['domain_rand'][key] = True
    for key in ('lin_vel_x_schedule', 'ang_vel_yaw_schedule', 'tracking_ang_vel_yaw_schedule'):
        c['commands'][key] = [0, 1]
    for key in ('l_schedule', 'p_schedule', 'y_schedule', 'tracking_ee_reward_schedule'):
        c['goal_ee'][key] = [0, 1]
    c['goal_ee']['underground_limit'] = -0.57
    return c


class DeepWBCTask:
    """Tensor lifecycle; does not simulate, advance or reset a physics engine.

    bind_state accepts current tensors in simulator DOF order (legs, six arm,
    two fingers), xyzw quaternion order, SI units. after_physics computes the
    terminal transition BEFORE reset. reset returns state writes for the caller;
    controller/observer reset and fresh simulator state remain caller duties.
    """
    def __init__(self, num_envs, config=None, device='cpu'):
        self.config = deepcopy(default_config() if config is None else config)
        self.cfg = namespace(self.config)
        self.device, self.num_envs = device, num_envs
        self.dt = self.cfg.sim.dt*self.cfg.control.decimation
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = math.ceil(self.max_episode_length_s/self.dt)
        self.push_interval = math.ceil(self.cfg.domain_rand.push_interval_s/self.dt)
        self.reward_scales = dict(self.config['rewards']['scales'])
        self.arm_reward_scales = dict(self.config['rewards']['arm_scales'])
        self.update_counter = self.common_step_counter = 0
        def zeros(*shape): return torch.zeros(num_envs, *shape, device=device)
        self.commands = zeros(3)
        self.base_yaw_euler = zeros(3)
        self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.time_out_buf = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.reset_buf = self.time_out_buf.clone()
        self.rew_buf, self.arm_rew_buf = zeros(), zeros()
        self.last_actions, self.last_dof_vel = zeros(18), zeros(20)
        self.feet_air_time = zeros(4)
        self.last_contacts = torch.zeros(num_envs, 4, device=device, dtype=torch.bool)
        for name in ('ee_start_sphere','ee_goal_sphere','ee_goal_cart','ee_goal_delta_orn_euler','ee_goal_orn_euler','curr_ee_goal_sphere','curr_ee_goal_cart'):
            setattr(self, name, zeros(3))
        self.curr_ee_goal = self.curr_ee_goal_sphere if self.cfg.goal_ee.command_mode == 'sphere' else self.curr_ee_goal_cart
        self.goal_timer = zeros()
        self.traj_timesteps = torch_rand_float(*self.cfg.goal_ee.traj_time, (num_envs,), device)/self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(*self.cfg.goal_ee.hold_time, (num_envs,), device)/self.dt
        for name in ('collision_lower_limits','collision_upper_limits','sphere_error_scale','orn_error_scale'):
            setattr(self, name, torch.tensor(self.config['goal_ee'][name], device=device))
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=device)[None,None,:]
        self.z_invariant_offset = torch.full((num_envs,1), .53, device=device)
        for stem in ('lin_vel_x','ang_vel_yaw'):
            for prefix in ('init_', 'final_'):
                setattr(self, prefix+stem+'_ranges', np.array(self.config['commands']['ranges'][prefix+stem]))
            setattr(self, stem+'_ranges', getattr(self, 'init_'+stem+'_ranges'))
            setattr(self, stem+'_schedule', self.config['commands'][stem+'_schedule'])
        for axis in 'lpy':
            for prefix in ('init_', 'final_'):
                setattr(self, prefix+'goal_ee_'+axis+'_ranges', np.array(self.config['goal_ee']['ranges'][prefix+'pos_'+axis]))
            setattr(self, 'goal_ee_'+axis+'_ranges', getattr(self, 'init_goal_ee_'+axis+'_ranges'))
            setattr(self, 'goal_ee_'+axis+'_schedule', self.config['goal_ee'][axis+'_schedule'])
        self.goal_ee_delta_orn_ranges = torch.tensor(self.cfg.goal_ee.ranges.final_delta_orn, device=device)
        self.tracking_ang_vel_yaw_schedule = self.cfg.commands.tracking_ang_vel_yaw_schedule
        self.final_tracking_ang_vel_yaw_exp = self.cfg.commands.ranges.final_tracking_ang_vel_yaw_exp
        self.tracking_ee_reward_schedule = self.cfg.goal_ee.tracking_ee_reward_schedule
        self.final_tracking_ee_reward = self.cfg.goal_ee.ranges.final_tracking_ee_reward
        self._prepare_reward_function()

    def bind_state(self, *, root_states, dof_pos, dof_vel, torques, actions,
                   ee_pos, ee_orn, contact_forces, foot_wrenches,
                   termination_contact_indices=(), **optional_reward_inputs):
        expected = {'root_states': (self.num_envs,13), 'dof_pos': (self.num_envs,20),
                    'dof_vel': (self.num_envs,20), 'torques': (self.num_envs,20),
                    'actions': (self.num_envs,18), 'ee_pos': (self.num_envs,3),
                    'ee_orn': (self.num_envs,4), 'foot_wrenches': (self.num_envs,4,6)}
        values = locals()
        for key, shape in expected.items():
            if values[key].shape != shape:
                raise ValueError(f'{key}: expected {shape}, got {values[key].shape}')
        for key in ('root_states','dof_pos','dof_vel','torques','actions','ee_pos','ee_orn','contact_forces'):
            setattr(self, key, values[key])
        self.force_sensor_tensor = foot_wrenches
        self.termination_contact_indices = list(termination_contact_indices)
        self.base_quat = root_states[:,3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, root_states[:,7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, root_states[:,10:13])
        yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler[:,2] = yaw
        self.base_yaw_quat = torch.stack((yaw*0, yaw*0, torch.sin(yaw/2), torch.cos(yaw/2)), -1)
        for key, value in optional_reward_inputs.items():
            if key not in ('projected_gravity','measured_heights','penalized_contact_indices','dof_pos_limits','dof_vel_limits','torque_limits','feet_indices','default_dof_pos'):
                raise TypeError(f'Unknown optional reward input: {key}')
            setattr(self, key, value)

    def after_physics(self):
        """Return leg reward, arm reward, done, extras before physics reset.

        If push_velocity_xy is present the caller must write root linear XY
        velocity now (original callback timing, after body velocity refresh).
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.update_curr_ee_goal()
        ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time/self.dt) == 0).nonzero().flatten()
        self._resample_commands(ids)
        extras = {}
        if self.cfg.domain_rand.push_robots and self.common_step_counter % self.push_interval == 0:
            v = torch_rand_float(-self.cfg.domain_rand.max_push_vel_xy, self.cfg.domain_rand.max_push_vel_xy, (self.num_envs,2), self.device)
            extras['push_velocity_xy'] = torch.where((self.commands.sum(-1)==0)[:,None], v*2.5, v)
        self.check_termination()
        self.compute_reward()
        if self.cfg.env.send_timeouts:
            extras['time_outs'] = self.time_out_buf.clone()
        return self.rew_buf.clone(), self.arm_rew_buf.clone(), self.reset_buf.clone(), extras

    def finish_step(self):
        """Call after reset writes and fresh simulator binding, as upstream."""
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

    def reset(self, env_ids, *, default_dof_pos, env_origins, start=False):
        """Return root/dof writes; caller must apply them to real physics.

        Root/DOF generation mirrors upstream. Goals start at previous target,
        and commands resample only for timeouts unless start=True.
        """
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(ids)
        if n == 0:
            return {'env_ids': ids, 'root_states': torch.empty(0,13,device=self.device),
                    'dof_pos': torch.empty(0,20,device=self.device),
                    'dof_vel': torch.empty(0,20,device=self.device), 'episode': {}}
        init = self.cfg.init_state
        root = torch.tensor(init.pos+init.rot+init.lin_vel+init.ang_vel, device=self.device).repeat(n,1)
        root[:,:3] += env_origins[ids]
        r = self.cfg.terrain.origin_perturb_range
        root[:,:2] += torch_rand_float(-r,r,(n,2),self.device)
        r = self.cfg.terrain.init_vel_perturb_range
        root[:,7:13] = torch_rand_float(-r,r,(n,6),self.device)
        dof_pos = default_dof_pos*torch_rand_float(.8,1.2,(n,20),self.device)
        self._resample_commands(ids if start else self.time_out_buf.nonzero().flatten())
        self._resample_ee_goal(ids, is_init=True)
        self.last_actions[ids] = 0
        self.last_dof_vel[ids] = 0
        self.feet_air_time[ids] = 0
        self.episode_length_buf[ids] = 0
        self.reset_buf[ids] = True
        self.goal_timer[ids] = 0
        episode = {}
        for prefix, sums in (('rew_',self.episode_sums),('metric_',self.episode_metric_sums)):
            for name, value in sums.items():
                episode[prefix+name] = value[ids].mean()/self.max_episode_length_s
                value[ids] = 0
        return {'env_ids': ids, 'root_states': root, 'dof_pos': dof_pos,
                'dof_vel': torch.zeros_like(dof_pos), 'episode': episode}

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            # else:
            #     self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.arm_reward_scales.keys()):
            scale = self.arm_reward_scales[key]
            if scale==0:
                self.arm_reward_scales.pop(key) 
            # else:
            #     self.arm_reward_scales[key] *= self.dt
        # prepare list of functions
        self.arm_reward_functions = []
        self.arm_reward_names = []
        for name, scale in self.arm_reward_scales.items():
            if name=="termination":
                continue
            self.arm_reward_names.append(name)
            name = '_reward_' + name
            self.arm_reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in list(self.reward_scales.keys()) + list(self.arm_reward_scales.keys())}

        self.metric_names = ['leg_energy_abs_sum', 'tracking_lin_vel_x_l1', 'tracking_ang_vel_yaw_exp', 'tracking_ee_cart', 'tracking_ee_sphere', 'tracking_ee_orn', 'leg_action_l2', 'torque', 'energy_square', 'foot_contacts_z']
        self.episode_metric_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) \
            for name in self.metric_names}

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
        self.rew_buf /= 100

        self.arm_rew_buf[:] = 0.
        for i in range(len(self.arm_reward_functions)):
            name = self.arm_reward_names[i]
            rew = self.arm_reward_functions[i]() * self.arm_reward_scales[name]
            self.arm_rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.arm_rew_buf[:] = torch.clip(self.arm_rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.arm_reward_scales:
            rew = self._reward_termination() * self.arm_reward_scales["termination"]
            self.arm_rew_buf += rew
            self.episode_sums["termination"] += rew
        
        self.arm_rew_buf /= 100

    def _get_curriculum_value(self, schedule, init_range, final_range, counter):
        return np.clip((counter - schedule[0]) / (schedule[1] - schedule[0]), 0, 1) * (final_range - init_range) + init_range

    def update_command_curriculum(self):
        self.update_counter += 1

        self.lin_vel_x_ranges = self._get_curriculum_value(self.lin_vel_x_schedule, self.init_lin_vel_x_ranges, self.final_lin_vel_x_ranges, self.update_counter)
        self.ang_vel_yaw_ranges = self._get_curriculum_value(self.ang_vel_yaw_schedule, self.init_ang_vel_yaw_ranges, self.final_ang_vel_yaw_ranges, self.update_counter)
        self.reward_scales['tracking_ang_vel_yaw_exp'] = self._get_curriculum_value(self.tracking_ang_vel_yaw_schedule, 0, self.final_tracking_ang_vel_yaw_exp, self.update_counter)

        self.goal_ee_l_ranges = self._get_curriculum_value(self.goal_ee_l_schedule, self.init_goal_ee_l_ranges, self.final_goal_ee_l_ranges, self.update_counter)
        self.goal_ee_p_ranges = self._get_curriculum_value(self.goal_ee_p_schedule, self.init_goal_ee_p_ranges, self.final_goal_ee_p_ranges, self.update_counter)
        self.goal_ee_y_ranges = self._get_curriculum_value(self.goal_ee_y_schedule, self.init_goal_ee_y_ranges, self.final_goal_ee_y_ranges, self.update_counter)
        # self.action_scale[-6:] = self._get_curriculum_value(self.arm_action_scale_schedule, 0, self.final_arm_action_scale, self.update_counter)
        if 'tracking_ee_sphere' in self.arm_reward_scales:
            self.arm_reward_scales['tracking_ee_sphere'] = self._get_curriculum_value(self.tracking_ee_reward_schedule, 0, self.final_tracking_ee_reward, self.update_counter)
        else:
            self.arm_reward_scales['tracking_ee_cart'] = self._get_curriculum_value(self.tracking_ee_reward_schedule, 0, self.final_tracking_ee_reward, self.update_counter)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.lin_vel_x_ranges[0], self.lin_vel_x_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = 0
        self.commands[env_ids, 2] = torch_rand_float(self.ang_vel_yaw_ranges[0], self.ang_vel_yaw_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :] *= (torch.logical_or(self.commands[env_ids, 0] > self.cfg.commands.lin_vel_x_clip, \
            torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip)).unsqueeze(1)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        r, p, _ = euler_from_quat(self.base_quat) 
        z = self.root_states[:, 2]

        r_threshold_buff = ((r > 0.2) & (self.curr_ee_goal[:, 2] >= 0)) | ((r < -0.2) & (self.curr_ee_goal[:, 2] <= 0))
        p_threshold_buff = ((p > 0.2) & (self.curr_ee_goal[:, 1] >= 0)) | ((p < -0.2) & (self.curr_ee_goal[:, 1] <= 0))
        z_threshold_buff = z < self.cfg.termination.z_threshold
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        # print(self.base_quat)
        # print(torch.stack([self.reset_buf, r_threshold_buff, p_threshold_buff, z_threshold_buff], dim=-1)[0])
        # print('r: ', r[0].item())
        # print('p: ', p[0].item())
        # print('z: ', z[0].item())
        # print('-----------------------------------------------------')
        # time.sleep(0.5)

        # self.reset_triggers = torch.stack([termination_contact_buf, r_threshold_buff, p_threshold_buff, z_threshold_buff, self.time_out_buf], dim=-1).nonzero(as_tuple=False)
        # if len(self.reset_triggers) > 0:
        #     print('reset_triggers: ', self.reset_triggers)

        self.reset_buf = termination_contact_buf | r_threshold_buff | p_threshold_buff | z_threshold_buff | self.time_out_buf

    def _get_init_start_ee_sphere(self):
        init_start_ee_cart = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        init_start_ee_cart[:, 0] = 0.15
        init_start_ee_cart[:, 2] = 0.15
        self.init_start_ee_sphere = cart2sphere(init_start_ee_cart)

    def _resample_ee_goal_sphere_once(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_l_ranges[0], self.goal_ee_l_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_p_ranges[0], self.goal_ee_p_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_y_ranges[0], self.goal_ee_y_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_delta_orn_ranges[0, 0], self.goal_ee_delta_orn_ranges[0, 1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_delta_orn_ranges[1, 0], self.goal_ee_delta_orn_ranges[1, 1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_delta_orn_ranges[2, 0], self.goal_ee_delta_orn_ranges[2, 1], (len(env_ids), 1), device=self.device)
        self.ee_goal_delta_orn_euler[env_ids] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)
        self.ee_goal_orn_euler[env_ids] = torch_wrap_to_pi_minuspi(self.ee_goal_delta_orn_euler[env_ids] + self.base_yaw_euler[env_ids])

    def _resample_ee_goal(self, env_ids, is_init=False):
        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()
            self._resample_ee_goal_orn_once(env_ids)
            # if is_init:
            #     self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[env_ids].clone()
            #     self._resample_ee_goal_sphere_once(env_ids, self.cfg.goal_ee.init_ranges)
            # else:
            self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
            for i in range(10):
                self._resample_ee_goal_sphere_once(env_ids)
                collision_mask = self.collision_check(env_ids)
                env_ids = env_ids[collision_mask]
                if len(env_ids) == 0:
                    break
            self.ee_goal_cart[init_env_ids, :] = sphere2cart(self.ee_goal_sphere[init_env_ids, :])
            self.goal_timer[init_env_ids] = 0.0

    def collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ...,  None], self.collision_check_t).squeeze(-1)
        ee_target_cart = sphere2cart(torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1), torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def update_curr_ee_goal(self):
        t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
        self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
        self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        self._resample_ee_goal(resample_id)

    def _reward_tracking_ee_sphere(self):
        ee_pos_local = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1))

        ee_pos_error = torch.sum(torch.abs(cart2sphere(ee_pos_local) - self.curr_ee_goal_sphere) * self.sphere_error_scale, dim=1)

        self.episode_metric_sums['tracking_ee_sphere'] += ee_pos_error
        return torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma)

    def _reward_tracking_ee_cart(self):
        target_ee = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1) + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

        ee_pos_error = torch.sum(torch.abs(self.ee_pos - target_ee), dim=1)

        self.episode_metric_sums['tracking_ee_cart'] += ee_pos_error
        return torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma)

    def _reward_tracking_ee_orn(self):
        # ee_orn_normalized = self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1)
        # ee_goal_orn_normalized = self.ee_goal_orn_quat / torch.norm(self.ee_goal_orn_quat, dim=-1).unsqueeze(-1)
        # orn_err = orientation_error(ee_goal_orn_normalized, ee_orn_normalized)
        ee_orn_euler = torch.stack(euler_from_quat(self.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs(torch_wrap_to_pi_minuspi(self.ee_goal_orn_euler - ee_orn_euler)) * self.orn_error_scale, dim=1)

        # self.episode_metric_sums['tracking_ee_orn'] += orn_err

        return torch.exp(-orn_err/self.cfg.rewards.tracking_ee_sigma)

    def _reward_hip_action_l2(self):
        action_l2 = torch.sum(self.actions[:, [0, 3, 6, 9]] ** 2, dim=1)
        self.episode_metric_sums['leg_action_l2'] += action_l2
        return action_l2

    def _reward_tracking_ee_orn_ry(self):
        # ee_orn_normalized = self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1)
        # ee_goal_orn_normalized = self.ee_goal_orn_quat / torch.norm(self.ee_goal_orn_quat, dim=-1).unsqueeze(-1)
        # orn_err = orientation_error(ee_goal_orn_normalized, ee_orn_normalized)
        ee_orn_euler = torch.stack(euler_from_quat(self.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs((torch_wrap_to_pi_minuspi(self.ee_goal_orn_euler - ee_orn_euler) * self.orn_error_scale)[:, [0, 2]]), dim=1)

        self.episode_metric_sums['tracking_ee_orn'] += orn_err

        return torch.exp(-orn_err/self.cfg.rewards.tracking_ee_sigma)

    def _reward_leg_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.torques[:, :12] * self.dof_vel[:, :12]), dim = 1)
        self.episode_metric_sums['leg_energy_abs_sum'] += energy
        return energy

    def _reward_leg_energy_sum_abs(self):
        energy = torch.abs(torch.sum(self.torques[:, :12] * self.dof_vel[:, :12], dim = 1))
        return energy

    def _reward_leg_action_l2(self):
        action_l2 = torch.sum(self.actions[:, :12] ** 2, dim=1)
        self.episode_metric_sums['leg_action_l2'] += action_l2
        return action_l2

    def _reward_leg_energy(self):
        energy = torch.sum(self.torques[:, :12] * self.dof_vel[:, :12], dim = 1)
        return energy

    def _reward_arm_energy_abs_sum(self):
        return torch.sum(torch.abs(self.torques[:, 12:-2] * self.dof_vel[:, 12:-2]), dim = 1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_x_l1(self):
        error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        self.episode_metric_sums['tracking_lin_vel_x_l1'] += error
        return - error + torch.abs(self.commands[:, 0])

    def _reward_tracking_lin_vel_x_exp(self):
        error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        self.episode_metric_sums['tracking_lin_vel_x_l1'] += error
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel_yaw_l1(self):
        error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return - error + torch.abs(self.commands[:, 2])

    def _reward_tracking_ang_vel_yaw_exp(self):
        error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        self.episode_metric_sums['tracking_ang_vel_yaw_exp'] += error
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y_l2(self):
        return (self.commands[:, 1] - self.base_lin_vel[:, 1]) ** 2

    def _reward_tracking_lin_vel_z_l2(self):
        return (self.commands[:, 2] - self.base_lin_vel[:, 2]) ** 2

    def _reward_survive(self):
        return torch.ones(self.num_envs, device=self.device)

    def _reward_foot_contacts_z(self):
        foot_contacts_z = torch.square(self.force_sensor_tensor[:, :, 2]).sum(dim=-1)
        self.episode_metric_sums['foot_contacts_z'] += foot_contacts_z
        return foot_contacts_z

    def _reward_torques(self):
        # Penalize torques
        torque = torch.sum(torch.square(self.torques), dim=1)
        self.episode_metric_sums['torque'] += torque
        return torque

    def _reward_energy_square(self):
        energy = torch.sum(torch.square(self.torques[:, :12] * self.dof_vel[:, :12]), dim=1)
        self.episode_metric_sums['energy_square'] += energy
        return energy

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def sample_creation_randomization(self):
        """Sample original distributions once; simulator must apply these.

        RNG stream equivalence is not claimed (upstream mixes NumPy/Torch).
        Negative sampled friction is deliberately not silently clamped.
        """
        cfg, n = self.cfg.domain_rand, self.num_envs
        def sample(bounds, width=1):
            return torch_rand_float(*bounds, (n,width), self.device)
        mass = torch.zeros(n,5,device=self.device)
        if cfg.randomize_base_mass:
            mass[:,:1] = sample(cfg.added_mass_range)
        if cfg.randomize_base_com:
            for i, axis in enumerate('xyz',1):
                mass[:,i:i+1] = sample(getattr(cfg,'added_com_range_'+axis))
        if cfg.randomize_gripper_mass:
            mass[:,4:] = sample(cfg.gripper_added_mass_range)
        if cfg.randomize_friction:
            ids = torch.randint(0,1000,(n,1))
            buckets = torch_rand_float(*cfg.friction_range,(1000,1),'cpu')
            friction = buckets[ids].reshape(n,1).to(self.device)
        else:
            friction = torch.ones(n,1,device=self.device)
        motor = torch.cat((sample(cfg.leg_motor_strength_range,12),sample(cfg.arm_motor_strength_range,6)),1) if cfg.randomize_motor else torch.ones(n,18,device=self.device)
        box_mass = sample(self.cfg.box.added_mass_range) if self.cfg.box.randomize_base_mass else torch.zeros(n,1,device=self.device)
        return {'mass_params': mass, 'friction': friction, 'motor_strength': motor,
                'box_added_mass': box_mass,
                'box_delta_y': sample(self.cfg.box.box_env_origins_y_range) *
                               (2*torch.randint(0,2,(n,1),device=self.device)-1)}

    def reset_box_position(self, root_states, box_delta_y):
        """Nominal box XYZ; episode reset consumes XY only, retaining physical Z.

        Z is the creation height, not a request to overwrite the live box Z.
        """
        return torch.stack((torch.full_like(root_states[:,0],self.cfg.box.box_env_origins_x),
                            root_states[:,1]+box_delta_y.reshape(-1),
                            torch.full_like(root_states[:,0],self.cfg.box.box_env_origins_z)),1)
