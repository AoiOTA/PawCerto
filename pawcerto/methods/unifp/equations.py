"""UniFP tensor equations extracted from pinned B2/Z1 environment.

Method bodies are upstream verbatim. A runtime supplies measured state and buffers;
this mixin does not create a simulator. See docs/unifp-implementation.md.
Source: unified-force/UniFP@68847a070f88d731058c3d8476929bc3b205f5bd.
Distributed under LICENSE and LICENSES/legged_gym/LICENSE.
"""
import numpy as np
import torch
from .math import (euler_from_quat, sphere2cart, cart2sphere, quat_apply,
                   quat_rotate_inverse, quat_from_euler_xyz, get_euler_xyz,
                   torch_rand_float)

INDEX_EE_POS_RADIUS_CMD = 3
INDEX_EE_POS_PITCH_CMD = 4
INDEX_EE_POS_YAW_CMD = 5
INDEX_EE_ROLL_CMD = 6
INDEX_EE_PITCH_CMD =7
INDEX_EE_YAW_CMD = 8
INDEX_EE_FORCE_X = 9
INDEX_EE_FORCE_Y = 10
INDEX_EE_FORCE_Z = 11
INDEX_BASE_FORCE_X = 12
INDEX_BASE_FORCE_Y = 13
INDEX_BASE_FORCE_Z = 14



class UniFPEquations:
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.base_euler_xyz[:,1])>1.0, torch.abs(self.base_euler_xyz[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf


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


    def get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        body_angles = torch.stack([r, p, y], dim=-1)

        if not return_yaw:
            return body_angles[:, :-1]
        else:
            return body_angles


    def compute_observations(self):
        """ Computes observations
        """

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)


        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        diff = self.dof_pos[:, :12] - self.ref_dof_pos

        arm_base_pos = self.base_pos + quat_apply(self.base_yaw_quat, self.arm_base_offset)
        ee_goal_local_cart = quat_rotate_inverse(self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos)

        ee_local_cart = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - self.get_ee_goal_spherical_center())
        # Spherical to cartesian coordinates in the arm base frame 
        radius = torch.norm(ee_local_cart, dim=1).view(self.num_envs,1)
        pitch = torch.asin(ee_local_cart[:,2].view(self.num_envs,1)/radius).view(self.num_envs,1)
        yaw = torch.atan2(ee_local_cart[:,1].view(self.num_envs,1), ee_local_cart[:,0].view(self.num_envs,1)).view(self.num_envs,1)
        self.ee_pos_sphe_arm = torch.cat((radius, pitch, yaw), dim=1).view(self.num_envs,3)
        
        base_quat_world = self.base_quat.view(self.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(0 * base_rpy_world[:, 0], 0 * base_rpy_world[:, 1], base_rpy_world[:, 2])
        forces_global_gripper = self.forces[:, self.gripper_idx, 0:3]
        self.forces_local[:, self.gripper_idx] = quat_rotate_inverse(base_quat_world_indep, forces_global_gripper).view(self.num_envs, 3)

        forces_global_base = self.forces[:, self.robot_base_idx, 0:3]
        self.forces_local[:, self.robot_base_idx] = quat_rotate_inverse(base_quat_world_indep, forces_global_base).view(self.num_envs, 3)
        

        # offset

        forces_global = self.forces[:, self.gripper_idx, 0:3]
        forces_cmd = self.current_Fxyz_gripper_cmd
        forces_cmd_global = quat_apply(self.base_yaw_quat, forces_cmd)
        forces_offset = (forces_global + forces_cmd_global)
        curr_ee_goal_cart_world_offset = forces_offset / self.gripper_force_kps + self.curr_ee_goal_cart_world
        ee_goal_offset_local_cart = quat_rotate_inverse(self.base_yaw_quat, curr_ee_goal_cart_world_offset - self.get_ee_goal_spherical_center())
        ee_goal_offset_local_sphere = cart2sphere(ee_goal_offset_local_cart)

        self.privileged_obs_buf = torch.cat((
                                    self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                    self.ee_pos_sphe_arm[:, 0:1] * self.obs_scales.ee_sphe_radius_cmd, 
                                    self.ee_pos_sphe_arm[:, 1:2] * self.obs_scales.ee_sphe_pitch_cmd,
                                    self.ee_pos_sphe_arm[:, 2:3] * self.obs_scales.ee_sphe_yaw_cmd, # 3
                                    self.forces_local[:, self.gripper_idx] * self.obs_scales.ee_force, # 3
                                    self.forces_local[:, self.robot_base_idx] * self.obs_scales.base_force, # 3
                                    diff, # 12
                                    self.mass_params_tensor, # 22
                                    self.friction_coeffs_tensor, #  1
                                    self.motor_strength[:, :17] - 1, # 17
                                    stance_mask, # 4
                                    contact_mask, # 4
                                    self.projected_gravity, # 3
                                    self.base_ang_vel * self.obs_scales.ang_vel,  # dim 3
                                    ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)[:, :-self.cfg.env.num_gripper_joints], # dim 17
                                    (self.dof_vel * self.obs_scales.dof_vel)[:, :-self.cfg.env.num_gripper_joints], # dim 17
                                    self.actions[:, :17], # dim 17
                                    sin_pos, # 1
                                    cos_pos, # 1
                                    (self.commands * self.commands_scale)[:, :15], # dim 15
                                    # base_lin_vel_offset * self.obs_scales.lin_vel, # dim 2
                                    ee_goal_offset_local_sphere[:, 0:1] * self.obs_scales.ee_sphe_radius_cmd, 
                                    ee_goal_offset_local_sphere[:, 1:2] * self.obs_scales.ee_sphe_pitch_cmd,
                                    ee_goal_offset_local_sphere[:, 2:3] * self.obs_scales.ee_sphe_yaw_cmd, # 3
                                    ),dim=-1)
        obs_pred = torch.cat((
                                    self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                    self.ee_pos_sphe_arm[:, 0:1] * self.obs_scales.ee_sphe_radius_cmd, 
                                    self.ee_pos_sphe_arm[:, 1:2] * self.obs_scales.ee_sphe_pitch_cmd,
                                    self.ee_pos_sphe_arm[:, 2:3] * self.obs_scales.ee_sphe_yaw_cmd, # 3
                                    self.forces_local[:, self.gripper_idx] * self.obs_scales.ee_force, # 3
                                    self.forces_local[:, self.robot_base_idx] * self.obs_scales.base_force, # 3
                                    ),dim=-1)
        
        obs_buf = torch.cat(( self.get_body_orientation(),  # dim 2
                                    self.base_ang_vel * self.obs_scales.ang_vel,  # dim 3
                                    ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)[:, :-self.cfg.env.num_gripper_joints], # dim 17
                                    (self.dof_vel * self.obs_scales.dof_vel)[:, :-self.cfg.env.num_gripper_joints], # dim 17
                                    self.actions[:, :17], # dim 17
                                    sin_pos, # 1
                                    cos_pos, # 1
                                    (self.commands * self.commands_scale)[:, :15], # dim 15
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
        self.obs_pred = obs_pred.clone()
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)


    def get_observations(self):
        return {'obs': self.obs_buf, 'privileged_obs': self.privileged_obs_buf, 'obs_pred': self.obs_pred}


    def _step_contact_targets(self):
        cycle_time = self.cfg.rewards.cycle_time
        standing_mask = ~self.get_walking_cmd_mask()
        self.gait_indices = torch.remainder(self.gait_indices + self.dt / cycle_time, 1.0)
        self.gait_indices[standing_mask] = 0


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if self.cfg.env.teleop_mode:
            return
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        zero_cmd_mask = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False) < self.cfg.commands.zero_vel_cmd_prob
        self.commands[env_ids, :3] *= ~zero_cmd_mask.unsqueeze(1)

        # set small commands to zero
        non_stop_sign = (torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip) | (torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip) | (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip)
        self.commands[env_ids, :3] *= non_stop_sign.unsqueeze(1)


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.motor_strength * self.cfg.control.action_scale

        default_torques = self.p_gains * (actions_scaled + self.default_dof_pos_wo_gripper - self.dof_pos_wo_gripper) - self.d_gains * self.dof_vel_wo_gripper
        last_torque = 64. * (torch.zeros([actions_scaled.shape[0], 1], device=actions_scaled.device) + self.default_dof_pos[:, -2] - self.dof_pos[:, -2:-1]) - 1.5 * self.dof_vel[:, -2:-1]
        gripper_torque = 64. * (torch.zeros([actions_scaled.shape[0], 1], device=actions_scaled.device) + self.default_dof_pos[:, -1] - self.dof_pos[:, -1:]) - 1.5 * self.dof_vel[:, -1:]
        
        torques = torch.cat([default_torques, last_torque, gripper_torque], dim=-1)

        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    def _resample_ee_goal_sphere_once(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_ranges["pos_l"][0], self.goal_ee_ranges["pos_l"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_ranges["pos_p"][0], self.goal_ee_ranges["pos_p"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_ranges["pos_y"][0], self.goal_ee_ranges["pos_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)


    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_ranges["delta_orn_r"][0], self.goal_ee_ranges["delta_orn_r"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_ranges["delta_orn_p"][0], self.goal_ee_ranges["delta_orn_p"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_ranges["delta_orn_y"][0], self.goal_ee_ranges["delta_orn_y"][1], (len(env_ids), 1), device=self.device)
        self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)


    def _resample_ee_goal(self, env_ids, is_init=False):
        if self.cfg.env.teleop_mode and is_init:
            self.curr_ee_goal_sphere[:] = self.init_start_ee_sphere[:]
            self.commands[:, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)] = self.curr_ee_goal_sphere.view(self.num_envs,3)
            return
        elif self.cfg.env.teleop_mode:
            return

        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()

            ee_local_cart = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - self.get_ee_goal_spherical_center())
                    # Spherical to cartesian coordinates in the arm base frame 
            radius = torch.norm(ee_local_cart, dim=1).view(self.num_envs,1)
            pitch = torch.asin(ee_local_cart[:,2].view(self.num_envs,1)/radius).view(self.num_envs,1)
            yaw = torch.atan2(ee_local_cart[:,1].view(self.num_envs,1), ee_local_cart[:,0].view(self.num_envs,1)).view(self.num_envs,1)
            ee_pos_sphe_arm = torch.cat((radius, pitch, yaw), dim=1).view(self.num_envs,3)

            if is_init:
                if self.global_steps < 0 * 24 and not self.play:
                    self.ee_goal_orn_delta_rpy[env_ids, :] = 0
                    self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                    self.ee_goal_sphere[env_ids] = self.init_start_ee_sphere[:]
                else:
                    self.ee_goal_orn_delta_rpy[env_ids, :] = 0
                    self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                    self.ee_goal_sphere[env_ids] = self.init_end_ee_sphere[:]
            else:
                if self.global_steps < 0 * 24 and not self.play:
                    self.ee_goal_orn_delta_rpy[env_ids, :] = 0
                    self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
                    self.ee_goal_sphere[env_ids] = self.init_start_ee_sphere[:]
                else:
                    self._resample_ee_goal_orn_once(env_ids)
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
        if not self.cfg.env.teleop_mode:
            t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
            self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
            self.commands[:, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)] = self.curr_ee_goal_sphere.view(self.num_envs,3)

        self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        ee_goal_cart_yaw_global = quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)
        # TODO: add twisting motion by fixing yaw at traj start
        self.curr_ee_goal_cart_world = self.get_ee_goal_spherical_center() + ee_goal_cart_yaw_global
        
        default_yaw = torch.atan2(ee_goal_cart_yaw_global[:, 1], ee_goal_cart_yaw_global[:, 0])
        default_pitch = -self.curr_ee_goal_sphere[:, 1] + self.cfg.goal_ee.arm_induced_pitch
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_delta_rpy[:, 0] + np.pi / 2, default_pitch + self.ee_goal_orn_delta_rpy[:, 1], self.ee_goal_orn_delta_rpy[:, 2] + default_yaw)
        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        
        if len(resample_id) > 0 and self.stop_update_goal:
            # set these env commands as 0
            self.commands[resample_id, 0] = 0
            self.commands[resample_id, 1] = 0
            self.commands[resample_id, 2] = 0
            
            
        self._resample_ee_goal(resample_id)


    def get_ee_goal_spherical_center(self):
        center = torch.cat([self.root_states[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        center = center + quat_apply(self.base_yaw_quat, self.ee_goal_center_offset)
        return center


    def _push_gripper(self, env_ids_all):
        """ Randomly pushes the gripper stators. Emulates an impulse by setting a randomized gripper stator velocity.
        """
        
        if self.cfg.commands.push_gripper_stators:
            # cmd force
            # FORCE CONTROLLED ENVS
            new_selected_env_ids_cmd = env_ids_all[(self.episode_length_buf % self.push_interval_gripper_cmd[:, 0]) == 0]
            
            # Define force and duration for the push 
            if new_selected_env_ids_cmd.nelement() > 0:
                
                self.freed_envs_gripper_cmd[new_selected_env_ids_cmd] = torch.rand(len(new_selected_env_ids_cmd), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.gripper_forced_prob_cmd
                min_force_cmd = self.cfg.commands.max_push_force_xyz_gripper_cmd[0]
                max_force_cmd = self.cfg.commands.max_push_force_xyz_gripper_cmd[1]

                self.force_target_gripper_cmd[new_selected_env_ids_cmd, 0] = torch_rand_float(min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                self.force_target_gripper_cmd[new_selected_env_ids_cmd, 1] = torch_rand_float(min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                self.force_target_gripper_cmd[new_selected_env_ids_cmd, 2] = torch_rand_float(min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                push_duration_gripper_cmd = torch_rand_float(self.push_duration_gripper_cmd_min, self.push_duration_gripper_cmd_max, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd)) # 4.0/self.dt
                push_duration_gripper_cmd = torch.clip(push_duration_gripper_cmd, max=(self.push_interval_gripper_cmd[new_selected_env_ids_cmd, 0] - self.settling_time_force_gripper)/2).to(self.device)
                self.push_end_time_gripper_cmd[new_selected_env_ids_cmd] = self.episode_length_buf[new_selected_env_ids_cmd] + push_duration_gripper_cmd
                self.push_duration_gripper_cmd[new_selected_env_ids_cmd] = push_duration_gripper_cmd
                
                self.selected_env_ids_gripper_cmd[new_selected_env_ids_cmd] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_gripper_cmd == 1]

                # Step 1: apply force from 0 to force_target_gripper_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1] < (self.push_end_time_gripper_cmd[self.selected_env_ids_gripper_cmd == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_cmd[env_ids_apply_push_step1].unsqueeze(-1)
                    
                    self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, :3] = (self.force_target_gripper_cmd[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time_gripper_cmd[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                    
                    self.commands[env_ids_apply_push_step1, INDEX_EE_FORCE_X] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 0] #torch.norm(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, :2], dim=1)
                    self.commands[env_ids_apply_push_step1, INDEX_EE_FORCE_Y] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 1] #torch.atan2(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 1], self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 0])
                    self.commands[env_ids_apply_push_step1, INDEX_EE_FORCE_Z] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 2]
 
                # Step 2: apply force from force_target_gripper_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1] > (self.push_end_time_gripper_cmd[self.selected_env_ids_gripper_cmd == 1] + self.settling_time_force_gripper).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_cmd[env_ids_apply_push_step2].unsqueeze(-1)
                    self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, :3] = self.force_target_gripper_cmd[env_ids_apply_push_step2, :3] - (self.force_target_gripper_cmd[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time_gripper_cmd[env_ids_apply_push_step2].unsqueeze(-1)+self.settling_time_force_gripper), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                
                    # World frame
                    self.commands[env_ids_apply_push_step2, INDEX_EE_FORCE_X] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 0] #torch.norm(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, :2], dim=1)
                    self.commands[env_ids_apply_push_step2, INDEX_EE_FORCE_Y] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 1] #torch.atan2(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 1], self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 0])
                    self.commands[env_ids_apply_push_step2, INDEX_EE_FORCE_Z] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 2]
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1] >= (self.push_end_time_gripper_cmd[self.selected_env_ids_gripper_cmd == 1] + self.settling_time_force_gripper + self.push_duration_gripper_cmd[self.selected_env_ids_gripper_cmd == 1]).type(torch.int32)]
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_gripper_cmd[env_ids_to_reset] = 0
                    self.force_target_gripper_cmd[env_ids_to_reset, :3] = 0.
                    self.current_Fxyz_gripper_cmd[env_ids_to_reset, :3] = 0.
                    self.push_end_time_gripper_cmd[env_ids_to_reset] = 0.
                    self.push_duration_gripper_cmd[env_ids_to_reset] = 0.
                    self.commands[env_ids_to_reset, INDEX_EE_FORCE_X] = 0.0
                    self.commands[env_ids_to_reset, INDEX_EE_FORCE_Y] = 0.0
                    self.commands[env_ids_to_reset, INDEX_EE_FORCE_Z] = 0.0
                    self.push_interval_gripper_cmd[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_gripper_cmd_min), int(self.push_interval_gripper_cmd_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_gripper_cmd[self.freed_envs_gripper_cmd] = 0
            self.force_target_gripper_cmd[self.freed_envs_gripper_cmd, :3] = 0.
            self.current_Fxyz_gripper_cmd[self.freed_envs_gripper_cmd, :3] = 0.
            self.push_end_time_gripper_cmd[self.freed_envs_gripper_cmd] = 0.
            self.push_duration_gripper_cmd[self.freed_envs_gripper_cmd] = 0. 
            self.commands[self.freed_envs_gripper_cmd, INDEX_EE_FORCE_X] = 0.0
            self.commands[self.freed_envs_gripper_cmd, INDEX_EE_FORCE_Y] = 0.0
            self.commands[self.freed_envs_gripper_cmd, INDEX_EE_FORCE_Z] = 0.0


            # ext force
            # FORCE CONTROLLED ENVS
            new_selected_env_ids_ext = env_ids_all[(self.episode_length_buf % self.push_interval_gripper_ext[:, 0]) == 0]
            
            # Define force and duration for the push 
            if new_selected_env_ids_ext.nelement() > 0:
                
                self.freed_envs_gripper_ext[new_selected_env_ids_ext] = torch.rand(len(new_selected_env_ids_ext), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.gripper_forced_prob_ext
                min_force_ext = self.cfg.commands.max_push_force_xyz_gripper_ext[0]
                max_force_ext = self.cfg.commands.max_push_force_xyz_gripper_ext[1]

                self.force_target_gripper_ext[new_selected_env_ids_ext, 0] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                self.force_target_gripper_ext[new_selected_env_ids_ext, 1] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                self.force_target_gripper_ext[new_selected_env_ids_ext, 2] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                push_duration_gripper_ext = torch_rand_float(self.push_duration_gripper_ext_min, self.push_duration_gripper_ext_max, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext)) # 4.0/self.dt
                push_duration_gripper_ext = torch.clip(push_duration_gripper_ext, max=(self.push_interval_gripper_ext[new_selected_env_ids_ext, 0] - self.settling_time_force_gripper)/2).to(self.device)
                self.push_end_time_gripper_ext[new_selected_env_ids_ext] = self.episode_length_buf[new_selected_env_ids_ext] + push_duration_gripper_ext
                self.push_duration_gripper_ext[new_selected_env_ids_ext] = push_duration_gripper_ext
                
                self.selected_env_ids_gripper_ext[new_selected_env_ids_ext] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_gripper_ext == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_gripper_ext == 1]

                # Step 1: apply force from 0 to force_target_gripper_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_ext == 1] < (self.push_end_time_gripper_ext[self.selected_env_ids_gripper_ext == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_ext[env_ids_apply_push_step1].unsqueeze(-1)
                    
                    self.forces[env_ids_apply_push_step1, self.gripper_idx, :3] = (self.force_target_gripper_ext[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time_gripper_ext[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                  
                # Step 2: apply force from force_target_gripper_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_ext == 1] > (self.push_end_time_gripper_ext[self.selected_env_ids_gripper_ext == 1] + self.settling_time_force_gripper).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_ext[env_ids_apply_push_step2].unsqueeze(-1)
                    
                    # world frame
                    self.forces[env_ids_apply_push_step2, self.gripper_idx, :3] = self.force_target_gripper_ext[env_ids_apply_push_step2, :3] - (self.force_target_gripper_ext[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time_gripper_ext[env_ids_apply_push_step2].unsqueeze(-1)+self.settling_time_force_gripper), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_ext == 1] >= (self.push_end_time_gripper_ext[self.selected_env_ids_gripper_ext == 1] + self.settling_time_force_gripper + self.push_duration_gripper_ext[self.selected_env_ids_gripper_ext == 1]).type(torch.int32)]                                        
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_gripper_ext[env_ids_to_reset] = 0
                    self.force_target_gripper_ext[env_ids_to_reset, :3] = 0.
                    self.push_end_time_gripper_ext[env_ids_to_reset] = 0.
                    self.push_duration_gripper_ext[env_ids_to_reset] = 0.
                    self.push_interval_gripper_ext[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_gripper_ext_min), int(self.push_interval_gripper_ext_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_gripper_ext[self.freed_envs_gripper_ext] = 0
            self.force_target_gripper_ext[self.freed_envs_gripper_ext, :3] = 0.
            self.push_end_time_gripper_ext[self.freed_envs_gripper_ext] = 0.
            self.push_duration_gripper_ext[self.freed_envs_gripper_ext] = 0. 
            
            self.forces[self.freed_envs_gripper_ext, self.gripper_idx, :3] = 0


    def _push_robot_base(self, env_ids_all):
        """ 
            Randomly pushes the robot base.
        """
        if self.cfg.commands.push_robot_base:
            # cmd force
            # FORCE CONTROLLED ENVS
            new_selected_env_ids_cmd = env_ids_all[(self.episode_length_buf % self.push_interval_base_cmd[:, 0]) == 0]
            
            # Define force and duration for the push 
            if new_selected_env_ids_cmd.nelement() > 0:
                
                self.freed_envs_base_cmd[new_selected_env_ids_cmd] = torch.rand(len(new_selected_env_ids_cmd), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.base_forced_prob_cmd
                min_force_cmd = self.cfg.commands.max_push_force_xyz_base_cmd[0]
                max_force_cmd = self.cfg.commands.max_push_force_xyz_base_cmd[1]

                self.force_target_base_cmd[new_selected_env_ids_cmd, 0] = torch_rand_float(min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                self.force_target_base_cmd[new_selected_env_ids_cmd, 1] = torch_rand_float(min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                self.force_target_base_cmd[new_selected_env_ids_cmd, 2] = torch.zeros((len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                push_duration_base_cmd = torch_rand_float(self.push_duration_base_cmd_min, self.push_duration_base_cmd_max, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd)) # 4.0/self.dt
                push_duration_base_cmd = torch.clip(push_duration_base_cmd, max=(self.push_interval_base_cmd[new_selected_env_ids_cmd, 0] - self.settling_time_force_base)/2).to(self.device)
                self.push_end_time_base_cmd[new_selected_env_ids_cmd] = self.episode_length_buf[new_selected_env_ids_cmd] + push_duration_base_cmd
                self.push_duration_base_cmd[new_selected_env_ids_cmd] = push_duration_base_cmd
                
                self.selected_env_ids_base_cmd[new_selected_env_ids_cmd] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_base_cmd == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_base_cmd == 1]

                # Step 1: apply force from 0 to force_target_base_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_cmd == 1] < (self.push_end_time_base_cmd[self.selected_env_ids_base_cmd == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_base_cmd[env_ids_apply_push_step1].unsqueeze(-1)
                    
                    self.current_Fxyz_base_cmd[env_ids_apply_push_step1, :3] = (self.force_target_base_cmd[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time_base_cmd[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                    
                    self.commands[env_ids_apply_push_step1, INDEX_BASE_FORCE_X] = self.current_Fxyz_base_cmd[env_ids_apply_push_step1, 0] #torch.norm(self.current_Fxyz_base_cmd[env_ids_apply_push_step1, :2], dim=1)
                    self.commands[env_ids_apply_push_step1, INDEX_BASE_FORCE_Y] = self.current_Fxyz_base_cmd[env_ids_apply_push_step1, 1] #torch.atan2(self.current_Fxyz_base_cmd[env_ids_apply_push_step1, 1], self.current_Fxyz_base_cmd[env_ids_apply_push_step1, 0])
                    self.commands[env_ids_apply_push_step1, INDEX_BASE_FORCE_Z] = self.current_Fxyz_base_cmd[env_ids_apply_push_step1, 2]
 
                # Step 2: apply force from force_target_base_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_cmd == 1] > (self.push_end_time_base_cmd[self.selected_env_ids_base_cmd == 1] + self.settling_time_force_base).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_base_cmd[env_ids_apply_push_step2].unsqueeze(-1)
                    self.current_Fxyz_base_cmd[env_ids_apply_push_step2, :3] = self.force_target_base_cmd[env_ids_apply_push_step2, :3] - (self.force_target_base_cmd[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time_base_cmd[env_ids_apply_push_step2].unsqueeze(-1)+self.settling_time_force_base), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                
                    # World frame
                    self.commands[env_ids_apply_push_step2, INDEX_BASE_FORCE_X] = self.current_Fxyz_base_cmd[env_ids_apply_push_step2, 0] #torch.norm(self.current_Fxyz_base_cmd[env_ids_apply_push_step2, :2], dim=1)
                    self.commands[env_ids_apply_push_step2, INDEX_BASE_FORCE_Y] = self.current_Fxyz_base_cmd[env_ids_apply_push_step2, 1] #torch.atan2(self.current_Fxyz_base_cmd[env_ids_apply_push_step2, 1], self.current_Fxyz_base_cmd[env_ids_apply_push_step2, 0])
                    self.commands[env_ids_apply_push_step2, INDEX_BASE_FORCE_Z] = self.current_Fxyz_base_cmd[env_ids_apply_push_step2, 2]
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_cmd == 1] >= (self.push_end_time_base_cmd[self.selected_env_ids_base_cmd == 1] + self.settling_time_force_base + self.push_duration_base_cmd[self.selected_env_ids_base_cmd == 1]).type(torch.int32)]
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_base_cmd[env_ids_to_reset] = 0
                    self.force_target_base_cmd[env_ids_to_reset, :3] = 0.
                    self.current_Fxyz_base_cmd[env_ids_to_reset, :3] = 0.
                    self.push_end_time_base_cmd[env_ids_to_reset] = 0.
                    self.push_duration_base_cmd[env_ids_to_reset] = 0.
                    self.commands[env_ids_to_reset, INDEX_BASE_FORCE_X] = 0.0
                    self.commands[env_ids_to_reset, INDEX_BASE_FORCE_Y] = 0.0
                    self.commands[env_ids_to_reset, INDEX_BASE_FORCE_Z] = 0.0
                    self.push_interval_base_cmd[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_base_cmd_min), int(self.push_interval_base_cmd_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_base_cmd[self.freed_envs_base_cmd] = 0
            self.force_target_base_cmd[self.freed_envs_base_cmd, :3] = 0.
            self.current_Fxyz_base_cmd[self.freed_envs_base_cmd, :3] = 0.
            self.push_end_time_base_cmd[self.freed_envs_base_cmd] = 0.
            self.push_duration_base_cmd[self.freed_envs_base_cmd] = 0. 
            self.commands[self.freed_envs_base_cmd, INDEX_BASE_FORCE_X] = 0.0
            self.commands[self.freed_envs_base_cmd, INDEX_BASE_FORCE_Y] = 0.0
            self.commands[self.freed_envs_base_cmd, INDEX_BASE_FORCE_Z] = 0.0

            # ext force
            # FORCE CONTROLLED ENVS
            new_selected_env_ids_ext = env_ids_all[(self.episode_length_buf % self.push_interval_base_ext[:, 0]) == 0]
            
            # Define force and duration for the push 
            if new_selected_env_ids_ext.nelement() > 0:
                
                self.freed_envs_base_ext[new_selected_env_ids_ext] = torch.rand(len(new_selected_env_ids_ext), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.base_forced_prob_ext
                min_force_ext = self.cfg.commands.max_push_force_xyz_base_ext[0]
                max_force_ext = self.cfg.commands.max_push_force_xyz_base_ext[1]

                self.force_target_base_ext[new_selected_env_ids_ext, 0] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                self.force_target_base_ext[new_selected_env_ids_ext, 1] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                self.force_target_base_ext[new_selected_env_ids_ext, 2] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext)) * self.cfg.commands.force_z_base_ext_scale
                push_duration_base_ext = torch_rand_float(self.push_duration_base_ext_min, self.push_duration_base_ext_max, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext)) # 4.0/self.dt
                push_duration_base_ext = torch.clip(push_duration_base_ext, max=(self.push_interval_base_ext[new_selected_env_ids_ext, 0] - self.settling_time_force_base)/2).to(self.device)
                self.push_end_time_base_ext[new_selected_env_ids_ext] = self.episode_length_buf[new_selected_env_ids_ext] + push_duration_base_ext
                self.push_duration_base_ext[new_selected_env_ids_ext] = push_duration_base_ext
                
                self.selected_env_ids_base_ext[new_selected_env_ids_ext] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_base_ext == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_base_ext == 1]

                # Step 1: apply force from 0 to force_target_base_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_ext == 1] < (self.push_end_time_base_ext[self.selected_env_ids_base_ext == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_base_ext[env_ids_apply_push_step1].unsqueeze(-1)
                    
                    self.forces[env_ids_apply_push_step1, self.robot_base_idx, :3] = (self.force_target_base_ext[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time_base_ext[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                  
                # Step 2: apply force from force_target_base_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_ext == 1] > (self.push_end_time_base_ext[self.selected_env_ids_base_ext == 1] + self.settling_time_force_base).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_base_ext[env_ids_apply_push_step2].unsqueeze(-1)
                    
                    # world frame
                    self.forces[env_ids_apply_push_step2, self.robot_base_idx, :3] = self.force_target_base_ext[env_ids_apply_push_step2, :3] - (self.force_target_base_ext[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time_base_ext[env_ids_apply_push_step2].unsqueeze(-1)+self.settling_time_force_base), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_base_ext == 1] >= (self.push_end_time_base_ext[self.selected_env_ids_base_ext == 1] + self.settling_time_force_base + self.push_duration_base_ext[self.selected_env_ids_base_ext == 1]).type(torch.int32)]
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_base_ext[env_ids_to_reset] = 0
                    self.force_target_base_ext[env_ids_to_reset, :3] = 0.
                    self.push_end_time_base_ext[env_ids_to_reset] = 0.
                    self.push_duration_base_ext[env_ids_to_reset] = 0.
                    self.push_interval_base_ext[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_base_ext_min), int(self.push_interval_base_ext_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_base_ext[self.freed_envs_base_ext] = 0
            self.force_target_base_ext[self.freed_envs_base_ext, :3] = 0.
            self.push_end_time_base_ext[self.freed_envs_base_ext] = 0.
            self.push_duration_base_ext[self.freed_envs_base_ext] = 0. 

            self.forces[self.freed_envs_base_ext, self.robot_base_idx, :3] = 0


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:2] = noise_scales.gravity * noise_level
        noise_vec[2:5] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[5:5+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[5+self.num_actions:5+self.num_actions*2] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[5+self.num_actions*2:5+self.num_actions*3] = 0. # previous actions
        noise_vec[5+self.num_actions*3:] = 0. # commands

        return noise_vec


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}


    def get_walking_cmd_mask(self, env_ids=None, return_all=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        walking_mask0 = torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip
        walking_mask1 = torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip
        walking_mask2 = torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2

        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask


    def  _get_phase(self):
        phase = self.gait_indices
        return phase


    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)

        sin_pos_l = sin_pos.clone() + self.cfg.rewards.target_joint_pos_thd
        sin_pos_r = sin_pos.clone() - self.cfg.rewards.target_joint_pos_thd
        
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 4), device=self.device)
        # FL RR foot stance
        stance_mask[:, 0] = sin_pos_l >= 0
        stance_mask[:, 3] = sin_pos_l >= 0
        # FR RL foot stance
        stance_mask[:, 1] = sin_pos_r < 0
        stance_mask[:, 2] = sin_pos_r < 0

        return stance_mask


    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone() + self.cfg.rewards.target_joint_pos_thd
        sin_pos_r = sin_pos.clone() - self.cfg.rewards.target_joint_pos_thd
        repeat_default_pos = self.default_dof_pos[:, :12].repeat(self.num_envs, 1)
        # self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        self.ref_dof_pos = repeat_default_pos.clone()
        scale_1 = self.cfg.rewards.target_joint_pos_scale / (1 - self.cfg.rewards.target_joint_pos_thd)
        scale_2 = scale_1 * 2
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = sin_pos_l[sin_pos_l > 0] * (1 - self.cfg.rewards.target_joint_pos_thd) / (1 + self.cfg.rewards.target_joint_pos_thd) * 0.0
        self.ref_dof_pos[:, 1] -= sin_pos_l * scale_1 # FL_thigh_joint
        self.ref_dof_pos[:, 2] += sin_pos_l * scale_2 # FL_calf_joint
        self.ref_dof_pos[:, 10] -= sin_pos_l * scale_1 # RR_thigh_joint
        self.ref_dof_pos[:, 11] += sin_pos_l * scale_2 # RR_calf_joint

        sin_pos_r[sin_pos_r < 0] = sin_pos_r[sin_pos_r < 0] * (1 - self.cfg.rewards.target_joint_pos_thd) / (1 + self.cfg.rewards.target_joint_pos_thd) * 0.0
        self.ref_dof_pos[:, 4] += sin_pos_r * scale_1 # FR_thigh_joint
        self.ref_dof_pos[:, 5] -= sin_pos_r * scale_2 # FR_calf_joint
        self.ref_dof_pos[:, 7] += sin_pos_r * scale_1 # RL_thigh_joint
        self.ref_dof_pos[:, 8] -= sin_pos_r * scale_2 # RL_calf_joint


    def _reward_tracking_ee_force_world(self):
        forces_global = self.forces[:, self.gripper_idx, 0:3]
        forces_cmd = self.current_Fxyz_gripper_cmd
        forces_cmd_global = quat_apply(self.base_yaw_quat, forces_cmd)
        forces_offset = (forces_global + forces_cmd_global)
        curr_ee_goal_cart_world_offset = forces_offset / self.gripper_force_kps + self.curr_ee_goal_cart_world
       
        ee_pos_error = torch.sum(torch.abs(self.ee_pos - curr_ee_goal_cart_world_offset), dim=1)
        rew = torch.exp(-ee_pos_error/self.cfg.rewards.tracking_ee_sigma * 2)
        return rew


    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])


    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)


    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)


    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques)[:, :12], dim=1)


    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel)[:, :12], dim=1)


    def _reward_dof_vel_arm(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel)[:, 12:17], dim=1)


    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[:, :12] / self.dt), dim=1)


    def _reward_dof_acc_arm(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)[:, 12:17] / self.dt), dim=1)


    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions)[:, :12], dim=1)


    def _reward_action_rate_arm(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions)[:, 12:17], dim=1)


    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)


    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf


    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits[:, :17], dim=1)


    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)


    def _reward_tracking_lin_vel_force_world(self):
        forces_global_base = self.forces[:, self.robot_base_idx, 0:3]
        forces_local_base = quat_rotate_inverse(self.base_yaw_quat, forces_global_base).view(self.num_envs, 3)
    
        forces_cmd_local = self.current_Fxyz_base_cmd
        forces_offset = (forces_local_base + forces_cmd_local)
        base_lin_vel_offset = (forces_offset / self.base_force_kds)[:, :2] + self.commands[:, :2]


        non_stop_sign = (torch.abs(base_lin_vel_offset[:, 0]) > self.cfg.commands.lin_vel_x_clip) | (torch.abs(base_lin_vel_offset[:, 1]) > self.cfg.commands.lin_vel_y_clip) | (torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_yaw_clip)
        base_lin_vel_offset[:, :3] *= non_stop_sign.unsqueeze(1)

        lin_vel_error = torch.sum(torch.square(base_lin_vel_offset - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)


    def _reward_feet_pos_xy(self):
        # Penalize xy axis feet linear velocity
        feet_pos_xy = self.rigid_state[:, self.feet_indices, :2]
        thigh_pos_xy = self.rigid_state[:, self.thigh_indices, :2]
        diff = torch.norm(feet_pos_xy-thigh_pos_xy, dim=2).view(self.num_envs, -1)
        mean_diff = torch.mean(diff, dim=1)
        return mean_diff


    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)


    def _reward_feet_height(self):
        feet_height = self.rigid_state[:, self.feet_indices[:2], 2] # Only front feet
        rew = torch.clamp(torch.max(feet_height, dim=-1)[0] - 0.10, max=0)
        cmd_stop_flag = ~self.get_walking_cmd_mask()
        rew[cmd_stop_flag] = 0
        return rew


    def _reward_feet_height_high(self):
        feet_height = self.rigid_state[:, self.feet_indices, 2]
        rew = torch.clamp(torch.max(feet_height, dim=-1)[0] - 0.20, min=0)
        cmd_stop_flag = ~self.get_walking_cmd_mask()
        rew[cmd_stop_flag] = 0
        return rew


    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= self.get_walking_cmd_mask() #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime


    def _reward_stand_still(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        rew[self.get_walking_cmd_mask()] = 0.
        return rew


    def _reward_ref_dof_leg(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        self.compute_ref_state()
        joint_pos = self.dof_pos.clone()[:, :12]
        pos_target = self.ref_dof_pos.clone()
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(joint_pos - pos_target)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.1)
        return rew


    def _reward_hip_pos(self):
        rew = torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
        return rew


    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)


    def _reward_alive(self):
        return 1.


    def _reward_feet_drag(self):
        feet_xyz_vel = torch.abs(self.rigid_state[:, self.feet_indices, 7:10]).sum(dim=-1)
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        dragging_vel = foot_forces * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        return rew


    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

