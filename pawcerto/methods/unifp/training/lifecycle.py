# Source: unified-force/UniFP@68847a070f88d731058c3d8476929bc3b205f5bd.
# Distributed under ../LICENSE and ../LICENSES/legged_gym/LICENSE.
"""Upstream lifecycle; Gym acquisition replaced by official Lab measured tensors."""
from collections import deque
import numpy as np
import torch
from ..math import *
from ..equations import *
from .config import class_to_dict


def get_euler_xyz_tensor(quat):
    result = torch.stack(get_euler_xyz(quat), dim=1)
    result[result > np.pi] -= 2 * np.pi
    return result


class UniFPLifecycle:
    def _parse_cfg(self, cfg):
        self.num_torques = self.cfg.env.num_torques
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)
     
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        # Gripper 
        self.push_interval_gripper_cmd_min = np.ceil(self.cfg.commands.push_gripper_interval_s_cmd[0] / self.dt)
        self.push_interval_gripper_cmd_max = np.ceil(self.cfg.commands.push_gripper_interval_s_cmd[1] / self.dt)
        self.push_interval_gripper_ext_min = np.ceil(self.cfg.commands.push_gripper_interval_s_ext[0] / self.dt)
        self.push_interval_gripper_ext_max = np.ceil(self.cfg.commands.push_gripper_interval_s_ext[1] / self.dt)
        self.push_duration_gripper_cmd_min = np.ceil(self.cfg.commands.push_gripper_duration_s_cmd[0] / self.dt)
        self.push_duration_gripper_cmd_max = np.ceil(self.cfg.commands.push_gripper_duration_s_cmd[1] / self.dt)
        self.push_duration_gripper_ext_min = np.ceil(self.cfg.commands.push_gripper_duration_s_ext[0] / self.dt)
        self.push_duration_gripper_ext_max = np.ceil(self.cfg.commands.push_gripper_duration_s_ext[1] / self.dt)
        self.settling_time_force_gripper = np.ceil(self.cfg.commands.settling_time_force_gripper_s / self.dt)
        # Base 
        self.push_interval_base_cmd_min = np.ceil(self.cfg.commands.push_base_interval_s_cmd[0] / self.dt)
        self.push_interval_base_cmd_max = np.ceil(self.cfg.commands.push_base_interval_s_cmd[1] / self.dt)
        self.push_interval_base_ext_min = np.ceil(self.cfg.commands.push_base_interval_s_ext[0] / self.dt)
        self.push_interval_base_ext_max = np.ceil(self.cfg.commands.push_base_interval_s_ext[1] / self.dt)
        self.push_duration_base_cmd_min = np.ceil(self.cfg.commands.push_base_duration_s_cmd[0] / self.dt)
        self.push_duration_base_cmd_max = np.ceil(self.cfg.commands.push_base_duration_s_cmd[1] / self.dt)
        self.push_duration_base_ext_min = np.ceil(self.cfg.commands.push_base_duration_s_ext[0] / self.dt)
        self.push_duration_base_ext_max = np.ceil(self.cfg.commands.push_base_duration_s_ext[1] / self.dt)
        self.settling_time_force_base = np.ceil(self.cfg.commands.settling_time_force_base_s / self.dt)
        
        self.action_delay = self.cfg.env.action_delay

        self.stop_update_goal = False

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # put robots at the origins defined by the terrain
        max_init_level = self.cfg.terrain.max_init_terrain_level  # start from 0
        if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
        self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]


    def _randomize_dof_props(self, env_ids):
        if self.cfg.commands.randomize_gripper_force_gains:
            min_kp, max_kp = self.cfg.commands.gripper_force_kp_range
            min_kd, max_kd = self.cfg.commands.gripper_force_kd_range
            self.gripper_force_kps[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_kp - min_kp) + min_kp
            if self.cfg.commands.gripper_prop_kd > 0:
                self.gripper_force_kds[env_ids, :] = self.gripper_force_kps[env_ids, :] * self.cfg.commands.gripper_prop_kd
            else:
                self.gripper_force_kds[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                        requires_grad=False).unsqueeze(1) * (
                                                    max_kd - min_kd) + min_kd
        if self.cfg.commands.randomize_base_force_gains:
            min_kp, max_kp = self.cfg.commands.base_force_kp_range
            min_kd, max_kd = self.cfg.commands.base_force_kd_range
            self.base_force_kps[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_kp - min_kp) + min_kp
            self.base_force_kds[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                        requires_grad=False).unsqueeze(1) * (
                                                    max_kd - min_kd) + min_kd

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._randomize_dof_props(env_ids)
        self._step_contact_targets()

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        self._randomize_dof_props(env_ids)
        self._resample_ee_goal(env_ids, is_init=True)

        # reset buffers
        self.last_torques[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.gait_indices[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.goal_timer[env_ids] = 0.

        # force control
        self.forces[env_ids, self.gripper_idx, :3] = 0.
        self.selected_env_ids_gripper_cmd[env_ids] = 0
        self.selected_env_ids_gripper_ext[env_ids] = 0
        self.push_end_time_gripper_cmd[env_ids] = 0.
        self.force_target_gripper_cmd[env_ids, :3] = 0.
        self.force_target_gripper_ext[env_ids, :3] = 0.
        self.push_duration_gripper_cmd[env_ids] = 0.
        self.current_Fxyz_gripper_cmd[env_ids, :3] = 0.



        self.forces[env_ids, self.robot_base_idx, :3] = 0.
        self.selected_env_ids_base_cmd[env_ids] = 0
        self.selected_env_ids_base_ext[env_ids] = 0
        self.push_end_time_base_cmd[env_ids] = 0.
        self.force_target_base_cmd[env_ids, :3] = 0.
        self.force_target_base_ext[env_ids, :3] = 0.
        self.push_duration_base_cmd[env_ids] = 0.
        self.current_Fxyz_base_cmd[env_ids, :3] = 0.
        
        self.commands[env_ids, INDEX_EE_FORCE_X] = 0.0
        self.commands[env_ids, INDEX_EE_FORCE_Y] = 0.0
        self.commands[env_ids, INDEX_EE_FORCE_Z] = 0.0

        self.commands[env_ids, INDEX_BASE_FORCE_X] = 0.0
        self.commands[env_ids, INDEX_BASE_FORCE_Y] = 0.0
        self.commands[env_ids, INDEX_BASE_FORCE_Z] = 0.0
        

        # Reset push gripper 
        if self.cfg.commands.push_gripper_stators:
            self.forces[env_ids, self.gripper_idx, :3] = 0.
            self.selected_env_ids_gripper_cmd[env_ids] = 0
            self.selected_env_ids_gripper_ext[env_ids] = 0
            self.push_end_time_gripper_cmd[env_ids] = 0.

        # Reset push robot base 
        if self.cfg.commands.push_robot_base:
            self.forces[env_ids, self.robot_base_idx, :3] = 0.
            self.selected_env_ids_base_cmd[env_ids] = 0
            self.selected_env_ids_base_ext[env_ids] = 0
            self.push_end_time_base_cmd[env_ids] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.root_states = self.runtime.root_states.clone()
        self.dof_pos, self.dof_vel = (v.clone() for v in self.runtime.joints())
        self.dof_pos_wo_gripper = self.dof_pos[:, :-self.cfg.env.num_gripper_joints]
        self.dof_vel_wo_gripper = self.dof_vel[:, :-self.cfg.env.num_gripper_joints]
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)

        self.arm_base_offset = torch.tensor([0.2, 0., 0.225], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        
        self.contact_forces = self.runtime.contact_forces.clone() # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = self.runtime.rigid_state.clone()

        self.gripper_position = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_idx, 0:3]
        self.gripper_velocity = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_idx, 7:10]
        # ee info
        self.ee_pos = self.rigid_state[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_state[:, self.gripper_idx, 3:7]
        self.ee_vel = self.rigid_state[:, self.gripper_idx, 7:]
        
        # target_ee info
        self.grasp_offset = self.cfg.arm.grasp_offset
        self.init_target_ee_base = torch.tensor(self.cfg.arm.init_target_ee_base, device=self.device).unsqueeze(0)

        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler[:, 0] = np.pi / 2
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2])
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)

        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_pos_sphe_arm = torch.zeros(self.num_envs, 3, device=self.device)

        self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device).unsqueeze(0)
        
        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        assert(self.cfg.goal_ee.command_mode in ['cart', 'sphere'])
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        self.ee_goal_center_offset = torch.tensor([self.cfg.goal_ee.sphere_center.x_offset, 
                                                   self.cfg.goal_ee.sphere_center.y_offset, 
                                                   self.cfg.goal_ee.sphere_center.z_invariant_offset], 
                                                   device=self.device).repeat(self.num_envs, 1)
        
        self.curr_ee_goal_cart_world = self.get_ee_goal_spherical_center() + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = torch.tensor([0., 0., -1.], device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_torques = torch.zeros_like(self.torques)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, 
                                            self.obs_scales.lin_vel, 
                                            self.obs_scales.ang_vel,
                                            self.obs_scales.ee_sphe_radius_cmd, 
                                            self.obs_scales.ee_sphe_pitch_cmd,
                                            self.obs_scales.ee_sphe_yaw_cmd,
                                            self.obs_scales.end_effector_roll_cmd, 
                                            self.obs_scales.end_effector_pitch_cmd,
                                            self.obs_scales.end_effector_yaw_cmd,
                                            self.obs_scales.ee_force,
                                            self.obs_scales.ee_force,
                                            self.obs_scales.ee_force,
                                            self.obs_scales.base_force,
                                            self.obs_scales.base_force,
                                            self.obs_scales.base_force,], device=self.device, requires_grad=False,) # TODO change this
        
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.cfg.env.num_gripper_joints, device=self.device)

        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
      
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

        for i in range(self.num_torques):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_wo_gripper = self.default_dof_pos[:, :-self.cfg.env.num_gripper_joints]
        
        # gait
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
        self.foot_velocities = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]

        # force control

        # Push gripper 
        self.freed_envs_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.freed_envs_gripper_ext = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.selected_env_ids_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.selected_env_ids_gripper_ext = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.push_interval_gripper_cmd = torch.randint(int(self.push_interval_gripper_cmd_min), int(self.push_interval_gripper_cmd_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_interval_gripper_ext = torch.randint(int(self.push_interval_gripper_ext_min), int(self.push_interval_gripper_ext_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_end_time_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.settling_time_force_gripper_s = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_end_time_gripper_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_gripper_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target_gripper_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target_gripper_ext = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_Fxyz_gripper_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.gripper_force_kps = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_force_kds = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Push robot 
        self.freed_envs_base_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.freed_envs_base_ext = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.selected_env_ids_base_cmd = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.selected_env_ids_base_ext = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.push_interval_base_cmd = torch.randint(int(self.push_interval_base_cmd_min), int(self.push_interval_base_cmd_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_interval_base_ext = torch.randint(int(self.push_interval_base_ext_min), int(self.push_interval_base_ext_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        self.push_end_time_base_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_base_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.settling_time_force_base_s = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_end_time_base_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration_base_ext = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target_base_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target_base_ext = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_Fxyz_base_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.base_force_kps = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_force_kds = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.forces_local = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device,
                                   requires_grad=False)

        self.global_steps = 0
