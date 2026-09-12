"""Tensor task semantics ported from RoboDuet a7e1528215c048199f90cb69ceb7749a1d745f28.

Source: automatic/legged_robot.py, automatic/curriculum.py, rewards/rewards.py
(MIT RoboDuet; inherited legged_gym BSD-3-Clause). Physics and controller buffers
are supplied by Isaac environment in upstream ordering and xyzw convention.
No simulator imports, runtime source execution, or proxy reward functions.
"""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch
from ..observations import (quat_apply, quat_mul, quat_conjugate,
                            quat_from_euler_xyz, quaternion_to_rpy)

quat_rotate = quat_apply

def quat_rotate_inverse(q, v):
    return quat_apply(quat_conjugate(q), v)

def quat_apply_yaw(q, v):
    q = q.clone()
    q[..., :2] = 0
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return quat_apply(q, v)

def quat_from_angle_axis(angle, axis):
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    return torch.cat((axis * torch.sin(angle[..., None] / 2), torch.cos(angle[..., None] / 2)), dim=-1)

def to_torch(value, device):
    return torch.tensor(value, device=device)

def torch_rand_float(low, high, shape, device):
    return (high-low)*torch.rand(*shape, device=device)+low

def _namespace(value):
    return SimpleNamespace(**{k: _namespace(v) for k,v in value.items()}) if isinstance(value, dict) else deepcopy(value)

def _rotation6d(quats):
    # PyTorch3D matrix_to_rotation_6d selects the first two ROWS.
    axes = torch.eye(3, device=quats.device, dtype=quats.dtype)
    matrix = torch.stack([quat_apply(quats, axis.expand(len(quats), -1)) for axis in axes], dim=-1)
    return matrix[:, :2, :].reshape(-1, 6)

class Curriculum:

    def set_to(self, low, high, value=1.0):
        inds = np.logical_and(self.grid >= low[:, None], self.grid <= high[:, None]).all(axis=0)
        assert len(inds) != 0, 'You are intializing your distribution with an empty domain!'
        self.weights[inds] = value

    def __init__(self, seed, **key_ranges):
        self.rng = np.random.RandomState(seed)
        self.cfg = cfg = {}
        self.indices = indices = {}
        for key, v_range in key_ranges.items():
            bin_size = (v_range[1] - v_range[0]) / v_range[2]
            cfg[key] = np.linspace(v_range[0] + bin_size / 2, v_range[1] - bin_size / 2, v_range[2])
            indices[key] = np.linspace(0, v_range[2] - 1, v_range[2])
        self.lows = np.array([range[0] for range in key_ranges.values()])
        self.highs = np.array([range[1] for range in key_ranges.values()])
        self.bin_sizes = {key: (v_range[1] - v_range[0]) / v_range[2] for key, v_range in key_ranges.items()}
        self._raw_grid = np.stack(np.meshgrid(*cfg.values(), indexing='ij'))
        self._idx_grid = np.stack(np.meshgrid(*indices.values(), indexing='ij'))
        self.keys = [*key_ranges.keys()]
        self.grid = self._raw_grid.reshape([len(self.keys), -1])
        self.idx_grid = self._idx_grid.reshape([len(self.keys), -1])
        self._l = l = len(self.grid[0])
        self.ls = {key: len(self.cfg[key]) for key in self.cfg.keys()}
        self.weights = np.zeros(l)
        self.indices = np.arange(l)

    def __len__(self):
        return self._l

    def update(self, **kwargs):
        pass

    def sample_bins(self, batch_size, low=None, high=None):
        """default to uniform"""
        if low is not None and high is not None:
            valid_inds = np.logical_and(self.grid >= low[:, None], self.grid <= high[:, None]).all(axis=0)
            temp_weights = np.zeros_like(self.weights)
            temp_weights[valid_inds] = self.weights[valid_inds]
            inds = self.rng.choice(self.indices, batch_size, p=temp_weights / temp_weights.sum())
        else:
            inds = self.rng.choice(self.indices, batch_size, p=self.weights / self.weights.sum())
        return (self.grid.T[inds], inds)

    def sample_uniform_from_cell(self, centroids):
        bin_sizes = np.array([*self.bin_sizes.values()])
        low, high = (centroids + bin_sizes / 2, centroids - bin_sizes / 2)
        return self.rng.uniform(low, high)

    def sample(self, batch_size, low=None, high=None):
        cgf_centroid, inds = self.sample_bins(batch_size, low=low, high=high)
        return (np.stack([self.sample_uniform_from_cell(v_range) for v_range in cgf_centroid]), inds)

class RewardThresholdCurriculum(Curriculum):

    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)
        self.episode_reward_lin = np.zeros(len(self))
        self.episode_reward_ang = np.zeros(len(self))
        self.episode_lin_vel_raw = np.zeros(len(self))
        self.episode_ang_vel_raw = np.zeros(len(self))
        self.episode_duration = np.zeros(len(self))

    def get_local_bins(self, bin_inds, ranges=0.1):
        if isinstance(ranges, float):
            ranges = np.ones(self.grid.shape[0]) * ranges
        bin_inds = bin_inds.reshape(-1)
        adjacent_inds = np.logical_and(self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) >= self.grid[:, bin_inds, None] - ranges.reshape(-1, 1, 1), self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) <= self.grid[:, bin_inds, None] + ranges.reshape(-1, 1, 1)).all(axis=0)
        return adjacent_inds

    def update(self, bin_inds, task_rewards, success_thresholds, local_range=0.5):
        is_success = 1.0
        for task_reward, success_threshold in zip(task_rewards, success_thresholds):
            is_success = is_success * (task_reward > success_threshold).cpu()
        if len(success_thresholds) == 0:
            is_success = np.array([False] * len(bin_inds))
        else:
            is_success = is_success.bool().numpy()
        self.weights[bin_inds[is_success]] = np.clip(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
        adjacents = self.get_local_bins(bin_inds[is_success], ranges=local_range)
        for adjacent in adjacents:
            adjacent_inds = np.array(adjacent.nonzero()[0])
            self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + 0.2, 0, 1)

class Rewards:

    def __init__(self, env):
        self.env: object = env

    def _reward_arm_control_limits(self):
        out_of_limits = -(self.env.plan_actions[:, 0] - self.env.cfg.commands.limit_body_pitch[0]).clip(max=0.0)
        out_of_limits += (self.env.plan_actions[:, 0] - self.env.cfg.commands.limit_body_pitch[1]).clip(min=0.0)
        out_of_limits += -(self.env.plan_actions[:, 1] - self.env.cfg.commands.limit_body_roll[0]).clip(max=0.0)
        out_of_limits += (self.env.plan_actions[:, 1] - self.env.cfg.commands.limit_body_roll[1]).clip(min=0.0)
        return out_of_limits

    def _reward_arm_control_smoothness_1(self):
        diff = torch.square(self.env.plan_actions - self.env.last_plan_actions)
        diff = diff * (self.env.last_plan_actions != 0)
        return torch.sum(diff, dim=1)

    def _reward_arm_energy(self):
        energy_sum = torch.sum(torch.square(self.env.torques[:, self.env.num_actions_loco:] * self.env.dof_vel[:, self.env.num_actions_loco:]), dim=1)
        return energy_sum

    def _reward_arm_dof_vel(self):
        return torch.sum(torch.square(self.env.dof_vel[..., self.env.num_actions_loco:]), dim=1)

    def _reward_arm_dof_acc(self):
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[..., self.env.num_actions_loco:] / self.env.dt), dim=1)

    def _reward_arm_action_rate(self):
        return torch.sum(torch.square(self.env.last_actions - self.env.actions)[..., self.env.num_actions_loco:], dim=1)

    def _reward_arm_manip_commands_tracking_combine(self):
        lpy = self.env.get_lpy_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        lpy_error = torch.sum(torch.abs(lpy - self.env.commands_arm_obs[:, 0:3]) / self.env.commands_arm_lpy_range, dim=1)
        rpy = self.env.get_alpha_beta_gamma_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        rpy_error = torch.sum(torch.abs(rpy - self.env.target_abg) / self.env.commands_arm_rpy_range, dim=1)
        return torch.exp(-(self.env.cfg.rewards.manip_weight_lpy * lpy_error + self.env.cfg.rewards.manip_weight_rpy * rpy_error))

    def _reward_arm_action_smoothness_1(self):
        diff = torch.square(self.env.joint_pos_target[:, self.env.num_actions_loco:-2] - self.env.last_joint_pos_target[:, self.env.num_actions_loco:-2])
        diff = diff * (self.env.last_actions[:, self.env.num_actions_loco:] != 0)
        return torch.sum(diff, dim=1)

    def _reward_arm_action_smoothness_2(self):
        diff = torch.square(self.env.joint_pos_target[:, self.env.num_actions_loco:-2] - 2 * self.env.last_joint_pos_target[:, self.env.num_actions_loco:-2] + self.env.last_last_joint_pos_target[:, self.env.num_actions_loco:-2])
        diff = diff * (self.env.last_actions[:, self.env.num_actions_loco:] != 0)
        diff = diff * (self.env.last_last_actions[:, self.env.num_actions_loco:] != 0)
        return torch.sum(diff, dim=1)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.env.commands_dog[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.env.commands_dog[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_loco_energy(self):
        return torch.sum(torch.square(self.env.torques[:, :self.env.num_actions_loco] * self.env.dof_vel[:, :self.env.num_actions_loco]), dim=1)

    def _reward_hip_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, [0, 3, 6, 9]] ** 2, dim=1)
        return action_l2

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.env.dof_vel[..., :self.env.num_actions_loco]), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[..., :self.env.num_actions_loco] / self.env.dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.env.last_actions - self.env.actions)[..., :self.env.num_actions_loco], dim=1)

    def _reward_collision(self):
        return torch.sum(1.0 * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_jump(self):
        reference_heights = 0
        body_height = self.env.base_pos[:, 2] - reference_heights
        jump_height_target = 0.0 + self.env.cfg.rewards.base_height_target
        reward = -torch.square(body_height - jump_height_target)
        return reward

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += -(1 - desired_contact[:, i]) * (1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += -(desired_contact[:, i] * (1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        return reward / 4

    def _reward_action_smoothness_1(self):
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actions_loco] - self.env.last_joint_pos_target[:, :self.env.num_actions_loco])
        diff = diff * (self.env.last_actions[:, :self.env.num_actions_loco] != 0)
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actions_loco] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actions_loco] + self.env.last_last_joint_pos_target[:, :self.env.num_actions_loco])
        diff = diff * (self.env.last_actions[:, :self.env.num_actions_loco] != 0)
        diff = diff * (self.env.last_last_actions[:, :self.env.num_actions_loco] != 0)
        return torch.sum(diff, dim=1)

    def _reward_feet_slip(self):
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        self.env.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip(self.env.foot_indices * 2.0 - 1.0, 0.0, 1.0) * 2.0)
        foot_height = self.env.foot_positions[:, :, 2].view(self.env.num_envs, -1)
        target_height = 0.04 * phases + 0.02
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_collision(self):
        return torch.sum(1.0 * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_orientation_heuristic(self):
        guide = torch.zeros_like(self.env.pitch)
        down_flag = self.env.delta_z < -self.env.cfg.hybrid.rewards.headupdown_thres
        up_flag = self.env.delta_z > self.env.cfg.hybrid.rewards.headupdown_thres + 0.3
        guide[down_flag] = torch.square(self.env.pitch - 0.4)[down_flag]
        guide[up_flag] = torch.square(self.env.pitch + 0.3)[up_flag]
        return guide

    def _reward_orientation_control(self):
        roll_pitch_commands = self.env.commands_dog[:, 3:5]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1], torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0], torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))
        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)
        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat), cur_footsteps_translated[:, i, :])
        desired_stance_width = 0.3
        desired_ys_nom = torch.tensor([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)
        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)
        phases = torch.abs(1.0 - self.env.foot_indices * 2.0) * 1.0 - 0.5
        frequencies = 3.0
        x_vel_des = self.env.commands_dog[:, 0:1]
        yaw_vel_des = self.env.commands_dog[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies)
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies)
        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset
        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
        return reward

    def _reward_vis_manip_commands_tracking_lpy(self):
        lpy = self.env.get_lpy_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        lpy_error = torch.sum(torch.abs(lpy - self.env.commands_arm_obs[:, 0:3]) / self.env.commands_arm_lpy_range, dim=1)
        return torch.exp(-lpy_error)

    def _reward_vis_manip_commands_tracking_rpy(self):
        rpy = self.env.get_alpha_beta_gamma_in_base_coord(torch.arange(self.env.num_envs, device=self.env.device))
        rpy_error = torch.sum(torch.abs(rpy - self.env.commands_arm_obs[:, 3:6]) / self.env.commands_arm_rpy_range, dim=1)
        return torch.exp(-rpy_error)

class RoboDuetTask:
    """Task-owned commands, clocks, curriculum, reward accumulation and termination.

    Order: bind current physics/controller buffers, advance, check_termination,
    compute_reward, reset task/physics for done IDs, build observations. The caller
    updates controller histories AFTER reward. No physics values are fabricated.
    """
    def __init__(self, config, num_envs, device="cpu", dt=None):
        config = deepcopy(config)
        stage = config.get("stage", {})
        cfg = config.get("Cfg", config)
        self.cfg = _namespace(cfg)
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dt = dt if dt is not None else self.cfg.sim.dt * self.cfg.control.decimation
        self.cfg.env.max_episode_length = int(np.ceil(self.cfg.env.episode_length_s / self.dt))
        self.max_episode_length = self.cfg.env.max_episode_length
        self.num_actions_loco = self.cfg.dog.num_actions_loco
        self.num_actions_arm = self.cfg.arm.num_actions_arm
        self.num_actions = self.cfg.hybrid.num_actions
        self.num_actuated_dof = self.num_actions
        self.num_dof = self.num_actions + 2
        self.obs_scales = self.cfg.obs_scales
        self.count = 0
        self.switch_open = False
        self.pretrained_to_hybrid_start = stage.get("pretrained_to_hybrid_start", 10000)
        self.pretrained_to_hybrid_end = stage.get("pretrained_to_hybrid_end", self.pretrained_to_hybrid_start)
        self.lr_down = 1 - 1/(1 + np.exp(-np.linspace(-7, 7, self.pretrained_to_hybrid_end-self.pretrained_to_hybrid_start)))
        self._pretrained_rewards = deepcopy(vars(self.cfg.rewards))
        self.pretrained_reward_scales = dict(vars(self.cfg.reward_scales))
        self.hybrid_reward_scales = dict(vars(self.cfg.hybrid.reward_scales))
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        self.common_step_counter = 0
        self.measured_heights = 0
        self._state_names = []
        for name, width in {
            "commands_dog": self.cfg.dog.dog_num_commands,
            "commands_arm": self.cfg.arm.arm_num_commands,
            "commands_arm_obs": self.cfg.arm.arm_num_commands,
            "target_abg": 3, "obj_quats": 4, "visual_rpy": 3,
            "obj_pose_in_ee": 3, "obj_abg_in_ee": 3,
            "gait_indices": None, "T_trajs": None,
            "desired_contact_states": 4, "foot_indices": 4,
            "clock_inputs": 4, "doubletime_clock_inputs": 4, "halftime_clock_inputs": 4,
            "rew_buf_dog": None, "rew_buf_arm": None,
            "rew_buf_pos_dog": None, "rew_buf_neg_dog": None,
            "rew_buf_pos_arm": None, "rew_buf_neg_arm": None,
        }.items():
            setattr(self, name, torch.zeros((num_envs,) if width is None else (num_envs,width), device=self.device))
            self._state_names.append(name)
        for name in ("episode_length_buf", "arm_time_buf"):
            setattr(self, name, torch.zeros(num_envs, device=self.device, dtype=torch.long))
            self._state_names.append(name)
        self.last_contacts = torch.zeros(num_envs,4,device=self.device,dtype=torch.bool)
        self._state_names.append("last_contacts")
        self.forward_vec = torch.tensor([1.,0.,0.], device=self.device).repeat(num_envs,1)
        self.gravity_vec = torch.tensor([0.,0.,-1.], device=self.device).repeat(num_envs,1)
        self.commands_arm_lpy_range = torch.tensor([getattr(self.cfg.arm.commands,k)[1]-getattr(self.cfg.arm.commands,k)[0] for k in ("l","p","y")], device=self.device).reshape(1,3)
        self.commands_arm_rpy_range = torch.tensor([getattr(self.cfg.arm.commands,k)[1]-getattr(self.cfg.arm.commands,k)[0] for k in ("roll_ee","pitch_ee","yaw_ee")], device=self.device).reshape(1,3)
        self._prepare_reward_function()
        self._init_command_distribution(torch.arange(num_envs,device=self.device))

    def update_physics(self, **state):
        """Bind current real physics and controller tensors, using upstream names."""
        for name, value in state.items():
            setattr(self, name, value)
        if "root_states" in state:
            self.base_pos = self.root_states[:, :3]
            self.base_quat = self.root_states[:, 3:7]
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def set_iteration(self, iteration, switch_open=None):
        self.count = int(iteration)
        self.set_stage((self.count > self.pretrained_to_hybrid_start) if switch_open is None else switch_open)

    def set_stage(self, stage):
        """Release flag is independent of reward count, as in the source runner."""
        if isinstance(stage, str):
            stage = {"stage1": False, "stage2": True, "pretrained": False, "hybrid": True}[stage.lower()]
        self.switch_open = bool(stage)
        self.cfg.rewards = _namespace(self._pretrained_rewards)
        if self.switch_open:
            vars(self.cfg.rewards).update(vars(self.cfg.hybrid.rewards))

    def get_reward_scales(self):
        if self.count < self.pretrained_to_hybrid_start:
            return self.pretrained_reward_scales
        if self.count < self.pretrained_to_hybrid_end:
            lr = self.lr_down[self.count-self.pretrained_to_hybrid_start]
            return {k:self.pretrained_reward_scales[k]*lr+v*(1-lr) for k,v in self.hybrid_reward_scales.items()}
        return self.hybrid_reward_scales

    def advance(self):
        self.episode_length_buf += 1
        self.arm_time_buf += 1
        self.common_step_counter += 1
        # The source applies integer remainder with zero T_trajs before releasing
        # the arm (CUDA behavior is not portable to CPU). There are no arm targets
        # to resample while closed. On release sample all still-uninitialized IDs.
        if self.switch_open:
            periods = (self.T_trajs / self.dt).long()
            ids = ((periods == 0) | (self.arm_time_buf % periods.clamp(min=1) == 0)).nonzero().flatten()
            self._resample_arm_commands(ids)
        ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time/self.dt) == 0).nonzero().flatten()
        self._resample_commands(ids)
        self._step_contact_targets()

    def reset(self, env_ids):
        """Call BEFORE resetting physical state, matching upstream reset_idx."""
        self._resample_commands(env_ids)
        self._resample_arm_commands(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.gait_indices[env_ids] = 0
        episode = {k: v[env_ids].clone() for k,v in self.episode_sums.items()}
        for value in self.episode_sums.values():
            value[env_ids] = 0
        return episode

    def state_dict(self):
        """Caller checkpoints physics/controller and global torch/numpy RNG too."""
        return {"tensors": {k:getattr(self,k).clone() for k in self._state_names},
                "count":self.count, "switch_open":self.switch_open,
                "common_step_counter":self.common_step_counter,
                "episode_sums":{k:v.clone() for k,v in self.episode_sums.items()},
                "command_sums":{k:v.clone() for k,v in self.command_sums.items()},
                "env_command_bins":self.env_command_bins.copy(),
                "env_command_categories":self.env_command_categories.copy(),
                "curriculum_weights":self.curricula[0].weights.copy(),
                "curriculum_rng":deepcopy(self.curricula[0].rng.get_state())}

    def load_state_dict(self, state):
        for k,v in state["tensors"].items():
            getattr(self,k).copy_(v.to(self.device))
        self.set_iteration(state["count"], state["switch_open"])
        self.common_step_counter = state["common_step_counter"]
        for key in ("episode_sums", "command_sums"):
            for k,v in state[key].items():
                getattr(self,key)[k].copy_(v.to(self.device))
        self.env_command_bins[:] = state["env_command_bins"]
        self.env_command_categories[:] = state["env_command_categories"]
        self.curricula[0].weights[:] = state["curriculum_weights"]
        self.curricula[0].rng.set_state(state["curriculum_rng"])

    def check_termination(self):
        """ Check if environments need to be reset
            """
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
        self.reset_buf |= self.time_out_buf
        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) < self.cfg.rewards.terminal_body_height
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)
        self.reverse_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        rpy = quaternion_to_rpy(self.base_quat)
        self.roll, self.pitch, self.y = (rpy[:, 0], rpy[:, 1], rpy[:, 2])
        if self.switch_open and self.cfg.hybrid.rewards.use_terminal_roll:
            reverse_buf1 = torch.logical_and(self.roll > self.cfg.hybrid.rewards.terminal_body_roll, self.commands_arm[:, 2] > 0.0)
            reverse_buf2 = torch.logical_and(self.roll < -self.cfg.hybrid.rewards.terminal_body_roll, self.commands_arm[:, 2] < 0.0)
            self.reverse_buf |= reverse_buf1 | reverse_buf2
        p_align = self.commands_arm[:, 1]
        l_align = self.commands_arm[:, 0]
        self.delta_z = l_align * torch.sin(p_align) + 0.38 - self.base_pos[:, 2]
        if self.switch_open and self.cfg.hybrid.rewards.use_terminal_pitch:
            reverse_buf3 = torch.logical_and(self.pitch < -self.cfg.hybrid.rewards.terminal_body_pitch, self.delta_z < -self.cfg.hybrid.rewards.headupdown_thres)
            reverse_buf4 = torch.logical_and(self.pitch > self.cfg.hybrid.rewards.terminal_body_pitch, self.delta_z > self.cfg.hybrid.rewards.headupdown_thres)
            self.reverse_buf |= reverse_buf3 | reverse_buf4
        if self.switch_open:
            time_exceed_half = self.arm_time_buf / (self.T_trajs / self.dt) > 0.6
            self.reverse_buf = self.reverse_buf & time_exceed_half
            self.reset_buf |= self.reverse_buf
        return self.reset_buf, self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
                Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
                adds each terms to the episode sums and to the total reward
            """
        reward_scales = self.get_reward_scales()
        self.rew_buf_dog[:] = 0.0
        self.rew_buf_pos_dog[:] = 0.0
        self.rew_buf_neg_dog[:] = 0.0
        self.rew_buf_arm[:] = 0.0
        self.rew_buf_pos_arm[:] = 0.0
        self.rew_buf_neg_arm[:] = 0.0
        for i in range(len(reward_scales)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * reward_scales[name]
            if name in ['vis_manip_commands_tracking_lpy', 'vis_manip_commands_tracking_rpy']:
                self.episode_sums[name] += rew
                continue
            self.rew_buf_dog += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos_dog += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg_dog += rew
            self.episode_sums[name] += rew
            if not name in ['tracking_lin_vel', 'tracking_ang_vel']:
                self.rew_buf_arm += rew
                if torch.sum(rew) >= 0:
                    self.rew_buf_pos_arm += rew
                elif torch.sum(rew) <= 0:
                    self.rew_buf_neg_arm += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf_dog[:] = torch.clip(self.rew_buf_dog[:], min=0.0)
            self.rew_buf_arm[:] = torch.clip(self.rew_buf_arm[:], min=0.0)
        elif self.cfg.rewards.only_positive_rewards_ji22_style:
            self.rew_buf_dog[:] = self.rew_buf_pos_dog[:] * torch.exp(self.rew_buf_neg_dog[:] / self.cfg.rewards.sigma_rew_neg)
            self.rew_buf_arm[:] = self.rew_buf_pos_arm[:] * torch.exp(self.rew_buf_neg_arm[:] / self.cfg.rewards.sigma_rew_neg)
        self.episode_sums['total'] += self.rew_buf_dog
        if 'termination' in reward_scales:
            rew = self.reward_container._reward_termination() * reward_scales['termination']
            self.rew_buf_dog += rew
            self.rew_buf_arm += rew
            self.episode_sums['termination'] += rew
            self.command_sums['termination'] += rew
        self.command_sums['lin_vel_raw'] += self.base_lin_vel[:, 0]
        self.command_sums['ang_vel_raw'] += self.base_ang_vel[:, 2]
        self.command_sums['lin_vel_residual'] += (self.base_lin_vel[:, 0] - self.commands_dog[:, 0]) ** 2
        self.command_sums['ang_vel_residual'] += (self.base_ang_vel[:, 2] - self.commands_dog[:, 2]) ** 2
        self.command_sums['ep_timesteps'] += 1
        return self.rew_buf_dog, self.rew_buf_arm

    def quat_to_angle(self, quat):
        quat = quat.to(self.device)
        y_vector = to_torch([0.0, 1.0, 0.0], device=self.device).repeat((quat.shape[0], 1))
        z_vector = to_torch([0.0, 0.0, 1.0], device=self.device).repeat((quat.shape[0], 1))
        x_vector = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((quat.shape[0], 1))
        roll_vec = quat_apply(quat, y_vector)
        alpha = torch.atan2(roll_vec[:, 2], roll_vec[:, 1])
        pitch_vec = quat_apply(quat, z_vector)
        beta = torch.atan2(pitch_vec[:, 0], pitch_vec[:, 2])
        yaw_vec = quat_apply(quat, x_vector)
        gamma = torch.atan2(yaw_vec[:, 1], yaw_vec[:, 0])
        return torch.stack([alpha, beta, gamma], dim=-1)

    def get_lpy_in_base_coord(self, env_ids):
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        self.grasper_move = torch.tensor([0.1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        self.grasper_move_in_world = quat_rotate(self.end_effector_state[env_ids, 3:7], self.grasper_move)
        self.grasper_in_world = self.end_effector_state[env_ids, :3] + self.grasper_move_in_world
        x = torch.cos(yaw) * (self.grasper_in_world[:, 0] - self.root_states[env_ids, 0]) + torch.sin(yaw) * (self.grasper_in_world[:, 1] - self.root_states[env_ids, 1])
        y = -torch.sin(yaw) * (self.grasper_in_world[:, 0] - self.root_states[env_ids, 0]) + torch.cos(yaw) * (self.grasper_in_world[:, 1] - self.root_states[env_ids, 1])
        z = torch.mean(self.grasper_in_world[:, 2].unsqueeze(1) - self.measured_heights, dim=1) - 0.38
        l = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        p = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2))
        y_aw = torch.atan2(y, x)
        return torch.stack([l, p, y_aw], dim=-1)

    def get_alpha_beta_gamma_in_base_coord(self, env_ids):
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        ee_in_base_quats = quat_mul(quat_conjugate(base_quats), self.end_effector_state[:, 3:7])
        abg = self.quat_to_angle(ee_in_base_quats)
        return abg

    def _lpy_to_world_xyz(self, env_ids):
        l = self.commands_arm[env_ids, 0]
        p = self.commands_arm[env_ids, 1]
        y = self.commands_arm[env_ids, 2]
        x = l * torch.cos(p) * torch.cos(y)
        y = l * torch.cos(p) * torch.sin(y)
        z = l * torch.sin(p)
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        x_ = x * torch.cos(yaw) - y * torch.sin(yaw) + self.root_states[env_ids, 0]
        y_ = x * torch.sin(yaw) + y * torch.cos(yaw) + self.root_states[env_ids, 1]
        z_ = z + self.measured_heights + 0.38
        return (x_, y_, z_)

    def _get_object_pose_in_ee(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        x, y, z = self._lpy_to_world_xyz(env_ids)
        xyz = torch.stack([x, y, z], dim=-1)
        dxyz = xyz - self.end_effector_state[:, 0:3]
        self.obj_pose_in_ee[:] = quat_apply(quat_conjugate(self.end_effector_state[:, 3:7]), dxyz)
        return self.obj_pose_in_ee[:]

    def _get_object_abg_in_ee(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        base_quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        rot_in_world = quat_mul(base_quats, self.obj_quats)
        rot_in_ee = quat_mul(quat_conjugate(self.end_effector_state[:, 3:7]), rot_in_world)
        self.obj_abg_in_ee[:] = self.quat_to_angle(rot_in_ee)
        return self.obj_abg_in_ee[:]

    def _resample_arm_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        if not self.switch_open:
            return
        self.commands_arm[env_ids, 0] = torch_rand_float(self.cfg.arm.commands.l[0], self.cfg.arm.commands.l[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        self.commands_arm[env_ids, 1] = torch_rand_float(self.cfg.arm.commands.p[0], self.cfg.arm.commands.p[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        self.commands_arm[env_ids, 2] = torch_rand_float(self.cfg.arm.commands.y[0], self.cfg.arm.commands.y[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        self.commands_arm_obs[env_ids, 0] = self.commands_arm[env_ids, 0]
        self.commands_arm_obs[env_ids, 1] = self.commands_arm[env_ids, 1]
        self.commands_arm_obs[env_ids, 2] = self.commands_arm[env_ids, 2]
        roll = torch_rand_float(self.cfg.arm.commands.roll_ee[0], self.cfg.arm.commands.roll_ee[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        pitch = torch_rand_float(self.cfg.arm.commands.pitch_ee[0], self.cfg.arm.commands.pitch_ee[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        yaw = torch_rand_float(self.cfg.arm.commands.yaw_ee[0], self.cfg.arm.commands.yaw_ee[1], (env_ids.shape[0], 1), device=self.device).squeeze()
        zero_vec = torch.zeros_like(roll)
        q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
        q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
        q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
        quats = quat_mul(q1, quat_mul(q2, q3))
        self.obj_quats[env_ids] = quats.reshape(-1, 4)
        assert torch.allclose(torch.norm(self.obj_quats[env_ids], dim=1), torch.ones(len(env_ids)).to(self.device), atol=1e-05), 'quats is not unit vector.'
        if self.cfg.hybrid.use_vision:
            self._get_object_pose_in_ee()
            self._get_object_abg_in_ee()
        self.visual_rpy[env_ids] = quaternion_to_rpy(self.obj_quats[env_ids]).to(self.device)
        self.target_abg[env_ids] = self.quat_to_angle(self.obj_quats[env_ids]).to(self.device)
        if self.cfg.use_rot6d:
            r6d = _rotation6d(quats)
            self.commands_arm_obs[env_ids, 3:9] = r6d.to(self.device)
        else:
            rpy = self.quat_to_angle(self.obj_quats[env_ids]).to(self.device)
            self.commands_arm_obs[env_ids, 3] = rpy[:, 0]
            self.commands_arm_obs[env_ids, 4] = rpy[:, 1]
            self.commands_arm_obs[env_ids, 5] = rpy[:, 2]
        self._resample_Traj_commands(env_ids)

    def _resample_Traj_commands(self, env_ids):
        time_range = (self.cfg.arm.commands.T_traj[1] - self.cfg.arm.commands.T_traj[0]) / self.dt
        time_interval = torch.from_numpy(np.random.choice(int(time_range + 1), len(env_ids))).to(self.device)
        self.T_trajs[env_ids] = torch.ones_like(self.T_trajs[env_ids]) * self.cfg.arm.commands.T_traj[0] + time_interval * self.dt
        self.arm_time_buf[env_ids] = torch.zeros_like(self.arm_time_buf[env_ids])

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)
        curriculum = self.curricula[0]
        task_rewards, success_thresholds = ([], [])
        for key in ['tracking_lin_vel', 'tracking_ang_vel', 'tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
            if key in self.command_sums.keys():
                task_rewards.append(self.command_sums[key][env_ids] / ep_len)
                success_thresholds.append(self.curriculum_thresholds[key] * self.pretrained_reward_scales[key])
        old_bins = self.env_command_bins[env_ids.cpu().numpy()]
        if len(success_thresholds) > 0:
            curriculum.update(old_bins, task_rewards, success_thresholds, local_range=np.array([0.55, 0.55, 0.55, 1.0, 1.0]))
        new_commands, new_bin_inds = curriculum.sample(batch_size=len(env_ids))
        self.env_command_bins[env_ids.cpu().numpy()] = new_bin_inds
        self.env_command_categories[env_ids.cpu().numpy()] = 0
        if not self.cfg.hybrid.plan_vel:
            self.commands_dog[env_ids, 0] = torch.Tensor(new_commands[:, 0]).to(self.device)
            self.commands_dog[env_ids, 1] = torch.Tensor(new_commands[:, 1]).to(self.device)
            self.commands_dog[env_ids, 2] = torch.Tensor(new_commands[:, 2]).to(self.device)
            zero_mask = torch.rand(len(env_ids)) < 0.1
            if len(zero_mask.nonzero()) > 0:
                self.commands_dog[env_ids[zero_mask], :3] = 0
            self.commands_dog[env_ids, 0] *= torch.abs(self.commands_dog[env_ids, 0]) > 0.07
            self.commands_dog[env_ids, 1] *= torch.abs(self.commands_dog[env_ids, 1]) > 0.07
            self.commands_dog[env_ids, 2] *= torch.abs(self.commands_dog[env_ids, 2]) > 0.1
        elif not self.switch_open:
            self.commands_dog[env_ids, 0] = torch.Tensor(new_commands[:, 0]).to(self.device)
            self.commands_dog[env_ids, 1] = torch.Tensor(new_commands[:, 1]).to(self.device)
            self.commands_dog[env_ids, 2] = torch.Tensor(new_commands[:, 2]).to(self.device)
            zero_mask = torch.rand(len(env_ids)) < 0.1
            if len(zero_mask.nonzero()) > 0:
                self.commands_dog[env_ids[zero_mask], :3] = 0
            self.commands_dog[env_ids, 0] *= torch.abs(self.commands_dog[env_ids, 0]) > 0.07
            self.commands_dog[env_ids, 1] *= torch.abs(self.commands_dog[env_ids, 1]) > 0.07
            self.commands_dog[env_ids, 2] *= torch.abs(self.commands_dog[env_ids, 2]) > 0.1
        if not self.switch_open:
            self.commands_dog[env_ids, 3] = torch.Tensor(new_commands[:, 3]).to(self.device)
            self.commands_dog[env_ids, 4] = torch.Tensor(new_commands[:, 4]).to(self.device)
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.0

    def _init_command_distribution(self, env_ids):
        self.category_names = ['trot']
        if self.cfg.commands.curriculum_type == 'RewardThresholdCurriculum':
            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed, x_vel=(self.cfg.commands.limit_vel_x[0], self.cfg.commands.limit_vel_x[1], self.cfg.commands.num_bins_vel_x), y_vel=(self.cfg.commands.limit_vel_y[0], self.cfg.commands.limit_vel_y[1], self.cfg.commands.num_bins_vel_y), yaw_vel=(self.cfg.commands.limit_vel_yaw[0], self.cfg.commands.limit_vel_yaw[1], self.cfg.commands.num_bins_vel_yaw), body_pitch=(self.cfg.commands.limit_body_pitch[0], self.cfg.commands.limit_body_pitch[1], self.cfg.commands.num_bins_body_pitch), body_roll=(self.cfg.commands.limit_body_roll[0], self.cfg.commands.limit_body_roll[1], self.cfg.commands.num_bins_body_roll))]
        if self.cfg.commands.curriculum_type == 'LipschitzCurriculum':
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold, binary_phases=self.cfg.commands.binary_phases)
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int64)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int64)
        low = np.array([self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0], self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.body_pitch_range[0], self.cfg.commands.body_roll_range[0]])
        high = np.array([self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1], self.cfg.commands.ang_vel_yaw[1], self.cfg.commands.body_pitch_range[1], self.cfg.commands.body_roll_range[1]])
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
                Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
            """
        self.reward_container = Rewards(self)
        for key in list(self.pretrained_reward_scales.keys()):
            scale = self.pretrained_reward_scales[key]
            if scale == 0:
                self.pretrained_reward_scales.pop(key)
            else:
                self.pretrained_reward_scales[key] *= self.dt
        for key in list(self.hybrid_reward_scales.keys()):
            self.hybrid_reward_scales[key] *= self.dt
        for name, scale in self.pretrained_reward_scales.items():
            if name not in self.hybrid_reward_scales:
                self.hybrid_reward_scales[name] = scale
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.hybrid_reward_scales.items():
            if name == 'termination':
                continue
            if not hasattr(self.reward_container, '_reward_' + name):
                raise ValueError(f"Missing upstream reward: {name}")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))
                if name not in self.pretrained_reward_scales:
                    self.pretrained_reward_scales[name] = 0.0
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) for name in self.hybrid_reward_scales.keys()}
        self.episode_sums['total'] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums_eval = {name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) for name in self.hybrid_reward_scales.keys()}
        self.episode_sums_eval['total'] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.command_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) for name in list(self.hybrid_reward_scales.keys()) + ['lin_vel_raw', 'ang_vel_raw', 'lin_vel_residual', 'ang_vel_residual', 'ep_timesteps']}


    def _step_contact_targets(self):
        frequencies = 3.0
        gaits = {'pronking': [0, 0, 0], 'trotting': [0.5, 0, 0], 'bounding': [0, 0.5, 0], 'pacing': [0, 0, 0.5]}
        phases, offsets, bounds = gaits['trotting']
        durations = 0.5
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
        if self.cfg.commands.pacing_offset:
            foot_indices = [self.gait_indices + phases + offsets + bounds, self.gait_indices + bounds, self.gait_indices + offsets, self.gait_indices + phases]
        else:
            foot_indices = [self.gait_indices + phases + offsets + bounds, self.gait_indices + offsets, self.gait_indices + bounds, self.gait_indices + phases]
        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
        for idxs in foot_indices:
            idxs[torch.norm(self.commands_dog[:, :3], dim=1) < 0.1] = 0.25
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations
            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations) * (0.5 / (1 - durations))
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])
        self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
        self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
        self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])
        self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
        self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
        self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])
        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf
        smoothing_multiplier_FL = smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) + smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1))
        smoothing_multiplier_FR = smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) + smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5 - 1))
        smoothing_multiplier_RL = smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) + smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5 - 1))
        smoothing_multiplier_RR = smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) + smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5 - 1))
        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR
        footswing_height_cmd = 0.04
        self.desired_footswing_height = footswing_height_cmd
