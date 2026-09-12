"""Released LFC lifecycle using official Lab state and position-drive I/O.

The task's original tensor equations are retained in lifecycle/rewards/sensors.
Runtime performs real physics only. No mock simulation or reward substitution.
"""
from copy import deepcopy
import numpy as np
import torch
from .config import resolve_config
from .lifecycle import LFCLifecycle
from .torch_math import quat_rotate_inverse
from .logger import Logger
from ..history import ObservationHistory


class LFCIsaacTrainingEnv(LFCLifecycle):
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, deepcopy(config)
        self.cfg = resolve_config(config)
        if self.cfg.env.add_balls or self.cfg.env.force_control_init_poses or self.cfg.commands.force_control:
            raise ValueError('This port implements the released default hybrid task, without optional object/NPZ tasks')
        if self.cfg.control.control_type != 'P' or self.cfg.asset.default_dof_drive_mode != 1:
            raise ValueError('The released path uses P targets and PhysX position drives')
        self.device, self.num_envs = runtime.device, runtime.num_envs
        if self.cfg.env.num_envs != self.num_envs:
            raise ValueError('Runtime and task environment counts differ')
        self.num_train_envs = self.num_envs
        self.num_obs, self.num_privileged_obs, self.num_actions = self.cfg.env.num_observations, self.cfg.env.num_privileged_obs, self.cfg.env.num_actions
        self.num_obs_history = self.num_obs*self.cfg.env.num_observation_history
        self.sim_params = self.cfg.sim
        self._parse_cfg(self.cfg)
        self.num_dof = self.num_dofs = self.num_actuated_dof = len(runtime.joint_names)
        self.dof_names, self.body_names = list(runtime.joint_names), list(runtime.body_names)
        self.num_bodies = self.total_rigid_body_num = len(self.body_names)
        self.num_object_bodies = 0
        self.robot_actor_idxs = torch.arange(self.num_envs, device=self.device)
        self.gripper_stator_index, self.robot_base_index = runtime.ee_id, runtime.base_id
        self.ee_measurement_index = runtime.ee_measurement_id
        self.feet_indices = torch.tensor(runtime.feet_indices, device=self.device, dtype=torch.long)
        self.penalised_contact_indices = torch.tensor([i for part in self.cfg.asset.penalize_contacts_on for i,name in enumerate(self.body_names) if part in name], device=self.device, dtype=torch.long)
        self.termination_contact_indices = torch.tensor([i for part in self.cfg.asset.terminate_after_contacts_on for i,name in enumerate(self.body_names) if part in name], device=self.device, dtype=torch.long)
        self.dof_pos_limits = runtime.dof_pos_limits.clone()
        mean = self.dof_pos_limits.mean(-1)
        extent = (self.dof_pos_limits[:,1]-self.dof_pos_limits[:,0])*self.cfg.rewards.soft_dof_pos_limit*.5
        self.dof_pos_limits[:,0],self.dof_pos_limits[:,1] = mean-extent,mean+extent
        self.dof_vel_limits,self.torque_limits = runtime.dof_vel_limits,runtime.torque_limits
        self.base_init_state = torch.tensor(self.cfg.init_state.pos+self.cfg.init_state.rot+self.cfg.init_state.lin_vel+self.cfg.init_state.ang_vel,device=self.device)
        self.terrain = self.terrain_obj = runtime.terrain
        self.height_samples = runtime.terrain.height_samples
        self.friction_samples = runtime.terrain.friction_samples
        self.terrain.cfg.env_width = self.cfg.terrain.terrain_width
        self.terrain.cfg.env_length = self.cfg.terrain.terrain_length
        self.cfg.terrain.env_origins = runtime.terrain.env_origins.cpu().numpy().copy()
        self.env_origins = torch.zeros(self.num_envs,3,device=self.device)
        self.terrain_levels = torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.terrain_types = torch.zeros_like(self.terrain_levels)
        self.terrain_origins = torch.zeros_like(self.env_origins)
        self._get_env_origins(self.robot_actor_idxs,self.cfg)
        self.up_axis_idx = 2
        self.initial_dynamics_dict = None
        self.init_done = False
        self.headless = True
        self.record_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0
        self.forces_deactivated = False
        self.first_sim_time_step = True
        self.complete_video_frames = None
        self.video_frames = []
        self.transition_observer = None
        self.substep_observer = None
        self.default_friction = getattr(runtime,'default_friction',1.0)
        self.default_restitution = getattr(runtime,'default_restitution',0.0)
        for name in ('obs_buf','privileged_obs_buf'):
            width = self.num_obs if name == 'obs_buf' else self.num_privileged_obs
            setattr(self,name,torch.zeros(self.num_envs,width,device=self.device))
        for name in ('rew_buf','rew_buf_pos','rew_buf_neg'):
            setattr(self,name,torch.zeros(self.num_envs,device=self.device))
        self.episode_length_buf = torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        for name in ('reset_buf','time_out_buf','contact_buf','body_height_buf','body_ori_buf','legs_torque_lim_buff','arm_torque_lim_buff'):
            setattr(self,name,torch.zeros(self.num_envs,device=self.device,dtype=torch.bool))
        self.extras = {}
        self._bind_state(runtime.read_state(), initial=True)
        self._init_custom_buffers__()
        self._randomize_rigid_body_props(self.robot_actor_idxs,self.cfg)
        self._apply_rigid_properties()
        self._randomize_gravity()
        self._randomize_ball_drag()
        self._init_command_distribution(self.robot_actor_idxs)
        self.lag_buffer = [torch.zeros_like(self.dof_pos) for _ in range(self.cfg.domain_rand.lag_timesteps+1)]
        self._init_buffers()
        self.initialize_sensors()
        self._prepare_reward_function()
        self.logger = Logger(self)
        self.history = ObservationHistory(self.num_envs,self.num_obs,self.cfg.env.num_observation_history,self.cfg.env.history_frame_skip,self.device)
        self.init_done = True
        self.global_steps = 0

    def _bind_state(self, state, initial=False):
        fields = {'root_states':'root_states','dof_pos':'dof_pos','dof_vel':'dof_vel','rigid_body_state':'rigid_body_state','contact_forces':'contact_forces','measured_dof_force':'dof_force'}
        for target,key in fields.items():
            value = state[key]
            if initial:
                setattr(self,target,value.clone())
            else:
                getattr(self,target).copy_(value)
        if initial:
            self.base_pos = self.root_states[:,0:3].clone()
            self.base_quat = self.root_states[:,3:7].clone()
            self.base_lin_vel = quat_rotate_inverse(self.base_quat,self.root_states[:,7:10])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat,self.root_states[:,10:13])
            self.foot_positions = self.rigid_body_state[:,self.feet_indices,:3]
            self.foot_velocities = self.rigid_body_state[:,self.feet_indices,7:10]
            self.gripper_position = self.rigid_body_state[:,self.gripper_stator_index,:3]
            self.gripper_velocity = self.rigid_body_state[:,self.gripper_stator_index,7:10]
            self.prev_base_pos = self.base_pos.clone()
            self.prev_foot_velocities = self.foot_velocities.clone()
            self.net_contact_forces = self.contact_forces.reshape(-1,3)

    def _apply_rigid_properties(self):
        self.runtime.set_material_properties(self.friction_coeffs[:,0],self.restitutions[:,0])
        self.runtime.set_base_com(self.com_displacements)
        if self.cfg.domain_rand.randomize_base_mass:
            self.runtime.set_base_mass_delta(self.payloads)

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        # Runtime applies the source coefficient consistently to all collision shapes.
        self.runtime.set_material_properties(self.friction_coeffs[:,0],self.restitutions[:,0])

    def _randomize_gravity(self, external_force=None):
        if external_force is not None:
            self.gravities[:] = external_force.to(self.device).unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            low,high = self.cfg.domain_rand.gravity_range
            force = torch.rand(3,device=self.device)*(high-low)+low
            self.gravities[:] = force.unsqueeze(0)
        gravity = self.gravities[0]+torch.tensor([0.,0.,-9.8],device=self.device)
        self.gravity_vec[:] = gravity.unsqueeze(0)/torch.norm(gravity)
        self.runtime.set_gravity(gravity)

    def _teleport_robots(self,env_ids,cfg):
        if cfg.terrain.teleport_robots:
            raise ValueError('Released task disables terrain teleportation')

    def get_observations(self):
        return {'obs':self.obs_buf,'privileged_obs':self.privileged_obs_buf,'obs_history':self.history.append(self.obs_buf)}

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        self.reset_idx(self.robot_actor_idxs)
        # Source VelocityTrackingEasyEnv.reset advances one zero-action policy step.
        obs,_,_,_ = self.step(torch.zeros(self.num_envs,self.num_actions,device=self.device))
        self.history.reset_rows(self.robot_actor_idxs)
        return {'obs':obs['obs'],'privileged_obs':obs['privileged_obs'],'obs_history':self.history.history}

    def step(self,actions):
        self.actions = torch.clip(actions,-self.cfg.normalization.clip_actions,self.cfg.normalization.clip_actions).to(self.device)
        self.pre_physics_step()
        for substep in range(self.cfg.control.decimation):
            if self.cfg.commands.control_only_z1:
                self.actions[:,:12] = 0
            self.actions[:,18] = -.1
            self._compute_torques(self.actions)
            if self.first_sim_time_step:
                if torch.any(self.episode_length_buf > 1):
                    self.envs_nb = torch.arange(self.num_envs,device=self.device)
                    self.ee_init_pos_world = self.rigid_body_state[:,self.gripper_stator_index,:3].clone()
                    self.first_sim_time_step = False
            else:
                ids = self.envs_nb[self.episode_length_buf == 1]
                if ids.numel():
                    self.ee_init_pos_world[ids] = self.rigid_body_state[ids,self.gripper_stator_index,:3].clone()
            self._push_gripper(self.robot_actor_idxs,self.cfg)
            self._push_robot_base(self.robot_actor_idxs,self.cfg)
            self._bind_state(self.runtime.step(self.joint_pos_target,world_forces=self.forces))
            if self.substep_observer is not None:
                self.substep_observer(self,substep)
        self.global_steps += self.num_envs
        self.post_physics_step()
        clip = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf,-clip,clip)
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf,-clip,clip)
        self.extras['int'] = (2100-self.compute_energy())*.0000003
        # These transition values must be current even if no environment reset
        # populated the original logger on this step.
        self.extras['time_outs'] = self.time_out_buf.clone()
        self.extras['env_bins'] = torch.as_tensor(self.env_command_bins,device=self.device)
        self.extras['privileged_obs'] = self.privileged_obs_buf
        history = self.history.append(self.obs_buf,self.reset_buf.nonzero(as_tuple=False).flatten())
        return {'obs':self.obs_buf,'privileged_obs':self.privileged_obs_buf,'obs_history':history}, self.rew_buf,self.reset_buf,self.extras

    def training_state(self):
        """Exposed task/physics state at a completed rollout boundary.

        PhysX internal solver/contact caches are not exposed by the public API;
        restoring this state is continuation, not bitwise simulator replay.
        """
        static_tensors = {'height_samples','friction_samples'}
        tensors = {k:v.detach().cpu().clone() for k,v in vars(self).items()
                   if isinstance(v,torch.Tensor) and k not in static_tensors}
        arrays = {k:v.copy() for k,v in vars(self).items() if isinstance(v,np.ndarray)}
        scalar_names = ('common_step_counter','global_steps','first_sim_time_step','init_training','forces_deactivated')
        curricula = []
        for curriculum in self.curricula:
            curricula.append({'arrays':{k:v.copy() for k,v in vars(curriculum).items() if isinstance(v,np.ndarray)},
                              'rng':curriculum.rng.get_state()})
        return {'tensors':tensors,'arrays':arrays,
                'scalars':{k:getattr(self,k) for k in scalar_names},
                'lag_buffer':[v.detach().cpu().clone() for v in self.lag_buffer],
                'history_buffer':self.history.buffer.detach().cpu().clone(),
                'history':self.history.history.detach().cpu().clone(),
                'episode_sums':{k:v.detach().cpu().clone() for k,v in self.episode_sums.items()},
                'command_sums':{k:v.detach().cpu().clone() for k,v in self.command_sums.items()},
                'curricula':curricula,'runtime':self.runtime.training_state()}

    def load_training_state(self, state):
        self.runtime.load_training_state(state['runtime'])
        for name,value in state['tensors'].items():
            current = getattr(self,name,None)
            if isinstance(current,torch.Tensor) and current.shape == value.shape:
                current.copy_(value.to(self.device))
            else:
                setattr(self,name,value.to(self.device).clone())
        for name,value in state['arrays'].items():setattr(self,name,value.copy())
        for name,value in state['scalars'].items():setattr(self,name,value)
        self.lag_buffer = [v.to(self.device).clone() for v in state['lag_buffer']]
        self.history.buffer = state['history_buffer'].to(self.device).clone()
        self.history.history = state['history'].to(self.device).clone()
        self.episode_sums = {k:v.to(self.device).clone() for k,v in state['episode_sums'].items()}
        self.command_sums = {k:v.to(self.device).clone() for k,v in state['command_sums'].items()}
        if len(state['curricula']) != len(self.curricula):
            raise ValueError('Checkpoint curriculum categories differ')
        for curriculum,saved in zip(self.curricula,state['curricula']):
            for name,value in saved['arrays'].items():setattr(curriculum,name,value.copy())
            curriculum.rng.set_state(saved['rng'])
        # task buffers retain the source pre/post-reset timing as saved; runtime
        # restore writes the exposed simulator state without inventing a step.
        self.extras = {}
