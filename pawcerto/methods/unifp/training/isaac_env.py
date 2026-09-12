"""Original UniFP lifecycle bound to measured official Lab state.

Gym tensor acquisition/writes are replaced by B2Z1Isaac. Equations, force
schedules, reset draw ranges, ordering and PPO observation protocol are retained.
"""
import torch
from ..equations import UniFPEquations
from ..history import force_stage_active
from ..math import quat_from_euler_xyz, quat_rotate_inverse, euler_from_quat, torch_rand_float
from .lifecycle import UniFPLifecycle, get_euler_xyz_tensor
from .config import resolve_config


class UniFPIsaacTrainingEnv(UniFPLifecycle, UniFPEquations):
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, config
        self.cfg = resolve_config(config)
        self.device, self.num_envs = runtime.device, runtime.num_envs
        for name, value in [('num_obs',self.cfg.env.num_observations),('num_privileged_obs',self.cfg.env.num_privileged_obs),
                            ('num_pred_obs',self.cfg.env.num_pred_obs),('num_single_obs',self.cfg.env.num_single_obs),
                            ('num_actions',self.cfg.env.num_actions)]:
            setattr(self,name,value)
        self.num_dof = self.num_dofs = len(runtime.joint_names)
        self.dof_names, self.num_bodies = runtime.joint_names, len(runtime.body_names)
        self.sim_params, self.terrain = self.cfg.sim, runtime.terrain
        self._parse_cfg(self.cfg)
        self._get_env_origins()
        self.gripper_idx, self.robot_base_idx = runtime.ee_id, runtime.base_id
        self.feet_indices = torch.tensor(runtime.feet_indices,device=self.device)
        self.thigh_indices = torch.tensor([runtime.body_names.index(leg+'_thigh') for leg in ('FL','FR','RL','RR')],device=self.device)
        self.penalised_contact_indices = torch.tensor([i for part in self.cfg.asset.penalize_contacts_on for i,n in enumerate(runtime.body_names) if part in n],device=self.device,dtype=torch.long)
        self.termination_contact_indices = torch.tensor([i for part in self.cfg.asset.terminate_after_contacts_on for i,n in enumerate(runtime.body_names) if part in n],device=self.device,dtype=torch.long)
        self.hip_indices = torch.tensor([self.dof_names.index(leg+'_hip_joint') for leg in ('FR','FL','RR','RL')],device=self.device)
        self.mass_params_tensor = runtime.mass_params_tensor
        self.friction_coeffs_tensor = runtime.friction_coeffs_tensor
        self.dof_pos_limits = runtime.dof_pos_limits.clone()
        midpoint = self.dof_pos_limits.mean(dim=-1)
        extent = (self.dof_pos_limits[:,1]-self.dof_pos_limits[:,0])*self.cfg.rewards.soft_dof_pos_limit*.5
        self.dof_pos_limits[:,0],self.dof_pos_limits[:,1] = midpoint-extent,midpoint+extent
        self.dof_vel_limits,self.torque_limits = runtime.dof_vel_limits,runtime.torque_limits
        dr = self.cfg.domain_rand
        self.motor_strength = (torch.cat((torch_rand_float(*dr.leg_motor_strength_range,(self.num_envs,12),device=self.device),
            torch_rand_float(*dr.arm_motor_strength_range,(self.num_envs,5),device=self.device)),dim=1)
            if dr.randomize_motor else torch.ones(self.num_envs,17,device=self.device))
        self.base_init_state = torch.tensor(self.cfg.init_state.pos+self.cfg.init_state.rot+self.cfg.init_state.lin_vel+self.cfg.init_state.ang_vel,device=self.device)
        self.play, self.init_done = False, False
        self.episode_length_buf = torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.rew_buf = torch.zeros(self.num_envs,device=self.device)
        self.transition_observer = None
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    @property
    def force_stage_info(self):
        return {'global_steps':self.global_steps,'threshold_policy_steps':self.cfg.commands.force_start_step*24,
                'active':force_stage_active(self.global_steps,self.cfg),
                'injected_gripper_force_max_N':float(self.forces[:,self.gripper_idx].abs().max()),
                'injected_base_force_max_N':float(self.forces[:,self.robot_base_idx].abs().max())}

    def reset(self):
        # BaseTask.reset executes one zero-action policy step after reset_idx.
        self.reset_idx(torch.arange(self.num_envs,device=self.device))
        observations,rewards,_,_ = self.step(torch.zeros(self.num_envs,self.num_actions,device=self.device))
        return observations,rewards

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_pos[env_ids,:12] = self.default_dof_pos[:,:12]*torch_rand_float(.5,1.5,(len(env_ids),12),device=self.device)
        self.dof_pos[env_ids,12:17] += torch_rand_float(-.5,.5,(len(env_ids),self.num_actions-12),device=self.device)
        self.dof_vel[env_ids] = 0.

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids,:3] += self.env_origins[env_ids]
        self.root_states[env_ids,:2] += torch_rand_float(-1.,1.,(len(env_ids),2),device=self.device)
        yaw = self.cfg.init_state.rand_yaw_range*torch_rand_float(-1.,1.,(len(env_ids),1),device=self.device).squeeze(1)
        self.root_states[env_ids,3:7] = quat_from_euler_xyz(0*yaw,0*yaw,yaw)
        self.root_states[env_ids,7:13] = torch_rand_float(-.5,.5,(len(env_ids),6),device=self.device)
        self.runtime.write_reset(env_ids,self.root_states[env_ids],self.dof_pos[env_ids],self.dof_vel[env_ids])

    def _push_robots(self):
        maximum = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:,7:9] = torch_rand_float(-maximum,maximum,(self.num_envs,2),device=self.device)
        self.root_states[:,7:9] = torch.where(self.commands.sum(dim=1).unsqueeze(-1)==0,self.root_states[:,7:9]*2.5,self.root_states[:,7:9])
        self.runtime.write_root_velocity(torch.arange(self.num_envs,device=self.device),self.root_states[:,7:13],self.root_states[:,3:7])

    def step(self, actions):
        self.actions = torch.clip(actions,-self.cfg.normalization.clip_actions,self.cfg.normalization.clip_actions).to(self.device)
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions)
            if force_stage_active(self.global_steps,self.cfg):
                self._push_gripper(torch.arange(self.num_envs,device=self.device))
            self.runtime.step_control(self.torques,self.forces)
            q,qd = self.runtime.joints()
            self.dof_pos[:],self.dof_vel[:] = q,qd
        self.runtime.refresh()
        self.root_states[:] = self.runtime.root_states
        self.rigid_state[:] = self.runtime.rigid_state
        self.contact_forces[:] = self.runtime.contact_forces
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat,self.root_states[:,7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat,self.root_states[:,10:13])
        yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler[:] = torch.cat((torch.zeros(self.num_envs,2,device=self.device),yaw[:,None]),dim=1)
        self.base_yaw_quat[:] = quat_from_euler_xyz(0*yaw,0*yaw,yaw)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat,self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self._post_physics_step_callback()
        self.update_curr_ee_goal()
        self.check_termination()
        self.compute_reward()
        if self.transition_observer is not None:
            self.transition_observer(self)
        self.extras = {'time_outs':self.time_out_buf} if self.cfg.env.send_timeouts else {}
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        self.compute_observations()
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:,7:13]
        self.last_rigid_state[:] = self.rigid_state
        self.last_torques[:] = self.torques
        bound = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf,-bound,bound)
        self.obs_pred = torch.clip(self.obs_pred,-bound,bound)
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf,-bound,bound)
        self.global_steps += 1
        return self.get_observations(),self.rew_buf,self.reset_buf,self.extras
