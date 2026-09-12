"""RoboDuet automatic task wired to the official Isaac Lab physical runtime.

Plane terrain and the default no-vision M controller are the supported source
path. Task observation caches intentionally retain pre-reset body data until
another physics step, matching the original automatic environment.
"""
from copy import deepcopy
import math

import torch

from ..controller import RoboDuetController, plan
from ..observations import RoboDuetObservations, quat_from_euler_xyz
from .task import RoboDuetTask


class RoboDuetIsaacTrainingEnv:
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, deepcopy(config)
        self.cfg = config['Cfg']
        self.device = torch.device(runtime.device)
        self.num_envs = runtime.num_envs
        self.num_train_envs = self.num_envs
        self.stage = 1
        if self.cfg['terrain']['mesh_type'] != 'plane':
            raise ValueError('RoboDuet Lab training currently implements the source plane terrain')
        dr = self.cfg['domain_rand']
        unsupported = [k for k in ('randomize_com_displacement', 'randomize_end_effector_force',
                                    'randomize_rigids_after_start') if dr[k]]
        if unsupported:
            raise ValueError(f'Unsupported nondefault dynamics: {unsupported}')
        self.controller = RoboDuetController(config, runtime.joint_names, runtime.torque_limits, self.device)
        self.observer = RoboDuetObservations(config, self.num_envs, self.device)
        self.task = RoboDuetTask(config, self.num_envs, self.device)
        self.dt = self.controller.dt * self.controller.decimation
        if not math.isclose(runtime.dt, self.controller.dt):
            raise ValueError('Physical dt differs from the configured RoboDuet controller')
        n = self.num_envs
        self.buffers = {name: torch.zeros(n,width,device=self.device) for name,width in {
            'actions':18,'last_actions':18,'last_last_actions':18,
            'joint_pos_target':20,'last_joint_pos_target':20,'last_last_joint_pos_target':20,
            'last_dof_vel':20,'torques':20,'plan_actions':2,'last_plan_actions':2,
            'motor_offsets':20,'payloads':1,'gravities':3}.items()}
        for name,width in [('motor_strengths',20),('Kp_factors',20),('Kd_factors',20),
                           ('friction_coeffs',1),('restitutions',1)]:
            self.buffers[name] = torch.ones(n,width,device=self.device)
        self.buffers['restitutions'].zero_()
        self.gravity_vec = torch.tensor([0.,0.,-1.],device=self.device).repeat(n,1)
        self.feet_indices = torch.tensor([runtime.body_names.index(leg+'_foot')
                                         for leg in ('FL','FR','RL','RR')],device=self.device)
        # Source concatenates matching lists, so duplicate bodies are deliberate.
        self.penalised_contact_indices = torch.tensor([i for fragment in self.cfg['asset']['penalize_contacts_on']
                                                      for i,name in enumerate(runtime.body_names) if fragment in name],
                                                     device=self.device,dtype=torch.long)
        limits = runtime.dof_pos_limits.to(self.device)
        midpoint = limits.mean(-1)
        span = (limits[:,1]-limits[:,0]) * self.cfg['rewards']['soft_dof_pos_limit']/2
        self.dof_pos_limits = torch.stack((midpoint-span,midpoint+span),-1)
        self._randomize_rigid_properties()
        self._randomize_gravity()
        self._refresh_physics()

    def _uniform(self, shape, interval):
        return torch.rand(shape,device=self.device)*(interval[1]-interval[0])+interval[0]

    def _randomize_rigid_properties(self):
        dr = self.cfg['domain_rand']
        for flag, key, bounds in [('randomize_base_mass','payloads','added_mass_range'),
                                  ('randomize_friction','friction_coeffs','friction_range'),
                                  ('randomize_restitution','restitutions','restitution_range')]:
            if dr[flag]:
                self.buffers[key].copy_(self._uniform(self.buffers[key].shape,dr[bounds]))
        self.runtime.set_material_properties(self.buffers['friction_coeffs'][:,0],self.buffers['restitutions'][:,0])
        self.runtime.set_base_mass_delta(self.buffers['payloads'][:,0],body_name=dr['base_mass_body'])

    def _randomize_dof_properties(self, ids):
        dr = self.cfg['domain_rand']
        for flag, key, bounds in [('randomize_motor_strength','motor_strengths','motor_strength_range'),
                                  ('randomize_motor_offset','motor_offsets','motor_offset_range'),
                                  ('randomize_Kp_factor','Kp_factors','Kp_factor_range'),
                                  ('randomize_Kd_factor','Kd_factors','Kd_factor_range')]:
            if dr[flag]:
                width = 20 if key == 'motor_offsets' else 1
                self.buffers[key][ids] = self._uniform((len(ids),width),dr[bounds])

    def _randomize_gravity(self, restore=False):
        dr = self.cfg['domain_rand']
        offset = self._uniform((3,),dr['gravity_range']) if dr['randomize_gravity'] and not restore else torch.zeros(3,device=self.device)
        self.buffers['gravities'][:] = offset
        gravity = offset + torch.tensor([0.,0.,-9.8],device=self.device)
        self.gravity_vec[:] = gravity / gravity.norm()
        self.runtime.set_gravity(gravity)

    @staticmethod
    def _xyzw(quat):
        return quat[..., [1,2,3,0]]

    def _refresh_physics(self):
        state = self.runtime.state()
        root = torch.cat((state['root_pos'],self._xyzw(state['root_quat']),
                          state['root_lin_vel'],state['root_ang_vel']),-1)
        ee = torch.cat((state['ee_pos'],self._xyzw(state['ee_quat']),
                        state['ee_lin_vel'],state['ee_ang_vel']),-1)
        self.task.update_physics(root_states=root,end_effector_state=ee,
                                 dof_pos=state['joint_pos'],dof_vel=state['joint_vel'],
                                 foot_positions=state['body_pos'][:,self.feet_indices],
                                 foot_velocities=state['body_lin_vel'][:,self.feet_indices],
                                 contact_forces=state['contact_forces'],feet_indices=self.feet_indices,
                                 penalised_contact_indices=self.penalised_contact_indices,
                                 default_dof_pos=self.controller.default_dof_pos,dof_pos_limits=self.dof_pos_limits,
                                 measured_heights=torch.zeros(self.num_envs,1,device=self.device),
                                 gravity_vec=self.gravity_vec,**self.buffers)

    def set_stage(self, stage):
        self.stage = stage
        self.runtime.set_stage(stage)
        self.task.set_stage(stage == 2)

    def set_iteration(self, iteration):
        self.task.set_iteration(iteration, self.stage == 2)

    def get_arm_observations(self):
        return self.observer.arm(vars(self.task))

    def get_dog_observations(self):
        return self.observer.dog(vars(self.task),self.stage == 2)

    def plan(self, guidance):
        self.buffers['plan_actions'].copy_(plan(self.task.commands_dog,guidance,self.config))

    def clear_cached(self, ids):
        self.observer.clear(ids)

    def _reset_indices(self, ids):
        if not len(ids): return {}
        episode = self.task.reset(ids)
        self._randomize_dof_properties(ids)
        q = self.controller.default_dof_pos * self._uniform((len(ids),20),(.5,1.5))
        root_pose = torch.zeros(len(ids),7,device=self.device)
        root_pose[:,:3] = root_pose.new_tensor(self.cfg['init_state']['pos'])
        # Source plane terrain has custom_origins=False: xy randomization here
        # would differ from _reset_root_states despite x/y_init_range existing.
        yaw = self._uniform((len(ids),),(-self.cfg['terrain']['yaw_init_range'],self.cfg['terrain']['yaw_init_range']))
        quat = quat_from_euler_xyz(torch.zeros_like(yaw),torch.zeros_like(yaw),yaw)
        root_pose[:,3:] = quat[:,[3,0,1,2]]
        velocity = self._uniform((len(ids),6),(-.5,.5))
        self.runtime.reset(ids,joint_pos=q,root_pose=root_pose,root_velocity=velocity)
        # Root/dof simulator tensors are explicitly assigned by the original
        # reset; EE/contact and derived local velocities remain pre-reset caches.
        self.task.root_states[ids] = torch.cat((root_pose[:,:3],quat,velocity),-1)
        self.task.dof_pos[ids] = q
        self.task.dof_vel[ids] = 0
        for key in ('last_actions','last_last_actions','last_dof_vel'):
            self.buffers[key][ids] = 0
        return episode

    def reset(self):
        self._reset_indices(torch.arange(self.num_envs,device=self.device))
        # VelocityTrackingEasyEnv.reset runs one zero-action physical step.
        self.step(torch.zeros(self.num_envs,12,device=self.device),torch.zeros(self.num_envs,6,device=self.device))
        self.observer.clear()

    def step(self, dog_action, arm_action):
        if self.stage == 1: arm_action = torch.zeros_like(arm_action)
        actions = torch.cat((dog_action,arm_action),-1)
        self.task.prev_foot_velocities = self.task.foot_velocities.clone()
        for _ in range(self.controller.decimation):
            q,qd = self.runtime.joints()
            output = self.controller.compute(actions,q,qd,kp_factors=self.buffers['Kp_factors'],
                                              kd_factors=self.buffers['Kd_factors'],motor_offsets=self.buffers['motor_offsets'],
                                              motor_strengths=self.buffers['motor_strengths'])
            self.runtime.step_control(output['leg_effort'],output['arm_position'])
        for key, source in [('actions','actions'),('joint_pos_target','joint_pos_target'),('torques','combined')]:
            self.buffers[key].copy_(output[source])
        self._refresh_physics()
        self.task.advance()
        dr = self.cfg['domain_rand']
        rand_interval = math.ceil(dr['rand_interval_s']/self.dt)
        ids = (self.task.episode_length_buf % rand_interval == 0).nonzero().flatten()
        self._randomize_dof_properties(ids)
        gravity_interval = math.ceil(dr['gravity_rand_interval_s']/self.dt)
        gravity_duration = math.ceil(gravity_interval*dr['gravity_impulse_duration'])
        if self.task.common_step_counter % gravity_interval == 0:
            self._randomize_gravity()
        if int(self.task.common_step_counter-gravity_duration) % gravity_interval == 0:
            self._randomize_gravity(restore=True)
        dones,timeouts = self.task.check_termination()
        dog_reward,arm_reward = self.task.compute_reward()
        dog_reward,arm_reward = dog_reward.clone(),arm_reward.clone()
        episode = self._reset_indices(dones.nonzero().flatten())
        self.buffers['last_plan_actions'].copy_(self.buffers['plan_actions'])
        self.buffers['last_last_actions'].copy_(self.buffers['last_actions'])
        self.buffers['last_actions'].copy_(self.buffers['actions'])
        self.buffers['last_last_joint_pos_target'].copy_(self.buffers['last_joint_pos_target'])
        self.buffers['last_joint_pos_target'].copy_(self.buffers['joint_pos_target'])
        self.buffers['last_dof_vel'].copy_(self.task.dof_vel)
        # Fix source reset_idx's stale extras alias: bootstrap only the actual
        # timeout step, not later steps that happen to have no reset.
        infos = {'episode':episode}
        if self.cfg['env']['send_timeouts']: infos['time_outs'] = timeouts
        return dog_reward,arm_reward,dones,infos

    def training_state(self):
        # Include the post-reset cached task observations, not freshly recomputed
        # observations, because that distinction changes the next policy input.
        cached_names = ('root_states','end_effector_state','dof_pos','dof_vel','foot_positions',
                        'foot_velocities','contact_forces','base_lin_vel','base_ang_vel','projected_gravity')
        return dict(runtime=self.runtime.training_state(),task=self.task.state_dict(),
                    buffers={k:v.clone() for k,v in self.buffers.items()},gravity_vec=self.gravity_vec.clone(),
                    cached={k:getattr(self.task,k).clone() for k in cached_names},
                    dog_history=self.observer.dog_obs_history,arm_history=self.observer.arm_obs_history,
                    stage=self.stage,physical_restore='explicit-state-no-PhysX-contact-cache')

    def load_training_state(self, state):
        self.runtime.load_training_state(state['runtime'])
        for key,value in state['buffers'].items(): self.buffers[key].copy_(value.to(self.device))
        self.gravity_vec.copy_(state['gravity_vec'].to(self.device))
        self._refresh_physics()
        self.task.load_state_dict(state['task'])
        for key,value in state['cached'].items(): setattr(self.task,key,value.to(self.device))
        self.task.base_pos = self.task.root_states[:,:3]
        self.task.base_quat = self.task.root_states[:,3:7]
        self.observer.dog_obs_history = state['dog_history'].to(self.device)
        self.observer.arm_obs_history = state['arm_history'].to(self.device)
        self.set_stage(state['stage'])

    def restore_arm_observation_cache(self, cache):
        result = {k:v.to(self.device) for k,v in cache.items()}
        result['obs_history'] = self.observer.arm_obs_history
        return result
