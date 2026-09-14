"""Original UMI task lifecycle on the project's Isaac Lab runtime.

Importing this module does not start Isaac. Only the designated GPU operator
creates the runtime after AppLauncher. Runtime owns physical properties and
contact observations; this adapter owns tasks, controls, rewards and PPO tensors.
"""
import torch
from pawcerto.methods.umi_on_legs.actuation import actuation_mode, AS2_NATIVE_SERVO
from pawcerto.methods.umi_on_legs import PoseSequence, UmiObservation, UmiController
from .semantics import (UmiReward, RewardState, critic_observation, termination,
                        emd_force_z, runtime_contract, require_resume_contract)


class UmiIsaacTrainingEnv:
    def __init__(self, runtime, config, trajectory_path, seed=0):
        self.runtime, self.config = runtime, config
        self.device = runtime.device
        self.num_envs = runtime.num_envs
        self.runtime_metadata = dict(config['pawcerto_runtime'])
        self.force_signal = self.runtime_metadata['force_signal']
        self.actuation_mode = actuation_mode(config)
        self.native_servo = self.actuation_mode == AS2_NATIVE_SERVO
        velocity_override = config.get('joint_velocity_limit_override_rad_s')
        if self.runtime_metadata != runtime_contract(self.force_signal, velocity_override, self.actuation_mode):
            raise ValueError('The adapter requires the current explicit reward/physics runtime contract')
        if getattr(runtime, 'joint_velocity_limit_override_rad_s', None) != velocity_override:
            raise ValueError('Runtime and adapter joint velocity overrides must agree')
        if getattr(runtime, 'force_signal', None) != self.force_signal:
            raise ValueError('Runtime and adapter EMD force_signal must agree')
        if getattr(runtime, 'actuation_mode', 'external-pd') != self.actuation_mode:
            raise ValueError('Runtime and adapter actuation modes must agree')
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.seed, self.reset_count, self.global_step = seed, 0, 0
        self.sampler = PoseSequence(trajectory_path, config['env']['tasks']['reaching']['sequence_sampler'], self.device)
        positions, rotations = self.sampler.sample(self.num_envs, seed)
        self.observer = UmiObservation(config, positions, rotations)
        self.controller = UmiController(config, self.num_envs, self.device)
        self.initial_kp = self.controller.kp.clone()
        self.initial_kd = self.controller.kd.clone()
        self.controller.kp = self.initial_kp.repeat(self.num_envs,1)
        self.controller.kd = self.initial_kd.repeat(self.num_envs,1)
        if self.native_servo:
            self.randomized_kd = self.controller.kd.clone()
            self._sync_servo_gains(torch.arange(self.num_envs, device=self.device))
        self.reward = UmiReward(config, runtime.body_names, runtime.joint_names, self.num_envs, self.device)
        self.max_episode_length_s = config['env']['cfg']['env']['episode_length_s']

    def _reset_indices(self, ids):
        if len(ids) == 0:
            return
        self.runtime.reset(ids)
        self.reset_count += 1
        pos, rot = self.sampler.sample(len(ids), self.seed+self.reset_count)
        self.observer.positions[ids] = pos
        self.observer.rotations[ids] = rot
        self.observer.reset(ids)
        self.controller.reset(ids)
        self.reward.curriculum.on_reset()
        rand = self.config['env']['cfg']['domain_rand']
        if rand['randomize_pd_params']:
            for name in ('kp', 'kd'):
                low, high = rand[f'{name}_ratio_range']
                ratios = torch.rand((len(ids),18), device=self.device, generator=self.generator)*(high-low)+low
                getattr(self.controller, name)[ids] = ratios*getattr(self, f'initial_{name}')
                if self.native_servo and name == 'kd':
                    self.randomized_kd[ids] = self.controller.kd[ids]
        if self.native_servo:
            self._sync_servo_gains(ids)

    def _sync_servo_gains(self, ids):
        # Each source is retained separately; do not add passive damping twice on reset.
        damping = self.runtime.training_setup()['dof_damping'][ids]
        self.controller.kd[ids] = self.randomized_kd[ids] + damping
        self.runtime.set_servo_gains(self.controller.kp[ids], self.controller.kd[ids], ids)

    def _observations(self):
        state = self.runtime.state()
        raw = self.runtime.training_state()
        setup = self.runtime.training_setup()
        setup = dict(setup, kp=self.controller.kp, kd=self.controller.kd)
        actor = self.observer.observe(state, self.controller.last_action,
                                      self.generator if self.config.get('use_noise',True) else None)
        privileged_state = dict(local_root_lin_vel=raw['local_root_lin_vel'],
                                local_root_ang_vel=state.local_root_ang_vel,
                                local_root_gravity=state.local_root_gravity,
                                dof_pos=state.dof_pos, dof_vel=state.dof_vel)
        # Released checkpoints both have one history frame; task block follows 42 state values.
        task_obs = actor[:,42:-18]
        critic = critic_observation(self.config, privileged_state, setup, task_obs, self.controller.last_action)
        return actor, critic

    def reset(self):
        self._reset_indices(torch.arange(self.num_envs,device=self.device))
        # Official Env.reset executes a complete zero-action policy step.
        # This advances pose history, curriculum EMA and the global perturbation clock.
        actor, critic, _, _, _ = self.step(torch.zeros((self.num_envs,18),device=self.device))
        return actor, critic

    def randomize_episode_progress(self):
        dt = self.config['env']['cfg']['sim']['dt']
        self.runtime.time[:] = torch.randint(0,int(self.max_episode_length_s/dt),
                                            (self.num_envs,),device=self.device,generator=self.generator)*dt

    def step(self, action):
        if self.config['env']['dense_rewards']:
            return self._step_dense(action)
        return self._step_sparse(action)

    def _step_sparse(self, action):
        previous_action = self.controller.last_action.clone()
        self.controller.push(action)
        for substep in range(self.controller.decimation):
            q, qd = self.runtime.joints()
            previous_qd = qd.clone()
            torque = self.controller.step(self.runtime,q,qd,substep)
        state = self.runtime.state()
        raw = self.runtime.training_state()
        self.observer.advance_pose(state.ee_pose)
        target_pos,target_rot = self.observer.targets(state.episode_time)
        position_error = (target_pos-state.ee_pose[:,:3,3]).norm(dim=-1)
        relative = target_rot @ state.ee_pose[:,:3,:3].transpose(-1,-2)
        trace = relative.diagonal(dim1=-2,dim2=-1).sum(-1).clamp(-1+1e-8,3-1e-8)
        orientation_error = torch.acos((trace-1)/2)
        reward_state = RewardState(state.dof_pos,state.dof_vel,previous_qd,torque,
                                   self.controller.last_action,previous_action,state.local_root_gravity,
                                   raw['root_pos'],raw['contact_forces'],
                                   emd_force_z(raw,self.force_signal,self.num_envs),raw['body_pos'],
                                   position_error,orientation_error)
        reward, terms = self.reward(reward_state)
        self.global_step += 1
        self.runtime.apply_domain_randomization(self.global_step)
        raw = self.runtime.training_state()
        policy_dt = self.controller.dt*self.controller.decimation
        # The source computes floor(episode_time / gym_dt), including randomized episode clock.
        episode_step = (self.runtime.time/policy_dt).long()
        done,timeouts = termination(self.config,raw['contact_forces'],self.runtime.body_names,
                                    episode_step,raw['root_pos'],getattr(self.runtime,'safe_bounds',None))
        info = dict(terms, time_outs=timeouts.clone(), position_error=position_error,
                    orientation_error=orientation_error,
                    root_height=raw['root_pos'][:,2].clone(),
                    supported_feet=(raw['feet_force_z']>1.).float().sum(-1),
                    pos_sigma=torch.full_like(reward,self.reward.curriculum.pos_sigma),
                    orn_sigma=torch.full_like(reward,self.reward.curriculum.orn_sigma))
        self._reset_indices(done.nonzero().flatten())
        actor,critic = self._observations()
        return actor,critic,reward,done,info

    def _step_dense(self, action):
        """Author evaluation cadence: task/history/reward at every physics step."""
        previous_action = self.controller.last_action.clone()
        self.controller.push(action)
        reward = torch.zeros(self.num_envs,device=self.device)
        reward_terms, stats = {}, {}
        min_up_dot = torch.full_like(reward,float('inf'))
        for substep in range(self.controller.decimation):
            q, qd = self.runtime.joints()
            previous_qd = qd.clone()
            torque = self.controller.step(self.runtime,q,qd,substep)
            state = self.runtime.state()
            raw = self.runtime.training_state()
            self.observer.advance_pose(state.ee_pose)
            target_pos,target_rot = self.observer.targets(state.episode_time)
            position_error = (target_pos-state.ee_pose[:,:3,3]).norm(dim=-1)
            relative = target_rot @ state.ee_pose[:,:3,:3].transpose(-1,-2)
            trace = relative.diagonal(dim1=-2,dim2=-1).sum(-1).clamp(-1+1e-8,3-1e-8)
            orientation_error = torch.acos((trace-1)/2)
            reward_state = RewardState(state.dof_pos,state.dof_vel,previous_qd,torque,
                                       self.controller.last_action,previous_action,state.local_root_gravity,
                                       raw['root_pos'],raw['contact_forces'],
                                       emd_force_z(raw,self.force_signal,self.num_envs),raw['body_pos'],
                                       position_error,orientation_error)
            sub_reward, terms = self.reward(reward_state)
            reward += sub_reward
            for key,value in terms.items():
                reward_terms[key] = reward_terms.get(key,0.) + value*self.reward.reward_dt
            support = (raw['feet_force_z']>1.).float().sum(-1)
            sub_stats = dict(position_error=position_error,orientation_error=orientation_error,
                             root_height=raw['root_pos'][:,2],supported_feet=support,
                             zero_supported_fraction=(support==0).float(),
                             pos_sigma=torch.full_like(reward,self.reward.curriculum.pos_sigma),
                             orn_sigma=torch.full_like(reward,self.reward.curriculum.orn_sigma))
            if getattr(self.runtime,'ground_contact_diagnostics',False):
                ground_force_z = self.runtime.feet_ground_force_z()
                ground_support = (ground_force_z>1.).float().sum(-1)
                sub_stats.update(ground_supported_feet=ground_support,
                                 zero_ground_supported_fraction=(ground_support==0).float(),
                                 ground_foot_force_z_total=ground_force_z.sum(-1))
                for index,leg in enumerate(('FR','FL','RR','RL')):
                    sub_stats[f'ground_foot_force_z_{leg}'] = ground_force_z[:,index]
            energy = self.config['env']['constraints'].get('energy')
            if energy is not None:
                mechanical_key = 'servo_mechanical_power_estimate' if self.native_servo else 'mechanical_power'
                electrical_key = 'servo_electrical_power_estimate' if self.native_servo else 'electrical_power'
                sub_stats[mechanical_key] = (torque*state.dof_vel).sum(-1)
                constants = torque.new_tensor(energy['torque_constant']['data'])
                voltage = torque.new_tensor(energy['voltage']['data'])
                # Preserve the author's reported electrical-power estimator.
                sub_stats[electrical_key] = (torque.abs()*constants*voltage).sum(-1)
            for key,value in sub_stats.items():
                stats[key] = (stats.get(key,0.)*substep+value)/(substep+1)
            min_up_dot = torch.minimum(min_up_dot,-state.local_root_gravity[:,2])
        self.global_step += 1
        self.runtime.apply_domain_randomization(self.global_step)
        raw = self.runtime.training_state()
        policy_dt = self.controller.dt*self.controller.decimation
        episode_step = (self.runtime.time/policy_dt).long()
        done,timeouts = termination(self.config,raw['contact_forces'],self.runtime.body_names,
                                    episode_step,raw['root_pos'],getattr(self.runtime,'safe_bounds',None))
        info = dict(stats,time_outs=timeouts.clone(),root_up_dot_min=min_up_dot)
        info.update({f'reward/{key}':value for key,value in reward_terms.items()})
        self._reset_indices(done.nonzero().flatten())
        actor,critic = self._observations()
        return actor,critic,reward,done,info

    def training_state_dict(self):
        state = dict(curriculum=self.reward.curriculum.state_dict(), global_step=self.global_step,
                    reset_count=self.reset_count, generator_state=self.generator.get_state(),
                    pawcerto_runtime=dict(self.runtime_metadata))
        if self.native_servo:
            state['native_servo_gain_samples'] = dict(kp=self.controller.kp.clone(),
                randomized_kd=self.randomized_kd.clone(),
                sampled_dof_damping=self.runtime.training_setup()['dof_damping'].clone(),
                effective_kd=self.controller.kd.clone())
        return state

    def load_training_state_dict(self,state):
        require_resume_contract(state.get('pawcerto_runtime'),self.runtime_metadata)
        self.reward.curriculum.load_state_dict(state['curriculum'])
        self.global_step,self.reset_count=state['global_step'],state['reset_count']
        self.generator.set_state(state['generator_state'].cpu())
