"""DeepWBC six-return training environment over the real Lab runtime.

The runtime owns physics, PD and delay. Task/observer own tensor semantics.
No simulator is imported here; a CPU interface double tests ordering only.
"""
from copy import deepcopy
import math
import torch
from ..task import DeepWBCTask, euler_from_quat
from ..observer import DeepWBCObserver


class DeepWBCIsaacTrainingEnv:
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, deepcopy(config)
        self.device = torch.device(runtime.device)
        self.num_envs = runtime.num_envs
        self.num_obs = config['env']['num_observations']
        self.num_privileged_obs = config['env']['num_privileged_obs']
        self.num_actions = config['env']['num_actions']
        if not math.isclose(runtime.dt, config['sim']['dt']):
            raise ValueError('Runtime physical timestep differs from DeepWBC configuration')
        if config['terrain']['curriculum'] or config['terrain']['measure_heights']:
            raise ValueError('Terrain curriculum/height sampling require an implemented backend task path')
        self.task = DeepWBCTask(self.num_envs, config, self.device)
        self.observer = DeepWBCObserver(self.num_envs, runtime.joint_names, config, self.device)
        self.max_episode_length = self.task.max_episode_length
        self.dt = self.task.dt
        self.termination_contact_indices = [i for fragment in config['asset']['terminate_after_contacts_on']
                                            for i,name in enumerate(runtime.body_names) if fragment in name]
        self.randomization = self.task.sample_creation_randomization()
        self.runtime.apply_creation_randomization(**{key: self.randomization[key] for key in
            ('mass_params','friction','motor_strength','box_added_mass')})
        self.origins = torch.zeros(self.num_envs,3,device=self.device)
        # Original single shared terrain placement, independent from clone grid.
        terrain = config['terrain']
        half_col = terrain['tot_cols']*terrain['horizontal_scale']/2
        half_row = terrain['tot_rows']*terrain['horizontal_scale']/2
        for axis, (low,high) in enumerate(((-2.5*half_col/5,-2*half_col/5),(-half_row+10,half_row-10))):
            self.origins[:,axis] = torch.rand(self.num_envs,device=self.device)*(high-low)+low
        self.obs_buf = None
        self._bind_state()

    @property
    def episode_length_buf(self):
        return self.task.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.task.episode_length_buf = value

    def _bind_state(self):
        state = self.runtime.state()
        self.state = state
        # Clone action and torque buffers: runtime reset clears these in place,
        # but source reward transition and last_actions retain applied action.
        self.task.bind_state(**{key: state[key].clone() for key in (
            'root_states','dof_pos','dof_vel','torques','actions','ee_pos','ee_orn',
            'contact_forces','foot_wrenches')},
            termination_contact_indices=self.termination_contact_indices)

    def _observe(self):
        t = self.task
        roll,pitch,_ = euler_from_quat(t.base_quat)
        latest = self.runtime.controller.history[:,-1]
        self.obs_buf = self.observer.observe(roll_pitch=torch.stack((roll,pitch),-1),
            base_ang_vel=t.base_ang_vel,dof_pos=t.dof_pos,dof_vel=t.dof_vel,
            latest_sim_actions=latest,foot_wrenches=t.force_sensor_tensor,
            commands=t.commands,ee_goal_lpy=t.curr_ee_goal_sphere,
            ee_delta_orn=t.ee_goal_delta_orn_euler,
            mass_params=self.randomization['mass_params'],friction=self.randomization['friction'],
            motor_strength=self.randomization['motor_strength'],episode_length=t.episode_length_buf)
        return self.obs_buf

    def _reset_indices(self, ids, start=False):
        if not len(ids):
            return {}
        writes = self.task.reset(ids, default_dof_pos=self.runtime.controller.default,
                                  env_origins=self.origins, start=start)
        self.runtime.reset(ids,root_states=writes['root_states'],joint_pos=writes['dof_pos'],joint_vel=writes['dof_vel'])
        box_pos = self.task.reset_box_position(writes['root_states'], self.randomization['box_delta_y'][ids])
        self.runtime.reset_box(ids,box_pos)
        self.observer.reset(ids)
        # Original reset updates root/q tensors, but keeps derived local velocity,
        # EE and contact caches until next post-physics refresh.
        self.task.root_states[ids] = writes['root_states']
        self.task.dof_pos[ids] = writes['dof_pos']
        self.task.dof_vel[ids] = writes['dof_vel']
        return writes['episode']

    def reset(self):
        self._reset_indices(torch.arange(self.num_envs,device=self.device),start=True)
        # Original BaseTask.reset performs one complete zero-action policy step.
        obs, priv, *_ = self.step(torch.zeros(self.num_envs,self.num_actions,device=self.device))
        return obs, priv

    def get_observations(self):
        if self.obs_buf is None:
            raise RuntimeError('Call reset before requesting observations')
        return self.obs_buf

    def get_privileged_observations(self):
        return None # 24 privileged slots are embedded in the 860-column input

    def update_command_curriculum(self):
        self.task.update_command_curriculum()

    def step(self, actions, *, transition_callback=None):
        self.runtime.push_actions(actions)
        for _ in range(self.config['control']['decimation']):
            self.runtime.step_control()
        self._bind_state()
        leg, arm, done, extras = self.task.after_physics()
        # Evaluation reads the actual transition before pushes/reset mutate it.
        # No extra observation/history update or reward evaluation is performed.
        if transition_callback is not None:
            transition_callback(self, leg, arm, done, extras)
        if 'push_velocity_xy' in extras:
            self.runtime.push_base_velocity_xy(extras.pop('push_velocity_xy'))
        extras['episode'] = self._reset_indices(done.nonzero().flatten())
        obs = self._observe()
        self.task.finish_step()
        return obs, None, leg, arm, done, extras
