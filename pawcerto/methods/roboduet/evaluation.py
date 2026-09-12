"""Fixed-command physical evaluation; all physics remains in the training runtime.

This is an engineering protocol, not an upstream quantitative benchmark.
"""
from copy import deepcopy
import json
from pathlib import Path
import types

import torch

from .observations import quat_apply, quat_conjugate, quat_mul, quat_from_euler_xyz


def nominal_config(config, num_envs):
    config = deepcopy(config)
    cfg = config['Cfg']
    cfg['env']['num_envs'] = num_envs
    cfg['terrain']['mesh_type'] = 'plane'
    cfg['noise']['add_noise'] = False
    for key in cfg['domain_rand']:
        if key.startswith('randomize_') or key == 'push_robots':
            cfg['domain_rand'][key] = False
    return config


def validate_protocol(protocol, config):
    if protocol['format'] != 'pawcerto.roboduet.fixed_protocol.v1':
        raise ValueError('Unknown evaluation protocol')
    cfg = config['Cfg']
    if cfg['use_rot6d'] or cfg['hybrid']['use_vision'] or cfg['hybrid']['plan_vel']:
        raise ValueError('Fixed evaluation supports the original no-vision LPY/ABG path')
    dt = cfg['sim']['dt'] * cfg['control']['decimation']
    for key in ('duration_s', 'trajectory_s'):
        if protocol[key] <= 0 or abs(round(protocol[key] / dt)*dt - protocol[key]) > 1e-7:
            raise ValueError(f'{key} must be positive and an integer control period')
    low, high = cfg['arm']['commands']['T_traj']
    if not low <= protocol['trajectory_s'] <= high:
        raise ValueError('T_traj outside supported range')
    for case in protocol['cases']:
        if not case['dog'] or len(case['dog']) != len(case['arm_lpy_rpy']):
            raise ValueError('Dog and arm phase counts must match')
        for values in case['arm_lpy_rpy']:
            if len(values) != 6:
                raise ValueError('Expected LPY and roll/pitch/yaw')
            for value, key in zip(values, ('l','p','y','roll_ee','pitch_ee','yaw_ee')):
                low, high = cfg['arm']['commands'][key]
                if not low <= value <= high:
                    raise ValueError(f'Arm {key} outside supported range')
        for values in case['dog']:
            if len(values) != 3:
                raise ValueError('Expected vx, vy, yaw rate')
            for value, key in zip(values, ('vel_x','vel_y','vel_yaw')):
                low, high = cfg['commands']['limit_'+key]
                if not low <= value <= high:
                    raise ValueError(f'Dog {key} outside supported range')
    return round(protocol['duration_s'] / dt)


class FixedCommands:
    """Replace only random command sampling, retaining the task's resample clock.

    On each native arm resample event write the next fixed LPY/RPY target and
    matching velocity. Orientation construction is Rz(yaw) Ry(pitch) Rx(roll),
    then source projected-axis ABG for the policy, exactly as task sampling.
    """
    def __init__(self, task, protocol):
        self.task, self.protocol = task, protocol
        self.phase = [-1] * len(protocol['cases'])
        task._resample_arm_commands = types.MethodType(lambda task, ids: self.arm(ids), task)
        task._resample_commands = types.MethodType(lambda task, ids: self.dog(ids), task)

    def dog(self, ids):
        for index in ids.tolist():
            case = self.protocol['cases'][index]
            phase = max(self.phase[index], 0) % len(case['dog'])
            self.task.commands_dog[index, :3] = self.task.commands_dog.new_tensor(case['dog'][phase])
        for value in self.task.command_sums.values():
            value[ids] = 0

    def arm(self, ids):
        task = self.task
        for index in ids.tolist():
            self.phase[index] += 1
            case = self.protocol['cases'][index]
            phase = self.phase[index] % len(case['arm_lpy_rpy'])
            value = task.commands_arm.new_tensor(case['arm_lpy_rpy'][phase])
            task.commands_arm[index, :3] = value[:3]
            task.commands_arm_obs[index, :3] = value[:3]
            quat = quat_from_euler_xyz(value[3:4], value[4:5], value[5:6])
            task.obj_quats[index] = quat[0]
            task.visual_rpy[index] = value[3:]
            task.target_abg[index] = task.quat_to_angle(quat)[0]
            task.commands_arm_obs[index, 3:6] = task.target_abg[index]
            task.T_trajs[index] = self.protocol['trajectory_s']
            task.arm_time_buf[index] = 0
        self.dog(ids)


def issued_targets(task):
    return {name: getattr(task, name).clone() for name in
            ('commands_dog','commands_arm','obj_quats','target_abg','T_trajs','arm_time_buf')}


def physical_sample(env, targets, dones, timeouts):
    """Read and clone terminal state before automatic reset; no reward proxies."""
    task = env.task
    ids = torch.arange(env.num_envs, device=env.device)
    measured_lpy = task.get_lpy_in_base_coord(ids)
    def cartesian(lpy):
        length, pitch, yaw = lpy.unbind(-1)
        return torch.stack((length*pitch.cos()*yaw.cos(), length*pitch.cos()*yaw.sin(), length*pitch.sin()), -1)
    position_error = (cartesian(measured_lpy)-cartesian(targets['commands_arm'][:, :3])).norm(dim=-1)
    forward = quat_apply(task.base_quat, task.forward_vec)
    yaw = torch.atan2(forward[:, 1], forward[:, 0])
    yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    target_world = quat_mul(yaw_quat, targets['obj_quats'])
    delta = quat_mul(quat_conjugate(target_world), task.end_effector_state[:,3:7])
    orientation_error = 2*torch.atan2(delta[:,:3].norm(dim=-1), delta[:,3].abs())
    up = task.base_quat.new_tensor([0.,0.,1.]).expand(env.num_envs,-1)
    measured_abg = task.get_alpha_beta_gamma_in_base_coord(ids)
    metrics = {
        'leg_linear_error_m_s': (task.base_lin_vel[:,:2]-targets['commands_dog'][:,:2]).norm(dim=-1),
        'leg_yaw_error_rad_s': (task.base_ang_vel[:,2]-targets['commands_dog'][:,2]).abs(),
        'ee_position_error_m': position_error,
        'ee_orientation_error_rad': orientation_error,
        'up_dot': (quat_apply(task.base_quat,up)*up).sum(dim=-1),
        'leg_effort_limit_fraction': (env.buffers['torques'][:,:12].abs() >= env.controller.torque_limits[:12]-1e-6).float().mean(-1),
        'action_clip_fraction': (env.buffers['actions'].abs() >= env.cfg['normalization']['clip_actions']).float().mean(-1),
    }
    # Sensor values are available at the final physics substep only. They do
    # not prove peak impact during the four-substep control interval.
    head_ids = [i for i, name in enumerate(env.runtime.body_names)
                if 'head' in name.lower() and name in env.runtime.contact_names]
    if head_ids:
        metrics['head_contact_force_N'] = task.contact_forces[:,head_ids].norm(dim=-1).amax(-1)
    tensors = dict(metrics, done=dones, timeout=timeouts,
        body_height_termination=task.body_height_buf, reverse_termination=task.reverse_buf,
        root_states=task.root_states, joint_pos=task.dof_pos, joint_vel=task.dof_vel,
        ee_state=task.end_effector_state, contact_forces=task.contact_forces,
        measured_lpy=measured_lpy, measured_abg=measured_abg,
        base_lin_vel=task.base_lin_vel, base_ang_vel=task.base_ang_vel, **targets)
    return {key:value.detach().cpu().clone() for key,value in tensors.items()}


class EpisodeRecorder:
    """Keep each case's first prefix including its failure, never post-reset data."""
    def __init__(self, names, dt, steps):
        self.names, self.dt, self.steps = names, dt, steps
        self.records = [[] for _ in names]
        self.active = [True] * len(names)
        self.initialization = [None] * len(names)
        self.policy_samples = []
        self.substeps = [[] for _ in names]

    def append(self, sample, *, initialization=False):
        for i in range(len(self.names)):
            if not self.active[i]:
                continue
            record = {key:value[i].clone() for key,value in sample.items()}
            finite = all(bool(torch.isfinite(value).all()) for value in record.values())
            record['finite'] = torch.tensor(finite)
            if initialization:
                self.initialization[i] = record
            else:
                self.records[i].append(record)
            if bool(record['done']) or not finite:
                self.active[i] = False

    def append_substep(self, env, substep, before, output, state, targets, *, initialization=False):
        # Runtime state() is the actual post-step source even for runtimes whose
        # step_control does not return it. The CPU fixture uses prescribed state.
        if state is None:
            state = env.runtime.state()
        quat = env._xyzw(state['root_quat'])
        up = quat.new_tensor([0.,0.,1.]).expand(env.num_envs,-1)
        tensors = dict(joint_pos_before=before[0], joint_vel_before=before[1],
            joint_pos=state['joint_pos'], joint_vel=state['joint_vel'],
            joint_pos_target=output['joint_pos_target'], clipped_actions=output['actions'],
            sent_effort_command=output['combined'], root_pos=state['root_pos'],
            root_quat_xyzw=quat, root_lin_vel=state['root_lin_vel'], root_ang_vel=state['root_ang_vel'],
            body_pos=state['body_pos'], contact_forces=state['contact_forces'],
            root_height=state['root_pos'][:,2], up_dot=quat_apply(quat,up)[:,2], **targets)
        copied = {key:value.detach().cpu().clone() for key,value in tensors.items()}
        for i, active in enumerate(self.active):
            if active:
                row = {key:value[i].clone() for key,value in copied.items()}
                row.update(initialization=initialization, substep=substep,
                           time_s=(len(self.substeps[i])+1)*env.runtime.dt)
                self.substeps[i].append(row)

    def summary(self):
        rows = []
        for i, name in enumerate(self.names):
            records = self.records[i]
            initial = self.initialization[i]
            last = records[-1] if records else initial
            if last is None:
                raise ValueError('No physical observations recorded')
            finite = bool(last['finite'])
            done, timeout = bool(last['done']), bool(last['timeout'])
            physical_failure = bool(last.get('body_height_termination',False)) or bool(last.get('reverse_termination',False)) or (done and not timeout)
            complete = len(records) == self.steps and finite and not physical_failure
            row = dict(case=name, samples=len(records), survival_s=len(records)*self.dt,
                full_duration=complete, non_timeout_termination=physical_failure,
                timeout=timeout, invalid_prefix=not records, finite=finite,
                body_height_termination=bool(last.get('body_height_termination',False)),
                reverse_termination=bool(last.get('reverse_termination',False)))
            for key in ('leg_linear_error_m_s','leg_yaw_error_rad_s','ee_position_error_m',
                        'ee_orientation_error_rad','leg_effort_limit_fraction','action_clip_fraction','head_contact_force_N'):
                values = [float(record[key]) for record in records if key in record and torch.isfinite(record[key])]
                row[key+'_mean'] = sum(values)/len(values) if values else None
                row[key+'_max'] = max(values) if values else None
            up = [float(record['up_dot']) for record in records if torch.isfinite(record['up_dot'])]
            row['up_dot_min'] = min(up) if up else None
            rows.append(row)
        return rows

    def save(self, output, metadata):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        torch.save(dict(metadata=metadata, initialization=self.initialization,
                        cases={name:records for name,records in zip(self.names,self.records)}), output/'states.pt')
        torch.save(self.policy_samples, output/'policy_inputs_outputs.pt')
        torch.save(dict(metadata=metadata, cases=dict(zip(self.names,self.substeps))), output/'substeps.pt')
        result = dict(metadata=metadata, cases=self.summary())
        (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result


@torch.inference_mode()
def evaluate_seed(env, policy, protocol):
    steps = validate_protocol(protocol, env.config)
    env.set_stage(2)
    env.set_iteration(policy.info['next_iteration'])
    FixedCommands(env.task, protocol)
    recorder = EpisodeRecorder([case['name'] for case in protocol['cases']], env.dt, steps)
    env._reset_indices(torch.arange(env.num_envs, device=env.device))
    target = issued_targets(env.task)
    env.step(torch.zeros(env.num_envs,12,device=env.device), torch.zeros(env.num_envs,6,device=env.device),
             pre_reset_callback=lambda current,dones,timeouts: recorder.append(
                 physical_sample(current,target,dones,timeouts), initialization=True),
             substep_callback=lambda current,index,before,output,state: recorder.append_substep(
                 current,index,before,output,state,target,initialization=True))
    env.observer.clear()
    # Match original arm observation -> act -> plan -> dog observation -> act -> step.
    env.get_arm_observations()  # Upstream play performs this getter before the loop.
    for step in range(steps):
        if not any(recorder.active):
            break
        arm_obs = env.get_arm_observations()
        arm_input = {k:v.detach().cpu().clone() for k,v in arm_obs.items()}
        arm = policy.arm_action(arm_input).to(env.device)
        env.plan(arm[:,-2:])
        dog_obs = env.get_dog_observations()
        dog_input = {k:v.detach().cpu().clone() for k,v in dog_obs.items()}
        dog = policy.dog_action(dog_input).to(env.device)
        graph_sample = dict(step=step, arm_input=arm_input, dog_input=dog_input,
                            arm_output=arm.cpu().clone(), dog_output=dog.cpu().clone())
        if len(recorder.policy_samples) < 2:
            recorder.policy_samples.append(graph_sample)
        else:
            recorder.policy_samples[-1] = graph_sample
        target = issued_targets(env.task)
        _,_,dones,_ = env.step(dog,arm[:,:6],pre_reset_callback=lambda current,done,timeout:
                              recorder.append(physical_sample(current,target,done,timeout)),
                              substep_callback=lambda current,index,before,output,state: recorder.append_substep(
                                  current,index,before,output,state,target))
        env.clear_cached(dones.nonzero().flatten())
    return recorder
