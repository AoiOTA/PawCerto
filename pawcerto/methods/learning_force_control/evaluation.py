"""Fixed-sampler LFC evaluation with original hybrid modes and pre-reset states."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from .observations import force_in_base_yaw


REASONS = ('contact', 'timeout', 'body_height', 'leg_torque', 'arm_torque',
           'ee_distance', 'body_orientation')


def fixed_task_config(config, task, *, num_envs=16):
    if set(task) != {'commands', 'ee_goal_lpy', 'ee_timing_s', 'hybrid_mode'}:
        raise ValueError('Task requires commands, ee_goal_lpy, ee_timing_s and hybrid_mode')
    commands = np.asarray(task['commands'], dtype=float)
    goal = np.asarray(task['ee_goal_lpy'], dtype=float)
    timing = float(task['ee_timing_s'])
    if (commands.shape != (3,) or goal.shape != (3,)
            or not np.isfinite(np.r_[commands, goal, timing]).all()
            or goal[0] <= 0 or timing <= 0 or num_envs < 1):
        raise ValueError('Expected finite three-vectors, positive EE radius/time and num_envs')
    if task['hybrid_mode'] not in ('binary', 'position', 'force'):
        raise ValueError('Use an original discrete hybrid-mode configuration')
    if -goal[0] * np.sin(goal[1]) + .1585 + .6 < .05:
        raise ValueError('Fixed EE goal violates source ground clearance; resampling fixed ranges cannot recover')
    result = deepcopy(config)
    env = result['env']
    env['env']['num_envs'] = num_envs
    for key, limit, value in zip(('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw'),
                                 ('limit_vel_x', 'limit_vel_y', 'limit_vel_yaw'), commands):
        env['commands'][key] = [float(value)] * 2
        env['commands'][limit] = [float(value)] * 2
    for key, value in zip(('ee_sphe_radius', 'ee_sphe_pitch', 'ee_sphe_yaw'), goal):
        env['commands'][key] = [float(value)] * 2
        env['commands']['limit_' + key] = [float(value)] * 2
    env['commands']['ee_timing'] = [timing] * 2
    env['commands']['limit_ee_timing'] = [timing] * 2
    env['commands']['hybrid_mode'] = task['hybrid_mode']
    return result


def position_target_world(commands, base_pos, base_quat):
    """Original position-reward target: negative pitch, yaw-only, fixed base z."""
    qx, qy, qz, qw = base_quat.unbind(-1)
    yaw = torch.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy.square() + qz.square()))
    radius, pitch, azimuth = commands[:, 15:18].unbind(-1)
    x = radius * torch.cos(pitch) * torch.cos(azimuth) + .2
    y = radius * torch.cos(pitch) * torch.sin(azimuth)
    z = -radius * torch.sin(pitch) + .1585 + .6
    return torch.stack((torch.cos(yaw)*x - torch.sin(yaw)*y + base_pos[:, 0],
                        torch.sin(yaw)*x + torch.cos(yaw)*y + base_pos[:, 1], z), -1)


def transition_snapshot(env):
    zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    masks = [env.contact_buf.bool(), env.time_out_buf.bool()]
    for enabled, field in (
        ('use_terminal_body_height', 'body_height_buf'),
        ('use_terminal_torque_legs_limits', 'legs_torque_lim_buff'),
        ('use_terminal_torque_arm_limits', 'arm_torque_lim_buff'),
        ('use_terminal_ee_position', 'ee_position_lim_buff'),
        ('use_terminal_roll_pitch', 'body_ori_buf'),
    ):
        masks.append(getattr(env, field).bool() if getattr(env.cfg.rewards, enabled) else zero)
    reasons = torch.stack(masks, -1)
    if not torch.equal(reasons.any(-1), env.reset_buf.bool()):
        raise RuntimeError('LFC termination reasons disagree with source reset decision')
    qx, qy, qz, qw = env.base_quat.unbind(-1)
    yaw = torch.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy.square() + qz.square()))
    target = position_target_world(env.commands, env.base_pos, env.base_quat)
    # Original Gym lookup gripperStator=-1 reads its final body gripperMover.
    # Force injection and the similarly named gripper_position use link06.
    measured = env.rigid_body_state[:, env.ee_measurement_index, :3]
    force_local = force_in_base_yaw(env.forces[:, env.gripper_stator_index, :3], yaw)
    values = {key: getattr(env, key) for key in (
        'root_states', 'dof_pos', 'dof_vel', 'joint_pos_target', 'commands', 'actions',
        'rigid_body_state', 'contact_forces', 'forces', 'projected_gravity',
        'base_lin_vel', 'base_ang_vel', 'force_or_position_control', 'freed_envs',
        'gripper_force_kps', 'gripper_force_kds', 'episode_length_buf', 'rew_buf')}
    values.update(source_pd_torques=env.torques, reason_mask=reasons, done=env.reset_buf.bool(),
        measured_ee_position=measured, ee_position_target_world=target,
        ee_position_error_m=(target-measured).norm(dim=-1),
        applied_ee_force_in_base_yaw=force_local,
        force_error_xyz_N=force_local-env.commands[:, 12:15],
        velocity_error_xy=env.base_lin_vel[:, :2]-env.commands[:, :2],
        yaw_rate_error=env.base_ang_vel[:, 2]-env.commands[:, 2],
        base_up_dot=1-2*(qx.square()+qy.square()),
        common_step_counter=torch.full((env.num_envs,), env.common_step_counter,
                                       dtype=torch.int64, device=env.device))
    return {key: value.detach().cpu().numpy().copy() for key, value in values.items()}


def evaluate_fixed_policy(env, model, *, steps, output, metadata, exported_policy):
    """Actually drive physics with the JIT pair; retain automatic source resets."""
    if steps < 1:
        raise ValueError('steps must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    previous_observer = env.transition_observer
    rows, pending = [], None
    compared, observed = 0, 0
    action_error, latent_error = 0., 0.
    status, error = 'failed', None

    def observe(current):
        nonlocal pending, observed
        if pending is not None:
            raise RuntimeError('Multiple LFC snapshots for one policy step')
        pending = transition_snapshot(current)
        observed += 1

    try:
        observations = env.reset()
        env.transition_observer = observe
        with torch.inference_mode():
            for step in range(steps):
                history = observations['obs_history']
                reference = model.act_student(history, policy_info={})
                expected_latent = model.get_student_latent(history)
                action = exported_policy(history)
                latent = exported_policy.adaptation(history)
                inputs = {'observations': history.detach().cpu().numpy().copy(),
                    'privileged_obs': observations['privileged_obs'].detach().cpu().numpy().copy(),
                    'raw_actions': action.detach().cpu().numpy().copy(),
                    'predicted_latent': latent.detach().cpu().numpy().copy()}
                if not all(np.isfinite(value).all() for value in inputs.values()):
                    raise FloatingPointError(f'Nonfinite LFC input/output before step {step}')
                torch.testing.assert_close(action, reference, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(latent, expected_latent, atol=1e-6, rtol=1e-5)
                action_error = max(action_error, float((action-reference).abs().max()))
                latent_error = max(latent_error, float((latent-expected_latent).abs().max()))
                compared += 1
                observations, _, _, _ = env.step(action)
                if pending is None:
                    raise RuntimeError('No pre-reset LFC transition was delivered')
                pending.update(inputs)
                rows.append(pending)
                pending = None
                if not all(np.isfinite(value).all() for value in rows[-1].values()):
                    raise FloatingPointError(f'Nonfinite LFC physical transition at step {step}')
        status = 'completed'
    except BaseException as exc:
        if pending is not None:
            pending.update(inputs)
            rows.append(pending)
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        env.transition_observer = previous_observer
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]} if rows else {}
        np.savez_compressed(output/'trajectory.npz', **arrays)
        finite = np.ones((len(rows), env.num_envs), dtype=bool)
        for array in arrays.values():
            finite &= np.isfinite(array).reshape(len(rows), env.num_envs, -1).all(-1)
        summary = {'status': status, 'error': error, 'requested_policy_steps': steps,
            'saved_policy_steps': len(rows), 'num_envs': env.num_envs,
            'finite_policy_steps': int(finite.all(-1).sum()),
            'finite_environment_transitions': int(finite.sum()), 'reason_order': REASONS,
            'exported_policy_observed_steps': observed, 'compared_policy_steps': compared,
            'exported_policy_consumed': observed > 0,
            'exported_action_max_abs_error': action_error if compared else None,
            'exported_latent_max_abs_error': latent_error if compared else None,
            'scope': 'finite execution-window transitions including terminal states and automatic resets, not complete episodes or learned force control',
            'force_metric': 'known applied spring/external force versus source force command; not independent measured contact force',
            'torque_metric': 'source_pd_torques is computed source PD output; default physics uses position drive targets, so this is not measured drive effort'}
        if rows:
            summary['termination_counts'] = dict(zip(REASONS, arrays['reason_mask'].sum((0,1)).astype(int).tolist()))
            summary['done_transitions'] = int(arrays['done'].sum())
        if finite.any():
            modes = arrays['force_or_position_control'].reshape(len(rows), env.num_envs)
            freed = arrays['freed_envs'].reshape(len(rows), env.num_envs).astype(bool)
            position = finite & (modes == 0)
            force = finite & (modes == 1)
            constrained_force = force & ~freed
            summary.update(position_transitions=int(position.sum()), force_transitions=int(force.sum()),
                constrained_force_transitions=int(constrained_force.sum()),
                base_up_dot_min=float(arrays['base_up_dot'][finite].min()),
                velocity_xy_error_rms=float(np.sqrt(np.square(arrays['velocity_error_xy'][finite]).mean())))
            summary['position_mode_ee_error_mean_m'] = float(arrays['ee_position_error_m'][position].mean()) if position.any() else None
            for label, mask in [('force_mode', force), ('constrained_force_mode', constrained_force)]:
                summary[label+'_force_xyz_error_rms_N'] = float(np.sqrt(np.square(arrays['force_error_xyz_N'][mask]).mean())) if mask.any() else None
        (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary
