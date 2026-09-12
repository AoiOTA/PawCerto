"""Fixed-command UniFP execution records, using the original task and resets."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from .math import quat_apply


REASONS = ('termination_contact', 'pitch', 'roll', 'timeout')


def fixed_task_config(config, task, *, num_envs=1):
    required = {'commands', 'ee_goal_lpy', 'traj_time_s', 'hold_time_s'}
    if set(task) != required:
        raise ValueError(f'Task requires exactly {sorted(required)}')
    commands = np.asarray(task['commands'], dtype=float)
    goal = np.asarray(task['ee_goal_lpy'], dtype=float)
    times = np.asarray([task['traj_time_s'], task['hold_time_s']], dtype=float)
    if (commands.shape != (3,) or goal.shape != (3,)
            or not np.isfinite(np.r_[commands, goal, times]).all()
            or goal[0] <= 0 or times[0] <= 0 or times[1] < 0 or num_envs < 1):
        raise ValueError('Require finite 3-vector commands/goal, positive radius/traj time and nonnegative hold time')
    result = deepcopy(config)
    env = result['env']
    env['env']['num_envs'] = num_envs
    for name, value in zip(('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw'), commands):
        env['commands']['ranges'][name] = [float(value)] * 2
    env['commands']['zero_vel_cmd_prob'] = 0.
    for name, value in zip(('pos_l', 'pos_p', 'pos_y'), goal):
        env['goal_ee']['ranges'][name] = [float(value)] * 2
    env['goal_ee']['traj_time'] = [float(times[0])] * 2
    env['goal_ee']['hold_time'] = [float(times[1])] * 2
    return result


def transition_snapshot(env):
    """Read once after reward/termination and before reset; no reward side effects."""
    reason_mask = torch.stack((
        (env.contact_forces[:, env.termination_contact_indices].norm(dim=-1) > 1).any(-1),
        env.base_euler_xyz[:, 1].abs() > 1.,
        env.base_euler_xyz[:, 0].abs() > .8,
        env.time_out_buf,
    ), -1)
    if not torch.equal(reason_mask.any(-1), env.reset_buf.bool()):
        raise RuntimeError('Saved reason masks differ from original task termination')
    injected = env.forces[:, env.gripper_idx, :3]
    commanded_world = quat_apply(env.base_yaw_quat, env.current_Fxyz_gripper_cmd)
    target = env.curr_ee_goal_cart_world + (injected + commanded_world) / env.gripper_force_kps
    values = {key: getattr(env, key) for key in (
        'root_states', 'dof_pos', 'dof_vel', 'torques', 'actions', 'commands',
        'base_lin_vel', 'base_ang_vel', 'base_euler_xyz', 'ee_pos',
        'curr_ee_goal_cart_world', 'curr_ee_goal_sphere', 'forces', 'contact_forces',
        'current_Fxyz_gripper_cmd', 'gripper_force_kps', 'episode_length_buf', 'rew_buf')}
    values.update(reason_mask=reason_mask, done=env.reset_buf.bool(),
        global_steps_during_physics=torch.full((env.num_envs,), env.global_steps,
                                              dtype=torch.int64, device=env.device),
        ee_compliance_target_world=target,
        ee_position_error_m=(env.ee_pos - env.curr_ee_goal_cart_world).norm(dim=-1),
        ee_compliance_error_m=(env.ee_pos - target).norm(dim=-1),
        base_velocity_error=env.base_lin_vel[:, :2] - env.commands[:, :2],
        yaw_rate_error=env.base_ang_vel[:, 2] - env.commands[:, 2],
        base_up_dot=1 - 2 * env.base_quat[:, :2].square().sum(-1))
    return {key: value.detach().cpu().numpy().copy() for key, value in values.items()}


def evaluate_fixed_policy(env, policy, *, steps, output, metadata, exported_policy=None):
    """Record a bounded execution window, retaining every reset and invalid prefix.

    Aggregates cover finite recorded transitions including terminal endpoints;
    they are not full-episode means or a learned-control acceptance result.
    """
    if steps < 1:
        raise ValueError('steps must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    rows = []
    parity_max = 0.
    estimate_parity_max = 0.
    compared_steps = 0
    exported_observed_steps = 0
    status, error = 'failed', None
    previous_observer = env.transition_observer
    pending = None

    def observe(current):
        nonlocal pending, exported_observed_steps
        if pending is not None:
            raise RuntimeError('Multiple transition snapshots for one policy step')
        pending = transition_snapshot(current)
        if exported_policy is not None:
            exported_observed_steps += 1

    try:
        # Reset uses source warmup and initialization; attach the observer afterward.
        env.reset()
        env.transition_observer = observe
        observations = env.get_observations()
        with torch.inference_mode():
            for step in range(steps):
                history = observations['obs']
                action = policy.act_inference(observations)
                estimates = policy.get_student_latent(history)
                input_row = {
                    'observations': history.detach().cpu().numpy().copy(),
                    'supervision': observations['obs_pred'].detach().cpu().numpy().copy(),
                    'estimates': estimates.detach().cpu().numpy().copy(),
                    'raw_actions': action.detach().cpu().numpy().copy(),
                }
                if not all(np.isfinite(value).all() for value in input_row.values()):
                    raise FloatingPointError(f'Nonfinite policy input/output before step {step}')
                if exported_policy is not None:
                    exported_action = exported_policy.act_inference(observations)
                    exported_estimates = exported_policy.get_student_latent(history)
                    torch.testing.assert_close(exported_action, action, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(exported_estimates, estimates, atol=1e-6, rtol=1e-5)
                    parity_max = max(parity_max, float((exported_action - action).abs().max()))
                    estimate_parity_max = max(estimate_parity_max, float((exported_estimates - estimates).abs().max()))
                    compared_steps += 1
                    # The actual simulation consumes the exported module's action.
                    action = exported_action
                observations, _, _, _ = env.step(action)
                if pending is None:
                    raise RuntimeError('Environment did not provide a pre-reset transition')
                pending.update(input_row)
                rows.append(pending)
                pending = None
                if not all(np.isfinite(value).all() for value in rows[-1].values()):
                    raise FloatingPointError(f'Nonfinite physical transition at step {step}')
        status = 'completed'
    except BaseException as exc:
        if pending is not None:
            pending.update(input_row)
            rows.append(pending)
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        env.transition_observer = previous_observer
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]} if rows else {}
        np.savez_compressed(output / 'trajectory.npz', **arrays)
        finite = np.ones((len(rows), env.num_envs), dtype=bool)
        for array in arrays.values():
            finite &= np.isfinite(array).reshape(len(rows), env.num_envs, -1).all(-1)
        summary = {
            'status': status, 'error': error, 'requested_policy_steps': steps,
            'saved_policy_steps': len(rows), 'finite_policy_steps': int(finite.all(-1).sum()),
            'finite_environment_transitions': int(finite.sum()),
            'num_envs': env.num_envs, 'reason_order': REASONS,
            'scope': 'execution window with original automatic resets; means include all finite transitions and terminal endpoints, not complete-episode or learned-control acceptance',
            'exported_policy_consumed': exported_observed_steps > 0,
            'exported_policy_observed_steps': exported_observed_steps,
            'compared_policy_steps': compared_steps,
            'exported_action_max_abs_error': parity_max if compared_steps else None,
            'exported_estimate_max_abs_error': estimate_parity_max if compared_steps else None,
        }
        if rows:
            summary['termination_counts'] = dict(zip(REASONS, arrays['reason_mask'].sum(axis=(0, 1)).astype(int).tolist()))
            summary['done_transitions'] = int(arrays['done'].sum())
        if finite.any():
            for key in ('ee_position_error_m', 'ee_compliance_error_m'):
                summary[key + '_mean_finite_transitions'] = float(arrays[key][finite].mean())
            summary['base_up_dot_min_finite_transitions'] = float(arrays['base_up_dot'][finite].min())
            summary['velocity_xy_error_rms_finite_transitions'] = float(np.sqrt(np.square(arrays['base_velocity_error'][finite]).mean()))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary
