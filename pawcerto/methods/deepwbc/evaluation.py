"""Fixed-task evaluation and pre-reset records; no simulator imports.

The ordinary task remains responsible for goal interpolation/resampling, rewards,
termination and reset. Diagnostics never call reward methods (they accumulate).
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from .task import DeepWBCTask, evaluation_config, euler_from_quat, quat_rotate_inverse
from .observer import goal_world, cartesian_to_sphere

REASONS = ('contact', 'goal_signed_roll', 'goal_signed_pitch', 'base_height', 'timeout')


def fixed_task_config(config, task, *, num_envs=1, seed=0, profile='original-play'):
    """Fix sampler ranges, not observation or physics; retain the source deadband."""
    required = {'commands', 'ee_goal_lpy', 'traj_time_s', 'hold_time_s'}
    if set(task) != required:
        raise ValueError(f'Task requires exactly {sorted(required)}')
    command = np.asarray(task['commands'], dtype=float)
    goal = np.asarray(task['ee_goal_lpy'], dtype=float)
    if command.shape != (3,) or goal.shape != (3,) or not np.isfinite(np.r_[command, goal]).all():
        raise ValueError('commands and ee_goal_lpy must be finite three-vectors')
    if command[1] != 0 or goal[0] <= 0:
        raise ValueError('Source fixes lateral command to zero; LPY length must be positive')
    times = np.asarray([task['traj_time_s'], task['hold_time_s']], dtype=float)
    if not np.isfinite(times).all() or times[0] <= 0 or times[1] < 0:
        raise ValueError('Require finite traj_time_s > 0 and hold_time_s >= 0')
    if num_envs < 1 or profile not in ('original-play', 'checkpoint'):
        raise ValueError('Invalid num_envs or evaluation profile')
    c = evaluation_config(config) if profile == 'original-play' else deepcopy(config)
    c['env']['num_envs'], c['seed'] = num_envs, seed
    # Source play calls update_command_curriculum once; fixed tasks use final
    # reward scales but degenerate initial AND final command/goal ranges.
    for stem, value in (('lin_vel_x', command[0]), ('ang_vel_yaw', command[2])):
        for prefix in ('init_', 'final_'):
            c['commands']['ranges'][prefix + stem] = [float(value)] * 2
    for axis, value in zip('lpy', goal):
        for prefix in ('init_', 'final_'):
            c['goal_ee']['ranges'][prefix + 'pos_' + axis] = [float(value)] * 2
    c['goal_ee']['traj_time'] = [float(times[0])] * 2
    c['goal_ee']['hold_time'] = [float(times[1])] * 2
    # Retain source terrain placement, noise, randomization and action delay.
    # Reject impossible fixed paths instead of exhausting source rejection ten
    # times and quietly using its last colliding sample.
    with torch.random.fork_rng(devices=[]):
        probe = DeepWBCTask(1, c, 'cpu')
        probe.ee_goal_sphere[:] = torch.tensor(goal, dtype=torch.float32)
        ids = torch.tensor([0])
        if probe.collision_check(ids).any():
            raise ValueError('Fixed goal intersects source collision/underground checks from initial zero goal')
        probe.ee_start_sphere.copy_(probe.ee_goal_sphere)
        if probe.collision_check(ids).any():
            raise ValueError('Fixed goal fails source collision/underground checks')
    return c


def transition_snapshot(env, leg, arm, done, extras):
    """Copy post-physics/pre-push/pre-reset state, including source reason masks."""
    t = env.task
    roll, pitch, yaw = euler_from_quat(t.base_quat)
    target = goal_world(t.curr_ee_goal_sphere, t.root_states[:, :2], yaw)
    local = quat_rotate_inverse(t.base_yaw_quat,
        t.ee_pos - torch.cat((t.root_states[:, :2], t.z_invariant_offset), -1))
    reasons = torch.stack((
        (t.contact_forces[:, t.termination_contact_indices].norm(dim=-1) > 1).any(dim=-1),
        ((roll > .2) & (t.curr_ee_goal[:, 2] >= 0)) | ((roll < -.2) & (t.curr_ee_goal[:, 2] <= 0)),
        ((pitch > .2) & (t.curr_ee_goal[:, 1] >= 0)) | ((pitch < -.2) & (t.curr_ee_goal[:, 1] <= 0)),
        t.root_states[:, 2] < t.cfg.termination.z_threshold, t.time_out_buf), -1)
    if not torch.equal(reasons.any(-1), done):
        raise RuntimeError('Evaluation reason masks diverged from task termination')
    values = {key: getattr(t, key) for key in ('root_states','dof_pos','dof_vel','torques',
        'actions','ee_pos','ee_orn','base_lin_vel','base_ang_vel','commands',
        'curr_ee_goal_sphere','ee_goal_sphere','ee_goal_orn_euler','goal_timer','episode_length_buf',
        'contact_forces')}
    values.update(foot_wrenches=t.force_sensor_tensor, ee_target_world=target,
        ee_error_l2_m=(t.ee_pos-target).norm(dim=-1),
        ee_error_l1_m=(t.ee_pos-target).abs().sum(-1),
        ee_error_sphere_weighted=(cartesian_to_sphere(local)-t.curr_ee_goal_sphere).abs().mul(t.sphere_error_scale).sum(-1),
        velocity_error=t.base_lin_vel[:, 0]-t.commands[:, 0],
        yaw_rate_error=t.base_ang_vel[:, 2]-t.commands[:, 2],
        reward_leg=leg, reward_arm=arm, done=done, reason_mask=reasons,
        latest_sim_actions=env.runtime.controller.history[:, -1])
    for key in ('ee_lin_vel', 'ee_ang_vel', 'time'):
        if key in env.state:
            values[key] = env.state[key]
    values['push_applied'] = torch.full_like(done, 'push_velocity_xy' in extras)
    values['push_velocity_xy'] = extras.get('push_velocity_xy', torch.zeros_like(t.commands[:, :2]))
    return {key: value.detach().cpu().numpy().copy() for key, value in values.items()}


class TrajectoryWriter:
    """Bounded-memory NPZ chunks plus per-episode JSONL; refuses overwrites."""
    def __init__(self, output, metadata, *, chunk_steps=256):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        if chunk_steps < 1:
            raise ValueError('chunk_steps must be positive')
        self.chunk_steps, self.rows, self.chunks, self.steps = chunk_steps, [], 0, 0
        self.metadata = metadata
        self.episodes = self.failures = self.timeouts = 0
        self.sums = dict(ee_squared=0., velocity_abs=0., yaw_abs=0.)
        self.samples = 0
        (self.output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
        self.events = (self.output/'episodes.jsonl').open('w')
        self.episode_ids = None
        self.episode_totals = None

    def append(self, row):
        n = len(row['done'])
        if self.episode_ids is None:
            self.episode_ids = np.zeros(n, dtype=np.int64)
            self.episode_totals = np.zeros((n, 4), dtype=float)
        row['episode_id'] = self.episode_ids.copy()
        row['step'] = np.full(n, self.steps, dtype=np.int64)
        finite = np.ones(n, dtype=bool)
        for value in row.values():
            if np.issubdtype(value.dtype, np.floating):
                finite &= np.isfinite(value).reshape(n, -1).all(-1)
        row['finite'] = finite
        self.rows.append(row)
        self.steps += 1
        self.episode_totals += np.stack((np.ones(n), row['reward_leg'], row['reward_arm'], row['ee_error_l2_m']**2), -1)
        for i in np.flatnonzero(row['done'] | ~finite):
            reasons = [name for name, flag in zip(REASONS, row['reason_mask'][i]) if flag]
            if not finite[i]:
                reasons.append('nonfinite_transition')
            count, leg, arm, ee = self.episode_totals[i]
            event = dict(env_id=int(i), episode_id=int(self.episode_ids[i]), step=self.steps-1,
                samples=int(count), reasons=reasons, reward_leg=float(leg) if np.isfinite(leg) else None,
                reward_arm=float(arm) if np.isfinite(arm) else None,
                ee_rmse_m=float(np.sqrt(ee/count)) if np.isfinite(ee) else None)
            self.events.write(json.dumps(event, allow_nan=False)+'\n')
            self.events.flush()
            self.episodes += int(row['done'][i])
            self.timeouts += int(row['reason_mask'][i, -1])
            self.failures += int(row['reason_mask'][i, :-1].any() or not finite[i])
            self.episode_ids[i] += 1
            self.episode_totals[i] = 0
        if finite.all():
            self.samples += n
            self.sums['ee_squared'] += float((row['ee_error_l2_m']**2).sum())
            self.sums['velocity_abs'] += float(np.abs(row['velocity_error']).sum())
            self.sums['yaw_abs'] += float(np.abs(row['yaw_rate_error']).sum())
        if len(self.rows) >= self.chunk_steps:
            self.flush()
        if not finite.all():
            raise FloatingPointError('Nonfinite transition saved; evaluation stopped without changing task done')

    def flush(self):
        if self.rows:
            np.savez_compressed(self.output/f'trajectory_{self.chunks:05d}.npz',
                **{key: np.stack([r[key] for r in self.rows]) for key in self.rows[0]})
            self.chunks += 1
            self.rows.clear()

    def close(self, *, status, error=None):
        self.flush()
        self.events.close()
        summary = dict(status=status, error=error, steps=self.steps, chunks=self.chunks,
            completed_episodes=self.episodes, failure_events=self.failures, timeout_events=self.timeouts,
            evidence_source=self.metadata['evidence_source'],
            task_success='not inferred; no acceptance threshold defined',
            ee_rmse_m=float(np.sqrt(self.sums['ee_squared']/self.samples)) if self.samples else None,
            velocity_mae_m_s=self.sums['velocity_abs']/self.samples if self.samples else None,
            yaw_rate_mae_rad_s=self.sums['yaw_abs']/self.samples if self.samples else None,
            unfinished_episode_samples=self.episode_totals[:, 0].astype(int).tolist() if self.episode_totals is not None else [])
        (self.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        return summary


def evaluate_fixed_policy(env, policy, *, steps, writer, exported_policy=None):
    """Evaluate mean/history policy; the runtime is provided by the real caller."""
    if steps < 1:
        raise ValueError('steps must be positive')
    status, error = 'failed', None
    try:
        with torch.inference_mode():
            env.update_command_curriculum()
            obs, _ = env.reset()  # source zero-action warmup; not counted in trace
            for _ in range(steps):
                actions = policy(obs)
                if actions.shape != (env.num_envs, env.num_actions) or not torch.isfinite(actions).all():
                    raise FloatingPointError('Invalid policy action before physics step')
                original_obs = obs.detach().cpu().numpy().copy()
                policy_actions = actions.detach().cpu().numpy().copy()
                exported = None
                if exported_policy is not None:
                    exported = exported_policy(obs)
                    if exported.shape != actions.shape or not torch.isfinite(exported).all():
                        raise FloatingPointError('Invalid exported policy action')
                def capture(e, leg, arm, done, extras):
                    row = transition_snapshot(e, leg, arm, done, extras)
                    row.update(observations=original_obs, policy_actions=policy_actions)
                    if exported is not None:
                        row['exported_actions'] = exported.detach().cpu().numpy().copy()
                        row['export_action_max_abs_error'] = np.max(np.abs(row['exported_actions']-policy_actions), axis=-1)
                    writer.append(row)
                obs, *_ = env.step(actions, transition_callback=capture)
            status = 'completed'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        summary = writer.close(status=status, error=error)
    return summary
