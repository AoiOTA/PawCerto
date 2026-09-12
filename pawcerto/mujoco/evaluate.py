"""Fixed-weight UMI trajectory replay on the actual floating Go2 + ARX5."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pawcerto.methods.umi_on_legs import RobotState, UmiPolicy
from .asset import ROOT
from .runtime import Go2Arx5Mujoco, SimulationInstability
from .contacts import contact_snapshot, FORCE_THRESHOLD_N


def robot_state(state):
    def batch(array):
        return torch.as_tensor(np.asarray(array), dtype=torch.float32).unsqueeze(0)
    pose = np.eye(4)
    pose[:3, :3] = state['ee_rotmat']
    pose[:3, 3] = state['ee_pos']
    return RobotState(batch(state['root_ang_vel_body']), batch(state['gravity_body']),
                      batch(state['joint_pos']), batch(state['joint_vel']),
                      batch(pose), torch.tensor([state['time']], dtype=torch.float32))


def evaluate(checkpoint_dir, trajectory_path, seconds=17., seed=0, *, policy=None, positions=None, rotations=None, raw_output=None, substep_trace=None):
    torch.set_num_threads(1)
    policy = UmiPolicy(checkpoint_dir) if policy is None else policy
    source = Path(checkpoint_dir)
    config_dir = source.parent if source.is_file() else source
    sim = Go2Arx5Mujoco(config_dir / 'config.json')
    if positions is None:
        positions, rotations = policy.trajectories(trajectory_path).sample(1, seed)
    observer = policy.observer(positions, rotations)
    last_action = torch.zeros(1, 18)
    # Original Env.reset performs one zero-action policy step before returning
    # the first actor observation, including the task pose-history update.
    warmup = sim.step(np.zeros(18), trace=substep_trace)
    observer.advance_pose(robot_state(warmup).ee_pose)
    records = []
    raw = []
    nonfoot_contacts = []
    foot_or_finger_contacts = []
    foot_ground_forces = []
    ground_supported_feet = []
    contact_sample_counts = {'external': 0, 'self': 0}
    any_contact_samples = 0
    failure = None
    for _ in range(round(seconds / (sim.model.opt.timestep * sim.decimation)) - 1):
        state = robot_state(sim.state())
        with torch.no_grad():
            action = policy.actor(observer.observe(state, last_action))
        try:
            physical = sim.step(action[0].numpy(), trace=substep_trace)
        except SimulationInstability as exc:
            if not records:
                raise  # preserve the numerical cause if no policy prefix exists
            failure = exc.details
            break
        current = robot_state(physical)
        observer.advance_pose(current.ee_pose)
        last_action = torch.from_numpy(sim.previous_action).float()[None]
        target_p, target_r = observer.targets(current.episode_time)
        p_error = np.linalg.norm(physical['ee_pos'] - target_p[0].numpy())
        relative_rotation = physical['ee_rotmat'].T @ target_r[0].numpy()
        angle_error = np.arccos(np.clip((np.trace(relative_rotation)-1)/2, -1, 1))
        foot_forces_z, foot_ground_z, contact_records = contact_snapshot(sim)
        nonfoot_records = [r for r in contact_records if r['nonfoot_nongripper']]
        nonfoot_contacts.extend(nonfoot_records)
        foot_or_finger_contacts.extend(r for r in contact_records if not r['nonfoot_nongripper'])
        foot_ground_forces.append(foot_ground_z)
        ground_supported_feet.append(int(np.sum(foot_ground_z > 1.)))
        active_kinds = {r['kind'] for r in nonfoot_records if r['force_norm_gt_1N']}
        any_contact_samples += bool(active_kinds)
        for kind in active_kinds:
            contact_sample_counts[kind] += 1
        supported_feet = int(np.sum(foot_forces_z > 1.))
        records.append([physical['time'], p_error, angle_error, physical['root_pos'][2],
                        physical['root_rotmat'][2,2], sim.data.ncon, np.max(np.abs(action.numpy())), supported_feet])
        raw.append(np.concatenate((physical['ee_pos'], physical['ee_rotmat'].ravel(),
                                   target_p[0].numpy(), target_r[0].numpy().ravel(),
                                   physical['root_pos'], physical['root_rotmat'].ravel(),
                                   physical['joint_pos'], physical['joint_vel'], action[0].numpy(),
                                   sim.last_torque, foot_forces_z)))
    values = np.asarray(records)
    if raw_output is not None:
        np.savez_compressed(raw_output, metrics=values, physical=np.asarray(raw),
                            foot_ground_force_z_N=np.asarray(foot_ground_forces),
                            ground_supported_feet=np.asarray(ground_supported_feet),
                            sampled_target_positions=positions.numpy(), sampled_target_rotations=rotations.numpy(),
                            physical_columns=np.asarray(['ee_pos:0:3', 'ee_rotmat:3:12', 'target_pos:12:15',
                                'target_rotmat:15:24', 'root_pos:24:27', 'root_rotmat:27:36',
                                'joint_pos:36:54', 'joint_vel:54:72', 'raw_action:72:90',
                                'last_substep_torque:90:108', 'FR_FL_RR_RL_force_z:108:112']))
    return {'checkpoint': str(checkpoint_dir), 'trajectory': str(trajectory_path), 'seed': seed,
            'joint_names': list(sim.joint_names), 'seconds': float(values[-1,0]),
            'requested_seconds': seconds, 'completion_status': 'invalid' if failure else 'complete',
            'metrics_scope': 'pre_failure_prefix' if failure else 'complete_episode',
            'failure': failure,
            'position_error_mean_m': float(values[:,1].mean()),
            'position_error_rms_m': float(np.sqrt(np.mean(values[:,1]**2))),
            'orientation_error_mean_rad': float(values[:,2].mean()),
            'position_error_max_m': float(values[:,1].max()),
            'orientation_error_max_rad': float(values[:,2].max()),
            'orientation_error_rms_rad': float(np.sqrt(np.mean(values[:,2]**2))),
            'inverted': bool(np.any(values[:,4] < 0)),
            'supported_feet_mean': float(values[:,7].mean()),
            'supported_feet_min': int(values[:,7].min()),
            'zero_supported_feet_fraction': float(np.mean(values[:,7] == 0)),
            'minimum_root_height_m': float(values[:,3].min()),
            'final_root_height_m': float(values[-1,3]), 'minimum_root_up_dot': float(values[:,4].min()),
            'maximum_absolute_raw_action': float(values[:,6].max()),
            'record_columns': ['time_s', 'ee_position_error_m', 'ee_orientation_error_rad',
                               'root_height_m', 'root_up_dot', 'contact_count', 'max_abs_raw_action', 'supported_feet_force_z_gt_1N'],
            'records': values.tolist(),
            'nonfoot_contact_records': nonfoot_contacts,
            'foot_or_finger_only_contact_records': foot_or_finger_contacts,
            'ground_supported_feet_records': ground_supported_feet,
            'ground_supported_feet_mean': float(np.mean(ground_supported_feet)),
            'ground_supported_feet_min': min(ground_supported_feet),
            'zero_ground_supported_feet_fraction': float(np.mean(np.asarray(ground_supported_feet) == 0)),
            'net_vs_ground_supported_feet_different_samples': int(np.sum(values[:,7] != ground_supported_feet)),
            'foot_force_definitions': {
                'supported_feet_legacy': 'Per-foot net world Fz over all contacts >1N, including self contacts',
                'ground_supported_feet': 'Per-foot world Fz summed only over contacts with named ground geom >1N',
                'contact_record_groups': 'nonfoot_contact_records plus foot_or_finger_only_contact_records together contain all sampled contacts'},
            'nonfoot_contact_sampling': {
                'sample_period_s': float(sim.model.opt.timestep * sim.decimation),
                'sample_location': 'Policy endpoint after runtime mj_forward, once per20ms; not every5ms physics substep',
                'force_threshold_N': FORCE_THRESHOLD_N,
                'force_threshold_measure': 'Per-contact world force Euclidean norm; strictly greater than threshold',
                'exclusions': 'Four named foot bodies and link6 collider-mesh finger geoms only; wrist/structural link6 geoms remain counted',
                'sampled_occupancy_s_is_not_continuous_duration': True},
            'nonfoot_contact_samples_gt_1N': any_contact_samples,
            'nonfoot_external_contact_samples_gt_1N': contact_sample_counts['external'],
            'nonfoot_self_contact_samples_gt_1N': contact_sample_counts['self'],
            'nonfoot_contact_sampled_occupancy_s': any_contact_samples * sim.model.opt.timestep * sim.decimation,
            'nonfoot_contact_max_force_N': max((r['force_norm_N'] for r in nonfoot_contacts), default=0.),
            'evidence': 'Independent MuJoCo fixed-weight diagnostic; neither original IsaacGym reproduction nor learned Isaac Lab WBC acceptance.'}


def summarize_cases(results, checkpoint_dir, trajectory_path, seconds, seed):
    complete = [r for r in results if r.get('completion_status', 'complete') == 'complete']
    invalid = [r for r in results if r.get('completion_status', 'complete') != 'complete']
    if not complete:
        return {'checkpoint': str(checkpoint_dir), 'seed': seed, 'cases': len(results),
                'requested_cases': len(results), 'complete_cases': 0, 'invalid_cases': len(invalid),
                'completion_status': 'incomplete', 'complete_case_metrics': None, 'case_results': results}
    measured = complete
    summary = {'checkpoint': str(checkpoint_dir), 'trajectory': str(trajectory_path),
               'seed': seed, 'cases': len(results), 'seconds_per_case': seconds,
               'position_error_mean_m': float(np.mean([r['position_error_mean_m'] for r in measured])),
               'position_error_rms_m': float(np.sqrt(np.mean([r['position_error_rms_m']**2 for r in measured]))),
               'orientation_error_mean_rad': float(np.mean([r['orientation_error_mean_rad'] for r in measured])),
               'position_error_max_m': max(r['position_error_max_m'] for r in measured),
               'minimum_root_up_dot': min(r['minimum_root_up_dot'] for r in measured),
               'inverted_cases': sum(r['inverted'] for r in measured),
               'supported_feet_mean': float(np.mean([r['supported_feet_mean'] for r in measured])),
               'zero_supported_feet_fraction': float(np.mean([r['zero_supported_feet_fraction'] for r in measured])),
               'ground_supported_feet_mean': float(np.mean([r['ground_supported_feet_mean'] for r in measured])),
               'zero_ground_supported_feet_fraction': float(np.mean([r['zero_ground_supported_feet_fraction'] for r in measured])),
               'net_vs_ground_supported_feet_different_samples': sum(r['net_vs_ground_supported_feet_different_samples'] for r in measured),
               'foot_force_definitions': results[0]['foot_force_definitions'],
               'nonfoot_contact_cases_gt_1N': sum(r['nonfoot_contact_samples_gt_1N'] > 0 for r in measured),
               'nonfoot_external_contact_samples_gt_1N': sum(r['nonfoot_external_contact_samples_gt_1N'] for r in measured),
               'nonfoot_self_contact_samples_gt_1N': sum(r['nonfoot_self_contact_samples_gt_1N'] for r in measured),
               'nonfoot_contact_sampled_occupancy_s_sum': sum(r['nonfoot_contact_sampled_occupancy_s'] for r in measured),
               'nonfoot_contact_sampling': results[0]['nonfoot_contact_sampling'],
               'case_results': results,
               'evidence': 'Paired diagnostic on sampled training-data trajectories, not a holdout success rate; unchanged MuJoCo physics, fixed weights, no new success threshold.'}
    summary['requested_cases'] = len(results)
    summary['complete_cases'] = len(complete)
    summary['invalid_cases'] = len(invalid)
    summary['completion_status'] = 'incomplete' if invalid else 'complete'
    if invalid:
        metric_keys = [key for key in summary if key not in (
            'checkpoint', 'trajectory', 'seed', 'cases', 'seconds_per_case', 'case_results',
            'evidence', 'foot_force_definitions', 'nonfoot_contact_sampling', 'requested_cases',
            'complete_cases', 'invalid_cases', 'completion_status')]
        summary['complete_case_metrics'] = {key: summary.pop(key) for key in metric_keys}
        summary['invalid_case_indices'] = [r['case_index'] for r in invalid]
        summary['evidence'] = ('Requested cases include numerical failures; complete-episode metrics are '
                               'reported only for the complete subset. Invalid prefixes remain in case_results, '
                               'not pooled into full-episode means. Training-data diagnostic, not holdout.')
    return summary


def evaluate_batch(checkpoint_dir, trajectory_path, output_dir, cases=16, seconds=17., seed=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = UmiPolicy(checkpoint_dir)
    # One shared draw is essential: sample(1, case_seed) is not this paired set.
    positions, rotations = policy.trajectories(trajectory_path).sample(cases, seed)
    results = []
    for index in range(cases):
        result = evaluate(checkpoint_dir, trajectory_path, seconds, seed, policy=policy,
                          positions=positions[index:index+1], rotations=rotations[index:index+1],
                          raw_output=output_dir / f'case_{index:02d}.npz')
        result['case_index'] = index
        (output_dir / f'case_{index:02d}.json').write_text(json.dumps(result, indent=2))
        results.append({key: value for key, value in result.items() if key not in ('records', 'nonfoot_contact_records', 'foot_or_finger_only_contact_records', 'ground_supported_feet_records')})
    summary = summarize_cases(results, checkpoint_dir, trajectory_path, seconds, seed)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'reference/checkpoints/tossing/ours')
    parser.add_argument('--trajectory', type=Path, default=ROOT / 'reference/data/tossing.pkl')
    parser.add_argument('--seconds', type=float, default=17.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/mujoco/evaluation.json')
    parser.add_argument('--cases', type=int, default=1, help='Batch draws once with sample(cases, seed); --output is then a directory')
    args = parser.parse_args()
    if args.cases == 1:
        result = evaluate(args.checkpoint, args.trajectory, args.seconds, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    else:
        result = evaluate_batch(args.checkpoint, args.trajectory, args.output, args.cases, args.seconds, args.seed)
    print(json.dumps({key: value for key, value in result.items() if key not in ('records', 'case_results', 'nonfoot_contact_records', 'foot_or_finger_only_contact_records', 'ground_supported_feet_records')}, indent=2))

    if result.get("completion_status") in ("invalid", "incomplete"):
        raise SystemExit(2)
