"""Bounded CPU FK readout of saved seed2027 UMI trajectories; never steps physics."""
from pathlib import Path
import argparse
import json
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pawcerto.mujoco.runtime import Go2Arx5Mujoco
from pawcerto.mujoco.contacts import FEET


def length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum())


def motion(points):
    return {'net_m': float(np.linalg.norm(points[-1]-points[0])),
            'path_m': length(points), 'axis_range_m': np.ptp(points, axis=0).tolist(),
            'max_distance_from_first_m': float(np.linalg.norm(points-points[0], axis=-1).max())}


def distribution(values):
    return dict(zip(('min', 'median', 'max'), map(float, np.quantile(values, (0, .5, 1)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-config', type=Path, required=True)
    parser.add_argument('--reference-config', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    inputs = {'official_ours': args.reference, 'candidate': args.policy}
    configs = {'official_ours': args.reference_config, 'candidate': args.policy_config}
    results = []
    for label in inputs:
        cases = []
        for index in range(16):
            path = inputs[label]/f'case_{index:02d}'
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['completion_status'] == 'complete'
            sim = Go2Arx5Mujoco(configs[label])
            model, data = sim.model, sim.data
            foot_geoms = []
            for name in FEET:
                ids = np.flatnonzero(model.geom_bodyid == model.body(name).id)
                assert len(ids) == 1 and model.geom_type[ids[0]] == mujoco.mjtGeom.mjGEOM_SPHERE
                foot_geoms.append(int(ids[0]))
            with np.load(path.with_suffix('.npz')) as raw:
                p, metrics = raw['physical'], raw['metrics']
                support = raw['foot_ground_force_z_N'] > 1
                assert len(p) == 849 and abs(metrics[-1, 0]-17) < 1e-8
                assert np.all(np.diff(metrics[:, 0]) > 0) and np.isfinite(p).all()
                foot_xyz = np.empty((len(p), 4, 3))
                reconstructed_ee = np.empty((len(p), 3))
                for j, row in enumerate(p):
                    data.qpos[:3] = row[24:27]
                    data.qpos[3:7] = Rotation.from_matrix(row[27:36].reshape(3, 3)).as_quat()[[3, 0, 1, 2]]
                    data.qpos[sim.qadr] = row[36:54]
                    mujoco.mj_kinematics(model, data)
                    foot_xyz[j] = data.geom_xpos[foot_geoms]
                    reconstructed_ee[j] = data.body('end_effector').xpos
                fk_error = float(np.max(np.abs(reconstructed_ee-p[:, :3])))
                assert fk_error < 1e-7, fk_error
                base = p[:, 24:27]
                rot = p[:, 27:36].reshape(-1, 3, 3)
                relative = np.einsum('nji,njk->nik', rot[:1], rot)
                angle = Rotation.from_matrix(relative).magnitude()
                euler = np.unwrap(Rotation.from_matrix(rot).as_euler('xyz'), axis=0)
                ee_in_base = np.einsum('nji,nj->ni', rot, p[:, :3]-base)
                clearance = foot_xyz[:, :, 2] - model.geom_size[foot_geoms, 0]
                displacement = np.linalg.norm(np.diff(foot_xyz[:, :, :2], axis=0), axis=-1)
                feet = {}
                for f, name in enumerate(FEET):
                    both_supported = support[:-1, f] & support[1:, f]
                    dxy = displacement[:, f][both_supported]
                    feet[name] = {
                        'unsupported_endpoint_samples': int((~support[:, f]).sum()),
                        'support_to_no_support_transitions': int((support[:-1, f] & ~support[1:, f]).sum()),
                        'no_support_to_support_transitions': int((~support[:-1, f] & support[1:, f]).sum()),
                        'sphere_ground_clearance_min_m': float(clearance[:, f].min()),
                        'sphere_ground_clearance_max_m': float(clearance[:, f].max()),
                        'max_clearance_time_s': float(metrics[np.argmax(clearance[:, f]), 0]),
                        'clearance_max_after_first_second_m': float(clearance[metrics[:, 0] >= 1, f].max()),
                        'max_clearance_at_unsupported_endpoint_m': float(clearance[~support[:, f], f].max()) if (~support[:, f]).any() else None,
                        'xy_net_displacement_m': float(np.linalg.norm(foot_xyz[-1, f, :2]-foot_xyz[0, f, :2])),
                        'xy_path_m': length(foot_xyz[:, f, :2]),
                        'adjacent_supported_endpoint_pairs': int(both_supported.sum()),
                        'xy_movement_sum_between_supported_endpoints_m': float(dxy.sum()),
                        'xy_movement_max_between_supported_endpoints_m': float(dxy.max()) if len(dxy) else None,
                    }
                cases.append({
                    'case_index': index, 'saved_start_s': float(metrics[0, 0]), 'saved_end_s': float(metrics[-1, 0]),
                    'target_translation': motion(p[:, 12:15]), 'base_xy': motion(base[:, :2]),
                    'base_height_min_m': float(base[:, 2].min()), 'base_height_max_m': float(base[:, 2].max()),
                    'base_height_range_m': float(np.ptp(base[:, 2])),
                    'base_orientation_net_deg': float(np.degrees(angle[-1])),
                    'base_orientation_max_from_first_deg': float(np.degrees(angle.max())),
                    'base_roll_pitch_yaw_range_deg': np.degrees(np.ptp(euler, axis=0)).tolist(),
                    'ee_motion_in_base_coordinates': motion(ee_in_base),
                    'arm_joint_ranges_rad': np.ptp(p[:, 48:54], axis=0).tolist(),
                    'feet': feet, 'ee_position_mean_m': meta['position_error_mean_m'],
                    'ee_orientation_mean_rad': meta['orientation_error_mean_rad'],
                    'completion_status':meta['completion_status'],'metrics_scope':meta['metrics_scope'],
                    'inverted':meta['inverted'],'minimum_root_up_dot':meta['minimum_root_up_dot'],
                    'ground_supported_feet_mean':meta['ground_supported_feet_mean'],
                    'zero_ground_supported_feet_fraction':meta['zero_ground_supported_feet_fraction'],
                    'nonfoot_external_samples_gt_1N': meta['nonfoot_external_contact_samples_gt_1N'],
                    'nonfoot_self_samples_gt_1N': meta['nonfoot_self_contact_samples_gt_1N'],
                    'nonfoot_external_bodies': sorted({name for c in meta['nonfoot_contact_records'] if c['kind']=='external' and c['force_norm_gt_1N'] for name in c['body_names'] if name!='world'}),
                    'fk_ee_max_abs_error_m': fk_error,
                })
        representative = max(cases, key=lambda c:c['base_xy']['path_m'])
        results.append({'policy': label, 'cases': cases,
                        'case_distribution': {
                            'target_max_distance_from_first_m': distribution([c['target_translation']['max_distance_from_first_m'] for c in cases]),
                            'base_xy_net_m': distribution([c['base_xy']['net_m'] for c in cases]),
                            'base_xy_path_m': distribution([c['base_xy']['path_m'] for c in cases]),
                            'base_height_range_m': distribution([c['base_height_range_m'] for c in cases]),
                            'base_orientation_max_change_deg': distribution([c['base_orientation_max_from_first_deg'] for c in cases]),
                            'foot_support_loss_transitions_sum': distribution([sum(f['support_to_no_support_transitions'] for f in c['feet'].values()) for c in cases]),
                            'max_foot_clearance_after_first_second_m': distribution([max(f['clearance_max_after_first_second_m'] for f in c['feet'].values()) for c in cases]),
                            'max_foot_clearance_m': distribution([max(f['sphere_ground_clearance_max_m'] for f in c['feet'].values()) for c in cases]),
                            'all_feet_xy_movement_sum_between_supported_endpoints_m': distribution([sum(f['xy_movement_sum_between_supported_endpoints_m'] for f in c['feet'].values()) for c in cases]),
                        }, 'representative_case_by_largest_base_xy_path': representative['case_index']})
    result = {'source': 'Existing complete sample(16,2027) records, user-supplied reference and candidate. Current variant retains native6D numerical and nearzeroEMD limitations.',
              'method': 'CPU original-model mj_kinematics only; no mj_step, new policy rollout, renderer, or GPU. Foot position is the center of each original collision sphere; clearance is centerZ minus radius above the fixed z=0 ground plane.',
              'sampling_boundary': 'Saved endpoints0.04..17s,20ms apart. Ground support is the existing summed ground-only Fz>1N. Unsupported does not by itself mean geometrically airborne. Adjacent supported endpoint movement does not establish continuous contact, sliding, stance, or normal stepping between endpoints. Sphere-center movement also cannot distinguish rolling from sliding of the material contact point.',
              'inference_boundary': 'Base displacement and arm motion document simultaneous behavior, not causal necessity, coordinated-gait quality, or a new success criterion. Motion paths are sampled sums and can include oscillation. Target amplitude is observed endpoint translation from first saved target; not a commanded base displacement.',
              'results': results}
    (root/'coordination_readout.json').write_text(json.dumps(result, indent=2)+'\n')
    lines=['# Saved-record leg/arm coordination readout','', result['source'],'',result['method'],'',result['sampling_boundary'],'',result['inference_boundary']]
    for group in results:
        dist=group['case_distribution']
        lines += ['', '## '+group['policy'], '', f'Across16 cases, base XY net displacement min/median/max: {dist["base_xy_net_m"]}; sampled path length: {dist["base_xy_path_m"]}. Maximum foot clearance after the first second: {dist["max_foot_clearance_after_first_second_m"]} m. The1s split is only a descriptive check separating startup from later motion, not a success criterion.', '', '| Case | Target max excursion m | Base XY net / path m | Base height range m | Max attitude change deg | Support-loss transitions | Max foot clearance mm | EE mean mm | Nonfoot external/self samples | Inverted | Mean ground feet / zero fraction |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|']
        for c in group['cases']:
            lines.append(f'| {c["case_index"]} | {c["target_translation"]["max_distance_from_first_m"]:.3f} | {c["base_xy"]["net_m"]:.3f} / {c["base_xy"]["path_m"]:.3f} | {c["base_height_range_m"]:.3f} | {c["base_orientation_max_from_first_deg"]:.1f} | {sum(f["support_to_no_support_transitions"] for f in c["feet"].values())} | {1000*max(f["sphere_ground_clearance_max_m"] for f in c["feet"].values()):.1f} | {1000*c["ee_position_mean_m"]:.2f} | {c["nonfoot_external_samples_gt_1N"]}/{c["nonfoot_self_samples_gt_1N"]} | {c["inverted"]} | {c["ground_supported_feet_mean"]:.3f} / {c["zero_ground_supported_feet_fraction"]:.4f} |')
        c=group['cases'][group['representative_case_by_largest_base_xy_path']]
        lines += ['',f'Representative case {c["case_index"]} is selected by the largest observed base XY path, not tracking success. Base net/path displacement: {c["base_xy"]["net_m"]:.4f}/{c["base_xy"]["path_m"]:.4f} m. EE motion in base coordinates has path {c["ee_motion_in_base_coordinates"]["path_m"]:.3f} m. Nonfoot ground bodies: {", ".join(c["nonfoot_external_bodies"]) or "none"}.', '', '| Foot | Unsupported endpoints | Support loss / regain | Max clearance mm @ time s; after1s mm | Supported endpoint pairs | XY movement sum / max between those endpoints mm |', '|---|---:|---:|---:|---:|---:|']
        for name,f in c['feet'].items():
            lines.append(f'| {name} | {f["unsupported_endpoint_samples"]} | {f["support_to_no_support_transitions"]} / {f["no_support_to_support_transitions"]} | {1000*f["sphere_ground_clearance_max_m"]:.2f} @ {f["max_clearance_time_s"]:.2f}; {1000*f["clearance_max_after_first_second_m"]:.2f} | {f["adjacent_supported_endpoint_pairs"]} | {1000*f["xy_movement_sum_between_supported_endpoints_m"]:.2f} / {1000*(f["xy_movement_max_between_supported_endpoints_m"] or 0):.2f} |')
    lines+=['','The JSON retains all per-case/per-foot values and min/median/max case distributions. Foot FK reconstruction was checked against each saved EE position (<1e-7m absolute error). A full5ms foot trajectory and material-point contact velocity are not present; no continuous-contact or gait-quality claim is made.']
    (root/'coordination_readout.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:v for k,v in x.items() if k!='cases'} for x in results],indent=2))


if __name__ == '__main__':
    main()
