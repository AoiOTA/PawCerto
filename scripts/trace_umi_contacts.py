"""Read every existing MuJoCo solve, retaining the unchanged paired replay."""
import argparse
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pawcerto.methods.umi_on_legs import UmiPolicy
from pawcerto.mujoco.evaluate import evaluate
from pawcerto.mujoco.runtime import Go2Arx5Mujoco


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class SubstepContactSummary:
    """Aggregate the existing trace callback; separate kinematics only for slip."""
    def __init__(self, config_path):
        from pawcerto.mujoco.asset import DEFAULT_MODEL
        from pawcerto.mujoco.runtime import JOINT_NAMES
        self.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
        self.data = mujoco.MjData(self.model)
        self.dt = json.loads(Path(config_path).read_text())['env']['cfg']['sim']['dt']
        self.vadr = [int(self.model.joint(n).dofadr[0]) for n in JOINT_NAMES]
        self.count = 0
        self.ground_sum = self.zero_ground = self.inverted = self.head_count = 0
        self.up_min = 1.
        self.max_joint_speed = 0.
        self.saturation = np.zeros(len(JOINT_NAMES), dtype=np.int64)
        self.joint_names = list(JOINT_NAMES)
        self.perbody = {}
        self.head_peak = 0.
        self.head_first = self.head_last = None
        self.head_kinds = dict(ground=0, self_contact=0, other_external=0)
        self.head_pairs = {}
        self.jac = np.zeros((3, self.model.nv))

    def __call__(self, row):
        self.count += 1
        supported = int(np.sum(np.asarray(row['foot_ground_z_N']) > 1.))
        self.ground_sum += supported
        self.zero_ground += supported == 0
        up = float(np.asarray(row['solver_state']['root_rotmat'])[2, 2])
        self.up_min = min(self.up_min, up)
        self.inverted += up < 0
        self.max_joint_speed = max(self.max_joint_speed, float(np.max(np.abs(np.asarray(row['qvel_after'])[self.vadr]))))
        self.saturation += np.abs(np.asarray(row['torque_before_limit']) - row['torque_applied']) > 1e-9
        active = [c for c in row['contacts'] if c['force_norm_gt_1N']]
        head = [c for c in active if any('head' in n.lower() for n in c['body_names'])]
        if head:
            self.head_count += 1
            self.head_peak = max(self.head_peak, max(c['force_norm_N'] for c in head))
            if self.head_first is None: self.head_first = row['time_before_s']
            self.head_last = row['time_before_s']
            kinds = set()
            for c in head:
                kind = 'self_contact' if c['kind'] == 'self' else 'ground' if 'world' in c['body_names'] else 'other_external'
                kinds.add(kind)
                key = kind + ':' + '|'.join(c['body_names'])
                pair = self.head_pairs.setdefault(key, dict(first_s=row['time_before_s'], last_s=0., peak_contact_norm_N=0.))
                pair['last_s'] = row['time_before_s']
                pair['peak_contact_norm_N'] = max(pair['peak_contact_norm_N'], c['force_norm_N'])
            for kind in kinds: self.head_kinds[kind] += 1
        ground = [c for c in active if c['kind'] == 'external' and 'world' in c['body_names']]
        if ground:
            # No mj_forward or step: independent qpos-only kinematics/Jacobians.
            self.data.qpos[:] = row['qpos_before']
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
        seen = set()
        for c in ground:
            name = next(n for n in c['body_names'] if n != 'world')
            b = self.perbody.setdefault(name, dict(first_s=row['time_before_s'], last_s=0., samples=0,
                peak_contact_norm_N=0., upward_impulse_Ns=0., max_ground_point_horizontal_speed_m_s=0.,
                nonfoot_nongripper=bool(c['nonfoot_nongripper'])))
            b['last_s'] = row['time_before_s']
            if name not in seen: b['samples'] += 1; seen.add(name)
            b['peak_contact_norm_N'] = max(b['peak_contact_norm_N'], c['force_norm_N'])
            sign = 1 if c['body_names'][1] == name else -1
            b['upward_impulse_Ns'] += sign * c['force_world_on_geom2_N'][2] * self.dt
            mujoco.mj_jac(self.model, self.data, self.jac, None, np.asarray(c['position_world_m']), self.model.body(name).id)
            speed = float(np.linalg.norm((self.jac @ np.asarray(row['qvel_before']))[:2]))
            b['max_ground_point_horizontal_speed_m_s'] = max(b['max_ground_point_horizontal_speed_m_s'], speed)

    def result(self):
        for b in self.perbody.values(): b['sampled_occupancy_s'] = b['samples'] * self.dt
        return dict(samples=self.count, sample_period_s=self.dt, includes_zero_action_warmup=True,
            head_contact_samples=self.head_count, head_sampled_occupancy_s=self.head_count*self.dt,
            head_first_s=self.head_first, head_last_s=self.head_last, head_peak_N=self.head_peak,
            head_contact_kind_samples=self.head_kinds, head_contact_pairs=self.head_pairs,
            minimum_root_up_dot=self.up_min, inverted_samples=self.inverted,
            ground_supported_feet_mean=self.ground_sum/self.count if self.count else None,
            zero_ground_supported_samples=self.zero_ground,
            saturated_joint_substeps=int(self.saturation.sum()), saturation_by_joint=dict(zip(self.joint_names,self.saturation.tolist())),
            maximum_joint_speed_rad_s=self.max_joint_speed, ground_contact_by_body=self.perbody,
            definition='Existing mj_step solve callback before policy endpoint forward; >1N per-contact force norm. Occupancy sums thresholded5ms samples, not exact continuous duration. Upward impulse integrates those active sampled ground forces. Slip is horizontal contact-point speed from independent pre-integration Jacobian times qvel. No extra forward/step on replay or diagnostic data. Head includes any head contact; per-body ground summary is external world only.')


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    policy = UmiPolicy(args.checkpoint)
    positions, rotations = policy.trajectories(args.trajectory).sample(args.cases, args.seed)
    if not 0 <= args.case < args.cases:
        raise ValueError('case must be inside the shared draw')
    p, r = positions[args.case:args.case+1], rotations[args.case:args.case+1]
    config = args.checkpoint.parent if args.checkpoint.is_file() else args.checkpoint
    sim = Go2Arx5Mujoco(config / 'config.json')
    metadata = {'checkpoint': str(args.checkpoint.resolve()), 'trajectory': str(args.trajectory.resolve()),
                'sample_cases': args.cases, 'sample_seed': args.seed, 'case': args.case,
                'physics_dt_s': sim.model.opt.timestep, 'decimation': sim.decimation,
                'delay_substeps': sim.delay.tolist(), 'joint_names': list(sim.joint_names),
                'joint_qpos_indices': sim.qadr.tolist(), 'joint_qvel_indices': sim.vadr.tolist(),
                'torque_limit': sim.torque_limit.tolist(), 'kp': sim.kp.tolist(), 'kd': sim.kd.tolist(),
                'integrator': mujoco.mjtIntegrator(sim.model.opt.integrator).name,
                'mujoco_version': mujoco.__version__, 'python': sys.executable,
                'contact_phase': 'Immediately after mj_step, before any extra mj_forward. Forces and Cartesian body state belong to its existing solve; qpos_after/qvel_after are integrated state, not refreshed Cartesian kinematics.',
                'contact_record_time': 'contact time_s is engine time_after_s; use time_before_s for this mj_step pre-integration solve configuration',
                'warmup': 'First four substeps are the existing zero-action reset warmup',
                'policy_endpoint': 'Existing mj_forward is still called once after four substeps',
                'target_clock': 'Target evaluated at solve configuration time_before_s; no observer/history mutation'}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    common = dict(checkpoint_dir=args.checkpoint, trajectory_path=args.trajectory,
                  seconds=args.seconds, seed=args.seed, policy=policy, positions=p, rotations=r)
    baseline = evaluate(**common, raw_output=args.output / 'baseline.npz')
    count = 0
    # Read the authoritative task from the observer, shared with evaluate.
    observer = policy.observer(p, r)
    import torch
    with (args.output / 'substeps.jsonl').open('x') as stream:
        def record(row):
            nonlocal count
            row['policy_step'] = count // sim.decimation
            target_p, target_r = observer.targets(torch.tensor([row['time_before_s']], dtype=torch.float32))
            row['target_pos'] = target_p[0].numpy()
            row['target_rotmat'] = target_r[0].numpy()
            row['ee_position_error_m'] = float(np.linalg.norm(row['solver_state']['ee_pos'] - row['target_pos']))
            stream.write(json.dumps(row, default=serial) + '\n')
            count += 1
        traced = evaluate(**common, raw_output=args.output / 'traced.npz', substep_trace=record)
    (args.output / 'result.json').write_text(json.dumps(traced, indent=2))
    exact = {}
    with np.load(args.output / 'baseline.npz') as a, np.load(args.output / 'traced.npz') as b:
        for key in a.files:
            exact[key] = bool(np.array_equal(a[key], b[key]))
    validation = {'endpoint_arrays_exact': exact, 'result_json_exact': baseline == traced,
                  'substeps': count, 'complete': traced['completion_status'] == 'complete'}
    (args.output / 'validation.json').write_text(json.dumps(validation, indent=2))
    if not all(exact.values()) or baseline != traced or not validation['complete']:
        raise RuntimeError(f'Trace verification failed: {validation}')
    print(json.dumps(validation))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, default=Path('reference/data/tossing.pkl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=16)
    parser.add_argument('--case', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--seconds', type=float, default=17.)
    run(parser.parse_args())
