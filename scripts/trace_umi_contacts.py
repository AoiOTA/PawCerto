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
