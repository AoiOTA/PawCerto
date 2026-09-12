"""Compare fixed RoboDuet evaluations on each case's shared finite prefix."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRICS = ('leg_linear_error_m_s', 'leg_yaw_error_rad_s',
           'ee_position_error_m', 'ee_orientation_error_rad')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', type=Path, required=True)
    parser.add_argument('--iterations', nargs='+', type=int, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2027, 2028, 2029])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.iterations) < 2 or len(set(args.iterations)) != len(args.iterations):
        raise ValueError('Select at least two distinct checkpoint iterations')
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import torch
    from pawcerto.artifacts import file_identity
    torch.set_num_threads(1)
    aggregate = {iteration: [] for iteration in args.iterations}
    cases, files, protocols = [], [], set()
    for seed in args.seeds:
        data = {}
        for iteration in args.iterations:
            folder = args.evaluations / f'model_{iteration}'
            protocols.add(file_identity(folder / 'protocol.json')['sha256'])
            path = folder / f'seed_{seed}' / 'states.pt'
            files.append(file_identity(path))
            data[iteration] = torch.load(path, map_location='cpu', weights_only=False)
        if len(protocols) != 1:
            raise ValueError('Evaluation protocols differ')
        policy_dts = {float(value['metadata']['control_dt']) for value in data.values()}
        if len(policy_dts) != 1 or not all(np.isfinite(dt) and dt > 0 for dt in policy_dts):
            raise ValueError('Recorded policy periods are invalid or differ')
        policy_dt = next(iter(policy_dts))
        first = data[args.iterations[0]]
        if any(list(value['cases']) != list(first['cases']) for value in data.values()):
            raise ValueError('Evaluation case names/order differ')
        for index, case in enumerate(first['cases']):
            rows = {iteration: data[iteration]['cases'][case] for iteration in args.iterations}
            count = min(map(len, rows.values()))
            initial = first['initialization'][index]
            initial_delta = {iteration: {
                key: float((data[iteration]['initialization'][index][key]-initial[key]).abs().max())
                for key in ('root_states', 'joint_pos')} for iteration in args.iterations}
            means = {}
            for iteration in args.iterations:
                values = np.array([[float(row[key]) for key in METRICS] for row in rows[iteration][:count]])
                if count:
                    if not np.isfinite(values).all():
                        raise ValueError(f'Nonfinite shared prefix: {iteration}, {seed}, {case}')
                    means[iteration] = dict(zip(METRICS, values.mean(0).tolist()))
                    aggregate[iteration].append(values.mean(0))
                else:
                    means[iteration] = None
            cases.append({'seed': seed, 'case': case, 'shared_policy_steps': count,
                'shared_duration_s': count*policy_dt,
                'observed_policy_duration_s': {i: len(rows[i])*policy_dt for i in args.iterations},
                'initialization_max_abs_delta_from_first': initial_delta,
                'shared_prefix_mean_errors': means})
    result = {
        'scope': 'per-case shortest observed policy prefix, including its terminal step; initialization excluded; equal case weights, not a complete-duration behavior comparison',
        'iterations': args.iterations, 'metric_order': METRICS,
        'protocol_sha256': next(iter(protocols)), 'inputs': files,
        'included_cases': len(aggregate[args.iterations[0]]),
        'empty_shared_prefix_cases': sum(case['shared_policy_steps'] == 0 for case in cases),
        'cases': cases,
        'equal_case_mean_errors': {i: dict(zip(METRICS, np.stack(values).mean(0).tolist()))
                                  if values else None for i, values in aggregate.items()},
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['equal_case_mean_errors'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
