#!/usr/bin/env python3
"""Check actor identity and run original/exported policies in separate CPU processes.

Consumes an existing export. Retains subprocess failures and never overwrites
an evidence directory. Parity is package validation, not behavior acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_saved(original, exported):
    import numpy as np
    with np.load(original.with_suffix('.npz'), allow_pickle=False) as a, np.load(exported.with_suffix('.npz'), allow_pickle=False) as b:
        if a.files != b.files:
            raise ValueError('Rollout array names differ')
        for key in a.files:
            if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype or not np.array_equal(a[key], b[key]):
                raise ValueError(f'Rollout arrays differ: {key}')
            if a[key].dtype.kind in 'fiu' and not np.isfinite(a[key]).all():
                raise ValueError(f'Non-finite rollout: {key}')
        names = a.files
    results = [json.loads(p.read_text()) for p in (original, exported)]
    for result in results:
        if result['completion_status'] != 'complete' or result['failure'] is not None:
            raise ValueError('Incomplete rollout cannot pass package consumption')
    if [{k: v for k, v in r.items() if k != 'checkpoint'} for r in results][0] != [{k: v for k, v in r.items() if k != 'checkpoint'} for r in results][1]:
        raise ValueError('Rollout JSON differs beyond checkpoint path')
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=17.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--worker', choices=('original', 'exported'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.worker:
        from pawcerto.mujoco.evaluate import evaluate
        source = args.checkpoint if args.worker == 'original' else args.export
        result = evaluate(source, args.trajectory, args.seconds, args.seed,
                          raw_output=args.output / f'{args.worker}.npz')
        dump(args.output / f'{args.worker}.json', result)
        return 0 if result['completion_status'] == 'complete' else 2
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'processes': {}, 'scope': 'One sampled trajectory; actor and independent execution parity only, not fixed16 or holdout acceptance.'}
    try:
        from pawcerto.methods.umi_on_legs import UmiPolicy
        from pawcerto.methods.umi_on_legs.robot_binding import joint_order
        original = UmiPolicy(args.checkpoint, device='cpu')
        exported = UmiPolicy(args.export, device='cpu')
        if original.config != exported.config:
            raise ValueError('Export configuration differs')
        if json.loads((args.export / 'joint_names.json').read_text()) != joint_order(original.config):
            raise ValueError('Export joint order differs from runtime')
        if not isinstance(exported.actor, torch.jit.ScriptModule):
            raise ValueError('Export does not load TorchScript directly')
        a, b = original.actor.state_dict(), exported.actor.state_dict()
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) or not torch.isfinite(a[k]).all() for k in a):
            raise ValueError('Actor state mismatch or non-finite tensor')
        dimension = original.config['runner']['alg']['actor_critic']['num_actor_obs']
        example = torch.linspace(-1, 1, 4 * dimension).reshape(4, dimension)
        with torch.inference_mode():
            left, right = original.actor(example), exported.actor(example)
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)
        report['actor_max_abs_error'] = float((left - right).abs().max())
        files = [args.export / name for name in ('actor.ts', 'config.json', 'joint_names.json')]
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        for kind in ('original', 'exported'):
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', kind,
                       '--checkpoint', str(args.checkpoint.resolve()), '--export', str(args.export.resolve()),
                       '--trajectory', str(args.trajectory.resolve()), '--output', str(args.output.resolve()),
                       '--seconds', str(args.seconds), '--seed', str(args.seed)]
            with (args.output / f'{kind}.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            report['processes'][kind] = {'command': command, 'exit_code': result.returncode}
            dump(args.output / 'verification.json', report)
            if result.returncode:
                raise RuntimeError(f'{kind} execution failed with exit {result.returncode}')
        report['exact_equal_arrays'] = verify_saved(args.output / 'original.json', args.output / 'exported.json')
        if before != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}:
            raise ValueError('Export changed during validation')
        report.update(status='package_parity_passed', bundle_sha256=before)
    except Exception:
        report['exception'] = traceback.format_exc()
        raise
    finally:
        dump(args.output / 'verification.json', report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
