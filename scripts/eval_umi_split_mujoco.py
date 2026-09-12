"""Evaluate every ID in an explicitly selected partition, using the normal MuJoCo path."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pawcerto.methods.umi_on_legs import UmiPolicy
from pawcerto.methods.umi_on_legs.data_split import configure_selection, evaluation_identity
from pawcerto.mujoco.evaluate import evaluate, summarize_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, default=Path('reference/data/tossing.pkl'))
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--partition', choices=('validation', 'test'), default='validation')
    parser.add_argument('--substep-summary', action='store_true', help='Aggregate existing 5ms contact trace, saturation and ground-point speed without saving raw JSONL')
    parser.add_argument('--seconds', type=float, default=17.)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    policy = UmiPolicy(args.checkpoint)
    policy_dt = policy.config['env']['cfg']['sim']['dt'] * policy.config['env']['controller']['decimation_count']
    if not args.seconds >= 2 * policy_dt:
        parser.error('--seconds must cover the reset step plus at least one policy step')
    training = policy.training_selection
    selection = configure_selection(policy.config, args.trajectory, args.split_manifest, args.partition)
    ids = selection['trajectory_ids']
    positions, rotations = policy.trajectories(args.trajectory).select_ids(ids, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    identity = evaluation_identity(training, selection)
    (args.output/'data_identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    results = []
    for index, trajectory_id in enumerate(ids):
        summary = None
        if args.substep_summary:
            from scripts.trace_umi_contacts import SubstepContactSummary
            summary = SubstepContactSummary((args.checkpoint.parent if args.checkpoint.is_file() else args.checkpoint)/'config.json')
        result = evaluate(args.checkpoint, args.trajectory, args.seconds, args.seed, policy=policy,
                          positions=positions[index:index+1], rotations=rotations[index:index+1],
                          raw_output=args.output/f'trajectory_{trajectory_id:03d}.npz', substep_trace=summary)
        if summary is not None:
            (args.output/f'trajectory_{trajectory_id:03d}-substeps.json').write_text(json.dumps(summary.result(), indent=2, allow_nan=False)+'\n')
        result.update(case_index=index, trajectory_id=trajectory_id)
        (args.output/f'trajectory_{trajectory_id:03d}.json').write_text(json.dumps(result, indent=2)+'\n')
        results.append({key: value for key, value in result.items() if not key.endswith('records')})
    report = summarize_cases(results, args.checkpoint, args.trajectory, args.seconds, args.seed)
    report.update(data_identity=identity, evaluated_trajectory_ids=ids,
                  trajectory_coverage='Every selected ID exactly once, in ascending source order')
    report['evidence'] = ('Fixed-duration partition evaluation; numerical/termination status remains in case results. '
                          'Held-out qualification requires the recorded matching train partition; '
                          'legacy full-pool weights are not held out. Unknown recording provenance remains unverified.')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    if report.get('completion_status') in ('invalid', 'incomplete'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
