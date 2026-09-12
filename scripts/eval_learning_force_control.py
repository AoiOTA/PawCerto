"""Consume an exported LFC student in the official Lab source task."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--robot-usd', type=Path, required=True)
    parser.add_argument('--exported-policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--preflight', action='store_true', help='CPU contract check only')
    args = parser.parse_args(argv)
    if args.steps < 1 or args.num_envs < 1:
        raise ValueError('steps and num-envs must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from pawcerto.artifacts import file_identity
    from pawcerto.methods.learning_force_control.export import load_training_checkpoint, JitPolicy
    from pawcerto.methods.learning_force_control.evaluation import fixed_task_config, evaluate_fixed_policy
    torch.set_num_threads(1)
    loaded = load_training_checkpoint(args.checkpoint)
    task = json.loads(args.task.read_text())
    config = fixed_task_config(loaded.config, task, num_envs=args.num_envs)
    usd = args.robot_usd.resolve(strict=True)
    if usd.suffix == '.txt':
        usd = Path(usd.read_text().strip()).resolve(strict=True)
    urdf = usd.parent.parent / 'merged.urdf'
    identity = {'robot_usd': file_identity(usd), 'merged_urdf': file_identity(urdf)}
    for key, value in identity.items():
        if value['sha256'] != loaded.config['runtime']['asset_identity'][key]['sha256']:
            raise ValueError('Evaluation robot asset differs from checkpoint: ' + key)
    contract = json.loads((args.exported_policy / 'contract.json').read_text())
    if (contract['format'] != 'pawcerto.learning_force_control.student.v1'
            or contract['checkpoint']['sha256'] != loaded.identity['sha256']):
        raise ValueError('Export does not match the evaluation checkpoint')
    exported = JitPolicy(args.exported_policy)
    metadata = {
        'method': 'learning_force_control', 'robot': 'b1_z1',
        'checkpoint': loaded.identity, 'checkpoint_next_iteration': loaded.next_iteration,
        'checkpoint_common_step_counter': loaded.common_step_counter,
        'checkpoint_global_steps': loaded.global_steps, **identity,
        'runtime_source': file_identity(ROOT / 'pawcerto/isaac/learning_force_control_runtime.py'),
        'task_source': file_identity(ROOT / 'pawcerto/methods/learning_force_control/training/isaac_env.py'),
        'config': config, 'fixed_task': task, 'seed': args.seed,
        'requested_policy_steps': args.steps,
        'command': list(sys.argv if argv is None else [sys.argv[0], *argv]),
        'exported_policy': {name: file_identity(args.exported_policy / name) for name in
                            ('body_latest.jit', 'adaptation_module_latest.jit')},
        'evaluation_overrides': 'fixed locomotion and EE sampler/limit ranges and requested environment count; retain source terrain, noise, domain randomization, automatic resets, binary mode resampling and per-episode spring-force schedules',
        'initialization': 'fresh task reset with source zero-action warmup; checkpoint counters retained, not a saved trajectory continuation',
        'snapshot_timing': 'after physics/reward/termination, before automatic reset',
        'force_body': 'link06', 'measurement_body': 'gripperMover',
        'force_labels': 'known applied spring/external force; contact forces recorded separately',
    }
    if args.preflight:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'preflight.json').write_text(json.dumps(dict(metadata,
            physics_run=False, evaluation_success=False), indent=2, allow_nan=False) + '\n')
        return
    launcher = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(visualizer='none' if args.headless else 'kit', device=args.device)
        import random
        import numpy as np
        from pawcerto.isaac.learning_force_control_runtime import B1Z1Isaac
        from pawcerto.methods.learning_force_control.training.config import resolve_config
        from pawcerto.methods.learning_force_control.training.isaac_env import LFCIsaacTrainingEnv
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cfg = resolve_config(config)
        runtime = B1Z1Isaac(num_envs=args.num_envs, device=args.device, usd_path=usd, config=cfg)
        env = LFCIsaacTrainingEnv(runtime, config)
        env.common_step_counter = loaded.common_step_counter
        # Runner counts samples in training rollouts; env includes reset warmups.
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        env.global_steps = saved['env_state']['scalars']['global_steps']
        loaded.model.to(args.device)
        exported.to(args.device)
        summary = evaluate_fixed_policy(env, loaded.model, steps=args.steps, output=args.output,
                                        metadata=metadata, exported_policy=exported)
        print(json.dumps(summary, allow_nan=False), flush=True)
    except BaseException as exc:
        if not args.output.exists():
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
            (args.output / 'summary.json').write_text(json.dumps({
                'status': 'failed_before_evaluation', 'error': f'{type(exc).__name__}: {exc}',
                'saved_policy_steps': 0, 'evaluation_success': False}, indent=2) + '\n')
        raise
    finally:
        if launcher is not None:
            launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__ == '__main__':
    main()
