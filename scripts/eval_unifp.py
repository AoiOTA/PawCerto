"""Evaluate a specified UniFP student checkpoint on a fixed task in Isaac Lab."""
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
    parser.add_argument('--config', type=Path)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--robot-usd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exported-policy', type=Path, help='Export directory; its actions actually drive the simulator')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--preflight', action='store_true', help='CPU input/config check only')
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise ValueError('steps must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    from pawcerto.artifacts import file_identity
    from pawcerto.methods.unifp.export import load_checkpoint, ExportedUniFP
    from pawcerto.methods.unifp.evaluation import fixed_task_config, evaluate_fixed_policy
    import torch
    torch.set_num_threads(1)
    supplied = json.loads(args.config.read_text()) if args.config else None
    loaded = load_checkpoint(args.checkpoint, config=supplied)
    if loaded.global_steps is None:
        raise ValueError('Physical evaluation requires the checkpoint global_steps to preserve force-stage timing')
    task = json.loads(args.task.read_text())
    config = fixed_task_config(loaded.config, task, num_envs=args.num_envs)
    usd = args.robot_usd.resolve(strict=True)
    if usd.suffix == '.txt':
        usd = Path(usd.read_text().strip()).resolve(strict=True)
    urdf = next((folder / 'merged.urdf' for folder in (usd.parent, usd.parent.parent)
                 if (folder / 'merged.urdf').is_file()), None)
    if urdf is None:
        raise FileNotFoundError('Converted UniFP USD requires its adjacent merged.urdf')
    asset_identity = {'robot_usd': file_identity(usd), 'merged_urdf': file_identity(urdf)}
    saved_identity = loaded.config.get('runtime', {}).get('asset_identity', {})
    for name, identity in saved_identity.items():
        if name in asset_identity and asset_identity[name]['sha256'] != identity['sha256']:
            raise ValueError(f'Evaluation {name} differs from checkpoint asset')
    exported = ExportedUniFP(args.exported_policy) if args.exported_policy else None
    if exported and exported.contract['checkpoint']['sha256'] != loaded.identity['sha256']:
        raise ValueError('Export checkpoint differs from evaluation checkpoint')
    metadata = {
        'method': 'unifp', 'robot': 'b2_z1', 'checkpoint': loaded.identity,
        'checkpoint_iteration': loaded.iteration, 'checkpoint_global_steps': loaded.global_steps,
        **asset_identity,
        'runtime_source': file_identity(ROOT / 'pawcerto/isaac/unifp_runtime.py'),
        'task_source': file_identity(ROOT / 'pawcerto/methods/unifp/training/isaac_env.py'),
        'config': config, 'fixed_task': task, 'seed': args.seed,
        'requested_steps': args.steps, 'command': list(sys.argv if argv is None else [sys.argv[0], *argv]),
        'evaluation_overrides': 'fixed command and goal sampler ranges, zero-command probability 0, requested environment count; original terrain, noise, randomization, resets, force schedules and deadbands retained',
        'exported_policy': file_identity(args.exported_policy / 'policy.pt') if exported else None,
        'snapshot_timing': 'after physics/reward/termination, before source automatic reset',
        'force_labels': 'known applied external force; contact forces recorded separately',
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
        from pawcerto.isaac.unifp_runtime import B2Z1Isaac
        from pawcerto.methods.unifp.terrain import UniFPTerrain
        from pawcerto.methods.unifp.training.config import resolve_config
        from pawcerto.methods.unifp.training.isaac_env import UniFPIsaacTrainingEnv
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cfg = resolve_config(config)
        terrain = UniFPTerrain(cfg.terrain)
        runtime = B2Z1Isaac(args.num_envs, args.device, usd, cfg, terrain)
        env = UniFPIsaacTrainingEnv(runtime, config)
        env.global_steps = loaded.global_steps
        loaded.model.to(args.device)
        if exported:
            exported = ExportedUniFP(args.exported_policy, device=args.device)
        metadata['global_steps_before_reset'] = env.global_steps
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
        import traceback
        traceback.print_exc()
        raise
    finally:
        if launcher is not None:
            launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__ == '__main__':
    main()
