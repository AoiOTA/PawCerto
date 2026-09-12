"""Train DeepWBC through the official Isaac Lab runtime with an explicit budget."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser():
    from pawcerto.isaac.deepwbc_runtime import FORCE_MODES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='Complete DeepWBC JSON config')
    p.add_argument('--schedule', choices=('public_fresh', 'paper'), help='Fresh recipe; paper changes budget/regularization only')
    p.add_argument('--robot-usd', type=Path, help='Converted original Go1/WidowX USD')
    p.add_argument('--force-mode', choices=FORCE_MODES, help='Explicit approximate six-axis signal; required for fresh training')
    p.add_argument('--iterations', type=int, required=True, help='Additional complete learning iterations')
    p.add_argument('--resume', type=Path, help='Trusted full runner checkpoint; physical episodes restart')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    return p


def resolve_config(args):
    from pawcerto.methods.deepwbc.config import default_config
    if args.resume:
        if args.config or args.schedule or args.num_envs is not None:
            raise ValueError('Resume uses checkpoint config; config/schedule/num-envs overrides are not allowed')
        import torch
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if 'config' not in saved:
            raise ValueError('Resume requires a PawCerto checkpoint carrying the resolved configuration')
        config = saved['config']
    else:
        if args.config and args.schedule:
            raise ValueError('Select a complete config or a named schedule')
        config = json.loads(args.config.read_text()) if args.config else default_config(args.schedule or 'public_fresh')
        if args.num_envs is not None:
            config['env']['num_envs'] = args.num_envs
        config['seed'] = args.seed
        config['pawcerto_schedule'] = 'custom' if args.config else args.schedule or 'public_fresh'
    if config['env']['num_envs'] < 1:
        raise ValueError('num-envs must be positive')
    return config


def run_iterations(runner, iterations, output_dir):
    """Stream metrics while retaining periodic, initial and final checkpoints."""
    output_dir = Path(output_dir)
    runner.save(output_dir/f'model_{runner.current_learning_iteration}.pt')
    with (output_dir/'metrics.jsonl').open('a') as output:
        for index in range(iterations):
            for result in runner.learn(1, save_final=index == iterations-1):
                line = json.dumps(result, allow_nan=False)
                output.write(line+'\n'); output.flush()
                print(line, flush=True)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.iterations < 1:
        p.error('iterations must be positive')
    config = resolve_config(args)
    saved_runtime = config.get('pawcerto_runtime', {})
    from pawcerto.isaac.deepwbc_runtime import FORCE_MODES, FORCE_SIGNAL_EVIDENCE
    force_mode = args.force_mode or saved_runtime.get('force_mode')
    if force_mode not in FORCE_MODES:
        p.error('--force-mode is required for a fresh run')
    if args.resume and force_mode != saved_runtime.get('force_mode'):
        raise ValueError('Resume force mode differs from checkpoint')
    usd = args.robot_usd or saved_runtime.get('robot_usd')
    if usd is None:
        p.error('--robot-usd is required for a fresh run')
    usd = Path(usd).resolve()
    if not usd.is_file():
        p.error(f'Robot USD does not exist: {usd}')
    if usd.suffix == '.txt':
        usd = Path(usd.read_text().strip()).resolve()
        if not usd.is_file():
            p.error(f'USD pointer target does not exist: {usd}')
    import hashlib
    identity = hashlib.sha256(usd.read_bytes()).hexdigest()
    if args.resume and saved_runtime.get('robot_usd_sha256') != identity:
        raise ValueError('Resume robot USD differs from checkpoint')
    config['pawcerto_runtime'] = dict(robot_usd=str(usd), robot_usd_sha256=identity,
        force_mode=force_mode, force_signal_parity=FORCE_SIGNAL_EVIDENCE[force_mode],
        resume_semantics='optimizer and iteration continuation with fresh physical episodes')
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(visualizer='none' if args.headless else 'kit', device=args.device)
    try:
        import random
        import numpy as np
        import torch
        from pawcerto.isaac.deepwbc_runtime import Go1WidowXIsaac
        from pawcerto.methods.deepwbc.terrain import build_terrain, make_terrain_cfg
        from pawcerto.methods.deepwbc.training.isaac_env import DeepWBCIsaacTrainingEnv
        from pawcerto.methods.deepwbc.training.runner import OnPolicyRunner
        seed = config.get('seed', args.seed)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        terrain = build_terrain(seed=seed, cfg=config['terrain'])
        terrain_cfg, terrain_contact_paths = make_terrain_cfg(terrain)
        runtime = Go1WidowXIsaac(num_envs=config['env']['num_envs'], device=args.device,
            usd_path=usd, config=config, force_signal=force_mode,
            terrain_cfg=terrain_cfg, terrain_contact_paths=terrain_contact_paths,
            env_origins=torch.zeros(config['env']['num_envs'],3,device=args.device))
        env = DeepWBCIsaacTrainingEnv(runtime, config)
        runner = OnPolicyRunner(env, config, log_dir=args.output, device=args.device)
        if args.resume:
            runner.load(args.resume)
            env.task.update_counter = runner.current_learning_iteration
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        run_iterations(runner, args.iterations, args.output)
        (args.output/'evidence.json').write_text(json.dumps({
            'iterations_completed': runner.current_learning_iteration,
            'force_signal_parity': config['pawcerto_runtime']['force_signal_parity'],
            'learning_success': 'not inferred from finite updates',
            'resume_semantics': config['pawcerto_runtime']['resume_semantics']}, indent=2)+'\n')
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__ == '__main__':
    main()
