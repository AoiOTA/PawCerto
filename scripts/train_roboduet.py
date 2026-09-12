"""Train the original RoboDuet automatic dual PPO in official Isaac Lab."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,help='Complete config JSON; default is the fixed upstream automatic configuration')
    parser.add_argument('--resume',type=Path,help='Full PawCerto RoboDuet iteration-boundary checkpoint')
    parser.add_argument('--num-envs',type=int,help='Override environment count for a new run')
    parser.add_argument('--iterations',type=int,required=True,help='Number of additional complete updates; explicit budget required')
    parser.add_argument('--stage-switch-iteration',type=int,help='Test override of zero-based switch boundary; original default 10000')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--save-every',type=int,default=1)
    parser.add_argument('--usd',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.iterations < 1 or args.save_every < 1:
        parser.error('iterations and save-every must be positive')
    if args.resume and (args.config or args.num_envs is not None or args.stage_switch_iteration is not None):
        parser.error('resume restores its saved configuration; do not override it')
    launcher = AppLauncher(args)
    try:
        import random
        import numpy as np
        import torch
        from pawcerto.methods.roboduet import default_config
        from pawcerto.isaac.roboduet_runtime import Go1Arx5Isaac, DEFAULT_USD
        from pawcerto.artifacts import file_identity
        from pawcerto.methods.roboduet.training import FORMAT, RoboDuetRunner
        from pawcerto.methods.roboduet.training.isaac_env import RoboDuetIsaacTrainingEnv
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.resume:
            source = torch.load(args.resume,map_location='cpu',weights_only=False)
            if source['format'] != FORMAT: raise ValueError('Expected a full RoboDuet training checkpoint')
            config = source['config']
            del source
        else:
            config = json.loads(args.config.read_text()) if args.config else default_config()
            config['seed'] = args.seed
            if args.num_envs is not None: config['Cfg']['env']['num_envs'] = args.num_envs
            if args.stage_switch_iteration is not None:
                if args.stage_switch_iteration < 0: raise ValueError('Stage switch iteration must be nonnegative')
                config['stage'].update(pretrained_to_hybrid_start=args.stage_switch_iteration,
                                       pretrained_to_hybrid_end=args.stage_switch_iteration)
        kwargs = dict(num_envs=config['Cfg']['env']['num_envs'],device=args.device)
        saved_asset = config.get('pawcerto_asset', {})
        usd = args.usd or saved_asset.get('usd_path') or DEFAULT_USD
        usd = Path(usd)
        if usd.suffix == '.txt': usd = Path(usd.read_text().strip())
        identity = file_identity(usd)
        urdf_identity = file_identity(usd.parent.parent / 'merged.urdf')
        if args.resume and not args.usd and saved_asset.get('usd_sha256') not in (None,identity['sha256']):
            raise ValueError('Saved RoboDuet USD content changed')
        kwargs['usd_path'] = usd
        runtime = Go1Arx5Isaac(**kwargs)
        env = RoboDuetIsaacTrainingEnv(runtime,config)
        runner = RoboDuetRunner(env,config,args.device,init_at_random_ep_len=not bool(args.resume))
        if args.resume: runner.load(args.resume)
        config['pawcerto_asset'] = dict(usd_path=identity['path'],usd_sha256=identity['sha256'],
                                        urdf_path=urdf_identity['path'],urdf_sha256=urdf_identity['sha256'],
                                        selection='explicit' if args.usd else 'checkpoint' if saved_asset else 'default')
        runner.config['pawcerto_asset'] = config['pawcerto_asset'].copy()
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        initial_checkpoint = args.output/f'model_{runner.next_iteration}.pt'
        if not (args.resume and initial_checkpoint.exists()):
            runner.save(initial_checkpoint)
        with (args.output/'metrics.jsonl').open('a') as output:
            for _ in range(args.iterations):
                if runner.device.type == 'cuda':
                    torch.cuda.synchronize(runner.device)
                    torch.cuda.reset_peak_memory_stats(runner.device)
                started = time.perf_counter()
                stats = runner.run_iteration()
                if runner.device.type == 'cuda': torch.cuda.synchronize(runner.device)
                stats['update_wall_seconds'] = time.perf_counter() - started
                stats['transitions_per_second'] = env.num_envs * runner.num_steps_per_env / stats['update_wall_seconds']
                if runner.device.type == 'cuda':
                    stats['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(runner.device)
                    stats['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(runner.device)
                line = json.dumps(stats,allow_nan=False)
                output.write(line+'\n')
                output.flush()
                print(line,flush=True)
                if runner.next_iteration % args.save_every == 0 or stats['rollout_stage'] != stats['stage']:
                    runner.save(args.output/f'model_{runner.next_iteration}.pt')
        runner.save(args.output/f'model_{runner.next_iteration}.pt')
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__ == '__main__':
    main()
