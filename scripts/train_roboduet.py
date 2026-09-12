"""Train the original RoboDuet automatic dual PPO in official Isaac Lab."""
import argparse
import json
from pathlib import Path
import sys

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
        from pawcerto.isaac.roboduet_runtime import Go1Arx5Isaac
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
        if args.usd: kwargs['usd_path'] = args.usd
        runtime = Go1Arx5Isaac(**kwargs)
        env = RoboDuetIsaacTrainingEnv(runtime,config)
        runner = RoboDuetRunner(env,config,args.device)
        if args.resume: runner.load(args.resume)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        runner.save(args.output/f'model_{runner.next_iteration}.pt')
        with (args.output/'metrics.jsonl').open('a') as output:
            for _ in range(args.iterations):
                stats = runner.run_iteration()
                line = json.dumps(stats,allow_nan=False)
                output.write(line+'\n')
                output.flush()
                print(line,flush=True)
                if runner.next_iteration % args.save_every == 0:
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
