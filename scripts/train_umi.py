"""Train original UMI PPO on the original robot in Isaac Lab PhysX."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'reference/checkpoints/tossing/ours/config.json')
    parser.add_argument('--trajectory',type=Path,default=ROOT/'reference/data/tossing.pkl')
    parser.add_argument('--split-manifest', type=Path, help='Grouped manifest; training always uses train partition')
    parser.add_argument('--usd-path', type=Path, help='Explicit robot USD or USD-path text file; otherwise use the recorded asset, then the workspace default')
    parser.add_argument('--urdf-path', type=Path, help='Matching merged URDF for joint limits and body topology; defaults to the recorded path, then the original robot URDF')
    parser.add_argument('--joint-names',type=Path,help='Optional exact joint-order JSON; defaults to the recorded robot binding')
    parser.add_argument('--robot', choices=('go2_arx5','as2_piper'), help='Adapt a fresh configuration; saved AS2 configurations retain their binding')
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--iterations',type=int,default=None,help='Default: original config max_iterations')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--weights',type=Path,help='Initialize original actor+critic weights without optimizer/iteration')
    parser.add_argument('--actor-weights',type=Path,help='Initialize an AS2 actor and action std across assembly changes; keep a fresh critic and optimizer')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--force-signal',choices=('reconstructed-solver','normal-contact'),
                        default='reconstructed-solver',help='EMD input; normal-contact explicitly selects the previous proxy')
    parser.add_argument('--save-every',type=int,default=None,help='Checkpoint interval; default: original config interval')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/umi_training')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if sum(p is not None for p in (args.resume, args.weights, args.actor_weights)) > 1:
        parser.error('--resume, --weights and --actor-weights are mutually exclusive')
    if args.save_every is not None and args.save_every < 1:
        parser.error('--save-every must be positive')
    launcher=AppLauncher(args)
    try:
        import random
        import numpy as np
        import torch
        from pawcerto.isaac.runtime import DEFAULT_USD, DEFAULT_URDF, Go2Arx5Isaac
        from pawcerto.methods.umi_on_legs.training import UmiTrainer, load_config
        from pawcerto.methods.umi_on_legs.training.isaac_env import UmiIsaacTrainingEnv
        from pawcerto.methods.umi_on_legs.training.semantics import runtime_contract,require_resume_contract
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        from pawcerto.methods.umi_on_legs.robot_binding import as2_config,robot_binding,joint_order,require_same_robot,require_same_as2_actor
        config=load_config(args.config)
        if args.robot == 'as2_piper':
            config=as2_config(config)
        binding=robot_binding(config)
        if args.robot is not None and args.robot != binding['name']:
            raise ValueError('--robot disagrees with the saved configuration')
        source_path=args.resume or args.weights or args.actor_weights
        source_checkpoint=torch.load(source_path,map_location='cpu',weights_only=False) if source_path else None
        if source_checkpoint is not None:
            check = require_same_as2_actor if args.actor_weights else require_same_robot
            check(config,source_checkpoint.get('config',{}))
        source_asset=(source_checkpoint.get('config',{}).get('pawcerto_asset') if source_checkpoint is not None else config.get('pawcerto_asset'))
        from pawcerto.methods.umi_on_legs.pretraining import resolve_training_asset
        destination_asset = config.get('pawcerto_asset') if args.actor_weights else source_asset
        config['pawcerto_asset'] = resolve_training_asset(config, destination_asset, args.usd_path, args.urdf_path)
        usd_path = Path(config['pawcerto_asset']['usd_path'])
        urdf_path = Path(config['pawcerto_asset']['urdf_path'])
        if config['env']['tasks']['reaching']['sequence_sampler'].get('trajectory_selection') and not args.split_manifest:
            raise ValueError('Saved split config requires --split-manifest; refusing silent full-pool training')
        from pawcerto.methods.umi_on_legs.data_split import configure_selection
        selection=configure_selection(config,args.trajectory,args.split_manifest,'train')
        config['seed']=args.seed
        config['env']['cfg']['env']['num_envs']=args.num_envs
        config['pawcerto_runtime']=runtime_contract(
            args.force_signal, config.get('joint_velocity_limit_override_rad_s'))
        source_runtime=None
        if source_path:
            source_selection=source_checkpoint.get('config',{}).get('env',{}).get('tasks',{}).get('reaching',{}).get('sequence_sampler',{}).get('trajectory_selection')
            if selection != source_selection:
                raise ValueError('Resume/weight initialization must preserve the training partition; start from scratch for a new split')
            source_runtime=source_checkpoint.get('config',{}).get('pawcerto_runtime')
            if args.resume:
                require_resume_contract(source_runtime,config['pawcerto_runtime'])
                from pawcerto.methods.umi_on_legs.pretraining import require_resume_family
                require_resume_family(source_checkpoint.get('config', {}), config)
            del source_checkpoint
        config['training_initialization']=dict(
            mode='resume' if args.resume else 'weights' if args.weights else 'actor_weights' if args.actor_weights else 'random',
            checkpoint=str(source_path.resolve()) if source_path else None,
            source_pawcerto_runtime=source_runtime,
            source_pawcerto_asset=source_asset if source_path else None)
        runtime=Go2Arx5Isaac(config,joint_order(config,args.joint_names),args.num_envs,args.device,
                           usd_path=usd_path,urdf_path=urdf_path,training=True,force_signal=args.force_signal)
        env=UmiIsaacTrainingEnv(runtime,config,args.trajectory,args.seed)
        trainer=UmiTrainer(env,config,args.device)
        if args.resume:
            trainer.load(args.resume)
        elif args.weights:
            trainer.load(args.weights,load_optimizer=False)
        elif args.actor_weights:
            trainer.load_actor(args.actor_weights)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'config.json').write_text(json.dumps(config,indent=2))
        iterations=config['runner']['max_iterations'] if args.iterations is None else args.iterations
        save_every=config['runner']['ckpt_save_interval'] if args.save_every is None else args.save_every
        trainer.save(args.output/f'model_{trainer.iteration}.pt')
        with (args.output/'metrics.jsonl').open('a') as metrics:
            for _ in range(iterations):
                stats=trainer.train_iteration()
                line=json.dumps(stats)
                metrics.write(line+'\n');metrics.flush()
                print(line,flush=True)
                if trainer.iteration%save_every==0:
                    trainer.save(args.output/f'model_{trainer.iteration}.pt')
        trainer.save(args.output/f'model_{trainer.iteration}.pt')
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__=='__main__':
    main()
