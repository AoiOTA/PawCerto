"""Bounded full-size PPO throughput pair; run each mode in a fresh process.

Restores the complete checkpoint including optimizer, task caches and RNG.
Legacy capture is forced only in this probe. No training configuration changes.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--mode',choices=['legacy','boundary'],required=True)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--measured',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.output.exists(): parser.error('output exists; preserve prior evidence')
    if args.warmup<0 or args.measured<1: parser.error('invalid update budget')
    launcher=AppLauncher(args)
    try:
        import random
        import numpy as np
        import torch
        from pawcerto.artifacts import file_identity
        from pawcerto.isaac.roboduet_runtime import Go1Arx5Isaac,DEFAULT_USD
        from pawcerto.methods.roboduet.training import FORMAT,RoboDuetRunner
        from pawcerto.methods.roboduet.training.isaac_env import RoboDuetIsaacTrainingEnv
        # Match train_roboduet entrypoint's initial seed; load restores saved RNG.
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if payload['format']!=FORMAT: raise ValueError('full training checkpoint required')
        config=payload['config']
        initial_iteration=payload['next_iteration']
        if payload['stage']!=1: raise ValueError('assigned comparison is stage 1')
        if initial_iteration+args.warmup+args.measured>config['stage']['pretrained_to_hybrid_start']:
            raise ValueError('comparison would cross assigned stage boundary')
        if config['Cfg']['env']['num_envs']!=4096 or config['RunnerArgs']['num_steps_per_env']!=24:
            raise ValueError('assigned full-size 4096 by 24 configuration required')
        del payload
        seed=config.get('seed',0)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        usd=Path(config.get('pawcerto_asset',{}).get('usd_path',DEFAULT_USD))
        if usd.suffix=='.txt': usd=Path(usd.read_text().strip())
        asset=file_identity(usd)
        if config.get('pawcerto_asset',{}).get('usd_sha256') not in (None,asset['sha256']):
            raise ValueError('checkpoint asset changed')
        runtime=Go1Arx5Isaac(num_envs=4096,device=args.device,usd_path=usd)
        if args.mode=='legacy':
            original=runtime.step_control
            runtime.step_control=lambda leg,arm,**kwargs:original(leg,arm,capture_state=True)
        env=RoboDuetIsaacTrainingEnv(runtime,config)
        runner=RoboDuetRunner(env,config,args.device,init_at_random_ep_len=False)
        runner.load(args.checkpoint)
        args.output.mkdir(parents=True)
        (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        paths=[Path(__file__),ROOT/'pawcerto/isaac/roboduet_runtime.py',
            *sorted((ROOT/'pawcerto/methods/roboduet').rglob('*.py'))]
        sources={str(p.relative_to(ROOT)):file_identity(p)['sha256'] for p in paths}
        (args.output/'identity.json').write_text(json.dumps(dict(mode=args.mode,checkpoint=file_identity(args.checkpoint),
            asset=asset,sources=sources,warmup=args.warmup,measured=args.measured,initial_iteration=initial_iteration),indent=2)+'\n')
        rows=[]
        with (args.output/'metrics.jsonl').open('w') as out:
            for i in range(args.warmup+args.measured):
                torch.cuda.synchronize(runner.device)
                torch.cuda.reset_peak_memory_stats(runner.device)
                started=time.perf_counter()
                stats=runner.run_iteration()
                torch.cuda.synchronize(runner.device)
                stats.update(update_wall_seconds=time.perf_counter()-started,phase='warmup' if i<args.warmup else 'measured',
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(runner.device),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(runner.device))
                stats['transitions_per_second']=4096*24/stats['update_wall_seconds']
                line=json.dumps(stats,allow_nan=False)
                out.write(line+'\n');out.flush();print(line,flush=True)
                rows.append(stats)
        # Outside timing; retain actual final parameters, optimizer and environment.
        runner.save(args.output/'final.pt')
        measured=[r for r in rows if r['phase']=='measured']
        elapsed=sum(r['update_wall_seconds'] for r in measured)
        report=dict(mode=args.mode,complete_updates=len(rows),warmup_updates=args.warmup,measured_updates=len(measured),
            measured_seconds=elapsed,mean_update_seconds=elapsed/len(measured),
            measured_transitions=len(measured)*4096*24,transitions_per_second=len(measured)*4096*24/elapsed,
            total_experiment_transitions=len(rows)*4096*24,final_iteration=runner.next_iteration,
            scope='full 4096x24 stage1 PPO checkpoint resume; timing excludes checkpoint writes; existing long run remains concurrent')
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))

if __name__=='__main__': main()
