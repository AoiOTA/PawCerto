"""Paired fresh-process state-capture probe; no optimizer or learning claim.

Run legacy and boundary modes with identical checkpoint, seed, count and steps.
The checkpoint's first N physical states are restored; task/curriculum starts
fresh with the same RNG in each process. PhysX internal contact cache is not saved.
"""
import argparse
import hashlib
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
    parser.add_argument('--num-envs',type=int,default=32)
    parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--seed',type=int,default=932)
    parser.add_argument('--output',type=Path,required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.output.exists(): parser.error('output exists; preserve prior evidence')
    launcher=AppLauncher(args)
    try:
        import random
        import numpy as np
        import torch
        from pawcerto.isaac.roboduet_runtime import Go1Arx5Isaac,DEFAULT_USD
        from pawcerto.methods.roboduet.training.isaac_env import RoboDuetIsaacTrainingEnv
        from pawcerto.artifacts import file_identity
        torch.set_num_threads(1)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        config=checkpoint['config']
        original_n=config['Cfg']['env']['num_envs']
        if not 0 < args.num_envs <= original_n: raise ValueError('invalid bounded environment count')
        config['Cfg']['env']['num_envs']=args.num_envs
        runtime=Go1Arx5Isaac(num_envs=args.num_envs,device=args.device,
            usd_path=config.get('pawcerto_asset',{}).get('usd_path',DEFAULT_USD))
        env=RoboDuetIsaacTrainingEnv(runtime,config)
        env._reset_indices(torch.arange(args.num_envs,device=args.device))
        physical=checkpoint['env_training_state']['runtime']
        physical={k:(v[:args.num_envs] if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==original_n else v)
                  for k,v in physical.items()}
        runtime.load_training_state(physical)
        env.set_stage(physical['stage'])
        env._refresh_physics()
        step_control=runtime.step_control
        if args.mode=='legacy':
            runtime.step_control=lambda leg,arm,**kwargs:step_control(leg,arm,capture_state=True)
        generator=torch.Generator(device=args.device).manual_seed(args.seed+1)
        dog=torch.randn(args.steps,args.num_envs,12,device=args.device,generator=generator)*.15
        arm=torch.randn(args.steps,args.num_envs,6,device=args.device,generator=generator)*.15
        records=[]
        def record(e,d,t):
            records.append({**{k:getattr(e.task,k).clone() for k in ('root_states','end_effector_state',
                'dof_pos','dof_vel','contact_forces','base_lin_vel','base_ang_vel','projected_gravity')},
                'dones':d.clone(),'timeouts':t.clone()})
        torch.cuda.synchronize()
        started=time.perf_counter()
        for i in range(args.steps):
            dr,ar,dones,info=env.step(dog[i],arm[i],pre_reset_callback=record)
            records[-1].update(dog_reward=dr.clone(),arm_reward=ar.clone())
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        args.output.mkdir(parents=True)
        torch.save({k:torch.stack([r[k] for r in records]).cpu() for k in records[0]},args.output/'trace.pt')
        report=dict(mode=args.mode,seed=args.seed,num_envs=args.num_envs,steps=args.steps,
            seconds=elapsed,transitions_per_second=args.steps*args.num_envs/elapsed,
            checkpoint=file_identity(args.checkpoint),
            source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [ROOT/'pawcerto/isaac/roboduet_runtime.py',ROOT/'pawcerto/methods/roboduet/training/isaac_env.py',Path(__file__)]},
            scope='fresh-process same checkpoint physical subset and fixed actions; includes identical trace capture overhead; no PPO throughput or learning claim')
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))

if __name__=='__main__': main()
