"""Train original UniFP through official Isaac Lab with an explicit update budget."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path)
    p.add_argument('--robot-usd',type=Path)
    p.add_argument('--iterations',type=int,required=True)
    p.add_argument('--num-envs',type=int)
    p.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--headless',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--force-boundary-probe',action='store_true',help='Separate stage execution probe after training, not trained force-stage evidence')
    args = p.parse_args()
    if args.iterations < 0 or (args.iterations == 0 and not (args.resume and args.force_boundary_probe)):
        p.error('--iterations must be positive, or zero for an explicit resumed force-boundary-only probe')
    from pawcerto.methods.unifp.training.config import default_config,resolve_config
    import torch
    if args.resume:
        if args.config or args.num_envs is not None:
            p.error('Resume uses checkpoint config; config/num-envs overrides are not allowed')
        saved = torch.load(args.resume,map_location='cpu',weights_only=False)
        config = saved['config']
    else:
        config = json.loads(args.config.read_text()) if args.config else default_config()
        if args.num_envs is not None:
            config['env']['env']['num_envs'] = args.num_envs
        config['seed'] = args.seed
    from pawcerto.artifacts import file_identity
    previous = config.get('runtime',{})
    usd = args.robot_usd or Path(previous.get('robot_usd',ROOT/'reference/isaac/b2_z1/usd_path.txt'))
    usd = usd.resolve(strict=True)
    if usd.suffix == '.txt':
        usd = Path(usd.read_text().strip()).resolve(strict=True)
    urdf = usd.parent.parent/'merged.urdf'
    identity = {'robot_usd':file_identity(usd),'merged_urdf':file_identity(urdf)}
    if args.resume and any(identity[key]['sha256'] != previous.get('asset_identity',{}).get(key,{}).get('sha256') for key in identity):
        raise ValueError('Resume asset differs from checkpoint identity')
    args.robot_usd = usd
    args.output.mkdir(parents=True,exist_ok=True)
    config['runtime'] = {'robot_usd':str(usd),'asset_identity':identity,'backend':'official Isaac Lab/PhysX',
        'force_signal':'known WORLD externally applied force at body COM; contact is separate',
        'resume_semantics':'optimizer/iteration/global_steps continuation; physical episodes restart'}
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(visualizer='none' if args.headless else 'kit',device=args.device)
    try:
        import random
        import numpy as np
        from pawcerto.methods.unifp.terrain import UniFPTerrain
        from pawcerto.isaac.unifp_runtime import B2Z1Isaac
        from pawcerto.methods.unifp.training.isaac_env import UniFPIsaacTrainingEnv
        from pawcerto.methods.unifp.training.runner import OnPolicyRunner
        seed=config.get('seed',0)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        cfg = resolve_config(config)
        terrain = UniFPTerrain(cfg.terrain)
        print('UNIFP_TERRAIN '+json.dumps({'shape':list(terrain.heightsamples.shape),'vertices':len(terrain.vertices),
            'triangles':len(terrain.triangles),'height_min_m':float(terrain.heightsamples.min()*cfg.terrain.vertical_scale),
            'height_max_m':float(terrain.heightsamples.max()*cfg.terrain.vertical_scale),'delatin_max_error':cfg.terrain.max_error}),flush=True)
        runtime = B2Z1Isaac(cfg.env.num_envs,args.device,args.robot_usd,cfg,terrain)
        env = UniFPIsaacTrainingEnv(runtime,config)
        runner = OnPolicyRunner(env,config['train'],log_dir=str(args.output),device=args.device)
        if args.resume:
            runner.load(args.resume)
        if args.iterations:
            runner.save(args.output/f'model_{runner.current_learning_iteration}.pt')
            runner.learn(args.iterations)
        evidence = {'iteration':runner.current_learning_iteration,'global_steps':env.global_steps,
            'optimizer_updates_this_process':args.iterations,
            'observations':{k:list(v.shape) for k,v in env.get_observations().items()},
            'joint_names':runtime.joint_names,'body_names':runtime.body_names,'reward_terms':env.reward_names,
            'randomization':{'mass_com_labels':env.mass_params_tensor.cpu().tolist(),
                'friction_labels':env.friction_coeffs_tensor.cpu().tolist(),
                'motor_strength':env.motor_strength.cpu().tolist(),
                'body_mass_readback':runtime.robot.data.body_mass.torch.cpu().tolist(),
                'body_com_readback':runtime.robot.data.body_com_pose_b.torch.cpu().tolist()},
            'force_stage':env.force_stage_info,'learning_success':False,
            'evidence_level':'limited official Lab integration execution; not complete UniFP training'}
        if args.force_boundary_probe:
            records=[]
            # Three zero-action physics steps cross the strict threshold.
            # Episode time is positioned at the sampled schedule start solely
            # for this labeled execution probe, after saving training output.
            env.global_steps=cfg.commands.force_start_step*24
            with torch.inference_mode():
                env.reset_idx(torch.arange(env.num_envs,device=env.device))
                env.episode_length_buf[:] = env.push_interval_gripper_ext[:,0]-1
            def record_transition(task):
                records.append({'global_steps_used':task.global_steps,
                    'active_during_physics':task.global_steps>cfg.commands.force_start_step*24,
                    'injected_world_force':task.forces[:,task.gripper_idx].cpu().tolist(),
                    'contact_world_force':task.contact_forces[:,task.gripper_idx].cpu().tolist(),
                    'gripper_position':task.gripper_position.cpu().tolist(),
                    'dones':task.reset_buf.cpu().tolist()})
            env.transition_observer=record_transition
            with torch.inference_mode():
                for _ in range(3):
                    env.step(torch.zeros(env.num_envs,env.num_actions,device=env.device))
            env.transition_observer=None
            evidence['force_boundary_execution_probe']={'artificial_global_steps':True,
                'artificial_episode_time':True,'zero_action_steps':3,'learning_evidence':False,'records':records}
        (args.output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__=='__main__':
    main()
