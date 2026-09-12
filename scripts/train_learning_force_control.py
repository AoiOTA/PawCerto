"""Train the fixed released LFC task on official Isaac Lab with explicit updates."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iterations',required=True,type=int)
    p.add_argument('--num-envs',type=int)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--config',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--robot-usd',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--headless',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--force-boundary-probe',action='store_true')
    args=p.parse_args()
    if args.iterations<0 or (args.iterations==0 and not (args.resume and args.force_boundary_probe)):
        p.error('Positive iterations required, or resume with explicit boundary-only probe')
    import torch
    from pawcerto.methods.learning_force_control.training.config import default_config,resolve_config
    from pawcerto.artifacts import file_identity
    if args.resume:
        if args.config or args.num_envs is not None:
            p.error('Resume uses its saved config and environment count')
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        config=saved['config']
    else:
        config=json.loads(args.config.read_text()) if args.config else default_config()
        if args.num_envs is not None:config['env']['env']['num_envs']=args.num_envs
        config['seed']=args.seed
    previous=config.get('runtime',{})
    usd=(args.robot_usd or Path(previous.get('robot_usd',ROOT/'reference/isaac/b1_z1/usd_path.txt'))).resolve(strict=True)
    if usd.suffix=='.txt':usd=Path(usd.read_text().strip()).resolve(strict=True)
    identity={'robot_usd':file_identity(usd),'merged_urdf':file_identity(usd.parent.parent/'merged.urdf')}
    if args.resume:
        for key,value in identity.items():
            if value['sha256'] != previous['asset_identity'][key]['sha256']:
                raise ValueError('Resume robot asset differs: '+key)
    else:
        config['runtime']={'robot_usd':str(usd),'asset_identity':identity,
            'backend':'official unmodified Isaac Lab and PhysX',
            'force_body':'link06','measurement_body':'gripperMover',
            'measurement_provenance':'actual original Gym gripperStator lookup -1 reads final gripperMover body',
            'resume_semantics':'exposed task/physics state continuation; solver warm-start not available'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(visualizer='none' if args.headless else 'kit',device=args.device)
    runner=None
    try:
        import random
        import numpy as np
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
        from pawcerto.isaac.learning_force_control_runtime import B1Z1Isaac
        from pawcerto.methods.learning_force_control.training.isaac_env import LFCIsaacTrainingEnv
        from pawcerto.methods.learning_force_control.training.runner import OnPolicyRunner
        cfg=resolve_config(config)
        runtime=B1Z1Isaac(num_envs=cfg.env.num_envs,device=args.device,usd_path=usd,config=cfg)
        env=LFCIsaacTrainingEnv(runtime,config)
        runner=OnPolicyRunner(env,config,args.output,args.device,initialize=not bool(args.resume))
        if args.resume:runner.load(args.resume)
        if args.iterations:
            if not args.resume:runner.save(args.output/'model_0.pt')
            runner.learn(args.iterations,init_at_random_ep_len=not bool(args.resume))
        evidence={'next_iteration':runner.next_iteration,'global_steps':runner.global_steps,
            'common_step_counter':env.common_step_counter,'environment_sample_steps':env.global_steps,
            'optimizer_updates_this_process':args.iterations,'num_envs':env.num_envs,
            'rollout_steps':runner.num_steps_per_env,'policy_dt':env.dt,'physics_dt':runtime.dt,
            'observations':{'obs':list(env.obs_buf.shape),'privileged_obs':list(env.privileged_obs_buf.shape),'obs_history':list(env.history.history.shape)},
            'joint_names':runtime.joint_names,'body_names':runtime.body_names,'reward_terms':env.reward_names,
            'force_body':runtime.body_names[env.gripper_stator_index],
            'measurement_body':runtime.body_names[env.ee_measurement_index],
            'task':'released locomotion plus binary force/position manipulation with random spring forces; add_balls=False',
            'randomization':{'friction_labels':env.friction_coeffs.cpu().tolist(),
                'com_labels':env.com_displacements.cpu().tolist(),'motor_strength_labels':env.motor_strengths.cpu().tolist(),
                'kp_factor_labels':env.Kp_factors.cpu().tolist(),'kd_factor_labels':env.Kd_factors.cpu().tolist(),
                'gripper_kp':env.gripper_force_kps.cpu().tolist(),'gripper_kd':env.gripper_force_kds.cpu().tolist()},
            'learning_success':False,'evidence_level':'bounded official Lab integration; not original long training'}
        if args.force_boundary_probe:
            records=[]
            def record(task):
                records.append({'episode_step':task.episode_length_buf.cpu().tolist(),
                    'hybrid':task.force_or_position_control.cpu().tolist(),'freed':task.freed_envs.cpu().tolist(),
                    'selected':task.selected_env_ids.cpu().tolist(),'force_commands':task.commands[:,12:15].cpu().tolist(),
                    'applied_world_force':task.forces[:,task.gripper_stator_index].cpu().tolist(),
                    'position_targets':task.joint_pos_target.cpu().tolist(),'reset':task.reset_buf.cpu().tolist()})
            with torch.inference_mode():
                # Force schedules use episode steps, not a late-training threshold.
                # Move only the diagnostic episode counters, after final checkpoint.
                env.episode_length_buf[:] = env.push_interval[:,0].long()-1
                env.transition_observer=record
                for _ in range(3):env.step(torch.zeros(env.num_envs,env.num_actions,device=env.device))
                env.transition_observer=None
            evidence['force_boundary_execution_probe']={'artificial_episode_time':True,'zero_action_policy_steps':3,
                'stage':'one step before source push interval, at interval, then ramp continuation',
                'policy_learning_evidence':False,'records':records}
        (args.output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    except BaseException as exc:
        import traceback
        failure={'exception':repr(exc),'traceback':traceback.format_exc()}
        if runner is not None:
            failure.update(next_iteration=runner.next_iteration,global_steps=runner.global_steps,
                           incomplete_rollout_steps=runner.alg.storage.step)
        (args.output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__=='__main__':main()
