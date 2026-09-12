"""Bounded original Gym drive readback and Lab action/state probe."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT = ROOT / 'outputs/roboduet-runtime-20260912'


def gym_probe():
    from isaacgym import gymapi
    import numpy as np
    import runpy
    asset_data = runpy.run_path(str(ROOT / 'pawcerto/mujoco/roboduet_asset.py'))
    URDF, JOINT_NAMES, DEFAULT_POS = (asset_data[n] for n in ('URDF', 'JOINT_NAMES', 'DEFAULT_POS'))
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt=.005
    params.up_axis=gymapi.UP_AXIS_Z
    params.gravity=gymapi.Vec3(0,0,-9.81)
    params.physx.use_gpu=False
    params.physx.num_position_iterations=4
    params.physx.num_velocity_iterations=1
    sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params)
    if sim is None:
        raise RuntimeError('Gym create_sim failed')
    options=gymapi.AssetOptions()
    options.collapse_fixed_joints=True
    options.default_dof_drive_mode=gymapi.DOF_MODE_EFFORT
    options.replace_cylinder_with_capsule=True
    options.flip_visual_attachments=True
    options.density=.001
    options.angular_damping=0.
    options.linear_damping=0.
    options.max_angular_velocity=1000.
    options.max_linear_velocity=1000.
    options.armature=0.
    options.thickness=.01
    # Isolate the actual actuator semantics using the source robot with a fixed
    # root and no gravity. This is a drive diagnostic, not a rollout result.
    options.fix_base_link=True
    options.disable_gravity=True
    asset=gym.load_asset(sim,str(URDF.parent),URDF.name,options)
    names=gym.get_asset_dof_names(asset)
    assert names==JOINT_NAMES,(names,JOINT_NAMES)
    env=gym.create_env(sim,gymapi.Vec3(-2,-2,0),gymapi.Vec3(2,2,2),1)
    actor=gym.create_actor(env,asset,gymapi.Transform(), 'robot',0,0)
    props=gym.get_actor_dof_properties(env,actor)
    kp=[40,70,70,25,25,25,50,50]
    kd=[3,15,15,2,2,2,20,20]
    props['stiffness'][12:]=kp
    props['damping'][12:]=kd
    gym.set_actor_dof_properties(env,actor,props)
    body_names=gym.get_actor_rigid_body_names(env,actor)
    bodies=gym.get_actor_rigid_body_properties(env,actor)
    before=[{'mass':b.mass,'inertia':[b.inertia.x.x,b.inertia.y.y,b.inertia.z.z]} for b in bodies]
    assert body_names[23]=='zarx_body6',body_names
    bodies[23].mass+=.1
    gym.set_actor_rigid_body_properties(env,actor,bodies,True)
    bodies=gym.get_actor_rigid_body_properties(env,actor)
    after=[{'mass':b.mass,'inertia':[b.inertia.x.x,b.inertia.y.y,b.inertia.z.z]} for b in bodies]
    options.fix_base_link=False
    mass_asset=gym.load_asset(sim,str(URDF.parent),URDF.name,options)
    mass_env=gym.create_env(sim,gymapi.Vec3(-2,-2,0),gymapi.Vec3(2,2,2),1)
    mass_actor=gym.create_actor(mass_env,mass_asset,gymapi.Transform(),'mass_robot',1,0)
    gym.prepare_sim(sim)
    initial=np.zeros(20,dtype=gymapi.DofState.dtype)
    initial['pos']=DEFAULT_POS
    initial['pos'][12]=.3
    cases={}
    for name,target,effort,stiffness in [('none',0.,0.,40.),('position',.6,0.,40.),('effort',0.,.6,40.),('both',.6,.6,40.),('zero_stiffness',.6,.6,0.)]:
        local=props.copy()
        local['stiffness'][12]=stiffness
        gym.set_actor_dof_properties(env,actor,local)
        gym.set_actor_dof_states(env,actor,initial,gymapi.STATE_ALL)
        targets=np.asarray(DEFAULT_POS,dtype=np.float32)
        targets[12]=target
        forces=np.zeros(20,dtype=np.float32)
        forces[12]=effort
        gym.set_actor_dof_position_targets(env,actor,targets)
        gym.apply_actor_dof_efforts(env,actor,forces)
        gym.simulate(sim)
        gym.fetch_results(sim,True)
        state=gym.get_actor_dof_states(env,actor,gymapi.STATE_ALL)
        cases[name]={'q':state['pos'].tolist(),'qd':state['vel'].tolist()}
    mass_cases = {}
    for name, mass_value in [('positive', .8), ('negative', -.5), ('restore', before[0]['mass'])]:
        mass_props = gym.get_actor_rigid_body_properties(mass_env,mass_actor)
        mass_props[0].mass = mass_value
        mass_props[0].com = gymapi.Vec3(0.,0.,0.)
        result = gym.set_actor_rigid_body_properties(mass_env,mass_actor,mass_props,True)
        read_mass = gym.get_actor_rigid_body_properties(mass_env,mass_actor)[0]
        gym.set_actor_dof_states(mass_env,mass_actor,initial,gymapi.STATE_ALL)
        gym.simulate(sim)
        gym.fetch_results(sim,True)
        state = gym.get_actor_dof_states(mass_env,mass_actor,gymapi.STATE_ALL)
        mass_cases[name] = {'requested':mass_value,'return':result,'mass':read_mass.mass,
            'inertia':[read_mass.inertia.x.x,read_mass.inertia.y.y,read_mass.inertia.z.z],
            'finite_after_step':bool(np.isfinite(state['pos']).all() and np.isfinite(state['vel']).all())}
    readback=gym.get_actor_dof_properties(env,actor)
    report={'joint_names':names,'body_names':body_names,'body_before_camera':before,'body_after_camera':after,
        'mass_cases':mass_cases,'drive_mode':readback['driveMode'].tolist(),'original_stiffness':props['stiffness'].tolist(),
        'original_damping':props['damping'].tolist(),'cases':cases,'initial_q':initial['pos'].tolist(),
        'scope':'fixed-root no-gravity CPU PhysX one-step drive diagnostic'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'gym-drive.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:case['qd'][12] for name,case in cases.items()}),flush=True)
    gym.destroy_sim(sim)


def contact_probe(runtime):
    import torch
    from types import SimpleNamespace
    from pawcerto.methods.roboduet.training.task import Rewards
    records=[]
    metadata={name:{'filters':runtime.scene['contact_'+name].cfg.filter_prim_paths_expr,
        'filter_count':runtime.scene['contact_'+name].contact_view.filter_count}
        for name in runtime.contact_names}
    for phase in ('slide','self_contact'):
        q=runtime.default_pos.expand(16,-1).clone()
        pose=torch.zeros((16,7),device=runtime.device)
        pose[:,2]=.34 if phase=='slide' else .6
        pose[:,3]=1.
        velocity=torch.zeros((16,6),device=runtime.device)
        velocity[:8,0]=.7
        velocity[8:,0]=-.7
        if phase=='self_contact':
            q[:,13]=-1.5
            q[:,14]=1.5
        runtime.reset(joint_pos=q,root_pose=pose,root_velocity=velocity)
        runtime.set_stage(1 if phase=='slide' else 2)
        for substep in range(48 if phase=='slide' else 8):
            current,qd=runtime.joints()
            leg=(35*(runtime.default_pos[:12]-current[:,:12])-qd[:,:12]).clamp(-runtime.torque_limits[:12],runtime.torque_limits[:12])
            state=runtime.step_control(leg,runtime.default_pos[12:].expand(16,-1))
            normal=torch.zeros_like(state['contact_forces'])
            filtered=torch.zeros_like(normal)
            ground_friction=torch.zeros_like(normal)
            self_friction=torch.zeros_like(normal)
            self_normal=torch.zeros_like(normal)
            for name in runtime.contact_names:
                body=runtime.body_names.index(name)
                data=runtime.scene['contact_'+name].data
                normal[:,body]=data.net_normal_forces_w.torch[:,0]
                filtered[:,body]=data.normal_force_matrix_w.torch[:,0].sum(1)
                ground_friction[:,body]=data.friction_force_matrix_w.torch[:,0,0]
                self_friction[:,body]=data.friction_force_matrix_w.torch[:,0,1:].sum(1)
                self_normal[:,body]=data.normal_force_matrix_w.torch[:,0,1:].sum(1)
            assert torch.isfinite(state['contact_forces']).all()
            entry={'phase':phase,'substep':substep,'normal':normal.cpu(),'filtered_normal':filtered.cpu(),
                'ground_friction':ground_friction.cpu(),'self_friction':self_friction.cpu(),
                'self_normal':self_normal.cpu(),'total':state['contact_forces'].cpu(),
                'body_velocity':state['body_lin_vel'].cpu()}
            # Call the real reward consumer only at the policy boundary, with
            # its upstream desired-contact input held at 0.5 for this probe.
            if (substep+1)%4==0:
                env=SimpleNamespace(contact_forces=state['contact_forces'],feet_indices=runtime.feet_indices,
                    desired_contact_states=torch.full((16,4),.5,device=runtime.device),
                    cfg=SimpleNamespace(rewards=SimpleNamespace(gait_force_sigma=100.)))
                reward=Rewards(env)
                entry['reward_total']=reward._reward_tracking_contacts_shaped_force().cpu()
                env.contact_forces=normal
                entry['reward_normal_only']=reward._reward_tracking_contacts_shaped_force().cpu()
            records.append(entry)
    torch.save({'records':records,'body_names':runtime.body_names,'metadata':metadata},OUT/'contact-forces.pt')
    report={'physics_dt':runtime.dt,'policy_decimation':4,'num_envs':16,'metadata':metadata,
        'normal_filter_residual_max_N':max(float((r['normal']-r['filtered_normal']).abs().max()) for r in records),
        'ground_friction_max_N':max(float(r['ground_friction'].abs().max()) for r in records),
        'self_friction_max_N':max(float(r['self_friction'].abs().max()) for r in records),
        'self_normal_max_N':max(float(r['self_normal'].abs().max()) for r in records),
        'sum_identity_max_N':max(float((r['total']-r['normal']-r['ground_friction']-r['self_friction']).abs().max()) for r in records),
        'reward_difference_max':max(float((r['reward_total']-r['reward_normal_only']).abs().max()) for r in records if 'reward_total' in r)}
    reward_env=SimpleNamespace(feet_indices=runtime.feet_indices,desired_contact_states=torch.full((16,4),.5),
        cfg=SimpleNamespace(rewards=SimpleNamespace(gait_force_sigma=100.)))
    reward=Rewards(reward_env)
    maximum=(0.,None)
    for record in records:
        reward_env.contact_forces=record['total']
        total_reward=reward._reward_tracking_contacts_shaped_force()
        reward_env.contact_forces=record['normal']
        normal_reward=reward._reward_tracking_contacts_shaped_force()
        difference=float((total_reward-normal_reward).abs().max())
        if difference>maximum[0]:
            maximum=(difference,(record['phase'],record['substep']))
    report['recorded_substep_reward_difference_max']=maximum[0]
    report['recorded_substep_reward_difference_location']=maximum[1]
    report['self_normal_pair_balance_max_N']=max(float(r['self_normal'].sum(1).abs().max()) for r in records)
    report['self_friction_pair_balance_max_N']=max(float(r['self_friction'].sum(1).abs().max()) for r in records)
    (OUT/'contact-forces.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='metadata'}),flush=True)


def lab_probe():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser()
    parser.add_argument('--engine',default='lab')
    parser.add_argument('--contact-only',action='store_true')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    launcher=AppLauncher(args)
    try:
        import torch
        from pawcerto.isaac.roboduet_runtime import Go1Arx5Isaac
        runtime=Go1Arx5Isaac(num_envs=16,device=args.device)
        if args.contact_only:
            contact_probe(runtime)
            return
        before=runtime.joints()[0].clone()
        report={'joint_names':runtime.joint_names,'native_joint_names':runtime.robot.joint_names,
            'body_names':runtime.body_names,'masses':runtime.robot.data.body_mass.torch[0].tolist(),
            'inertias':runtime.robot.data.body_inertia.torch[0].tolist(),'phases':{}}
        for stage in (1,2):
            runtime.reset()
            runtime.set_stage(stage)
            start=runtime.joints()[0].clone()
            target=runtime.default_pos[12:].expand(16,-1).clone()
            target[:,0]+=.3
            rows=[]
            for step in range(12):
                q,qd=runtime.joints()
                leg=(35*(runtime.default_pos[:12]-q[:,:12])-qd[:,:12]).clamp(-runtime.torque_limits[:12],runtime.torque_limits[:12])
                state=runtime.step_control(leg,target)
                if not all(bool(torch.isfinite(value).all()) for value in state.values()):
                    raise RuntimeError(f'Nonfinite state in stage {stage}, step {step}')
                locked=12 if stage==1 else 18
                rows.append({'step':step,'q':state['joint_pos'][0].tolist(),'qd':state['joint_vel'][0].tolist(),
                    'locked_postsolve_max_error':float((state['joint_pos'][:,locked:]-runtime.default_pos[locked:]).abs().max()),
                    'root_pos':state['root_pos'][0].tolist(),'ee_pos':state['ee_pos'][0].tolist()})
            report['phases'][str(stage)]={'samples':rows,'arm_displacement':float((runtime.joints()[0][:,12:18]-start[:,12:18]).abs().max())}
        (OUT/'lab-runtime.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({stage:value['arm_displacement'] for stage,value in report['phases'].items()}),flush=True)
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))

if __name__=='__main__':
    if '--engine' in sys.argv and sys.argv[sys.argv.index('--engine')+1]=='gym':
        gym_probe()
    else:
        lab_probe()
