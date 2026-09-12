"""Experimental PRE solver-residual six-axis wrench; GPU owner only."""
import argparse
import json
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); p.add_argument('--usd-path',type=Path,default=ROOT/'reference/isaac/go1_widowx/usd_path.txt'); AppLauncher.add_app_launcher_args(p); args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
launcher=AppLauncher(args)
try:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from pawcerto.isaac.deepwbc_runtime import Go1WidowXIsaac
    from pawcerto.methods.deepwbc.config import default_config
    from analyze_deepwbc_wrench import FEET, SCENARIOS, rotation
    import warp as wp
    from isaaclab.utils.math import matrix_from_quat
    from pawcerto.isaac.solver_force import StockSolverForce
    cfg=default_config()
    ground=AssetBaseCfg(prim_path='/World/WrenchGround',spawn=sim_utils.CuboidCfg(size=(200.,200.,.1),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=cfg['sim']['physx']['contact_offset'],rest_offset=cfg['sim']['physx']['rest_offset']),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=cfg['terrain']['static_friction'],dynamic_friction=cfg['terrain']['dynamic_friction'],restitution=0.)),init_state=AssetBaseCfg.InitialStateCfg(pos=(0,0,-.05)))
    env=Go1WidowXIsaac(terrain_cfg=ground,terrain_contact_paths=['/World/WrenchGround/geometry/mesh'],num_envs=1,device=args.device,usd_path=args.usd_path,config=cfg)
    d=env.robot.data; ids=env.feet_indices; dt=env.dt
    def n(t): return t.detach().cpu().numpy().copy()
    view=env.robot.root_view
    def t(method):
        value=getattr(view,method)()
        return (value if isinstance(value,torch.Tensor) else wp.to_torch(value)).to(args.device).clone()
    properties={}
    def field(name,radians=False):
        values=[[env.sim.stage.GetPrimAtPath(path).GetAttribute('physxRigidBody:'+name).Get() for path in paths] for paths in view.link_paths]
        if any(x is None for row in values for x in row): raise ValueError('Missing property '+name)
        properties[name]=values
        result=torch.tensor(values,dtype=torch.float32,device=args.device)
        return torch.deg2rad(result) if radians else result
    parents=[-1]+[view.shared_metatype.link_parent_indices[name] for name in env.body_names[1:]]
    static={name:t(name) for name in ('get_masses','get_inertias','get_dof_max_velocities','get_dof_armatures','get_disable_gravities')}
    com_b=d.body_com_pose_b.torch[...,:3].clone()
    kwargs=dict(body_inertias=static['get_inertias'].reshape(1,len(parents),3,3),dt=dt,
        dof_max_velocity=static['get_dof_max_velocities'],body_max_linear_velocity=field('maxLinearVelocity'),
        body_max_angular_velocity=field('maxAngularVelocity',True),body_linear_damping=field('linearDamping'),
        body_angular_damping=field('angularDamping'),armature=static['get_dof_armatures'],
        disable_gravity=static['get_disable_gravities'],retain_accelerations=field('retainAccelerations'),
        gyroscopic_forces=field('enableGyroscopicForces'),external_forces_every_iteration=env.sim.cfg.physics.enable_external_forces_every_iteration)
    (args.output/'properties.json').write_text(json.dumps(dict(properties=properties,parents=parents,body_names=env.body_names,joint_names=env.robot.joint_names,link_paths=view.link_paths),indent=2))
    np.savez(args.output/'static.npz',**{k:n(v) for k,v in static.items()},com_offset_b=n(com_b))
    model=StockSolverForce(parents,static['get_masses'],**kwargs)
    gravity_force=static['get_masses'][...,None]*torch.tensor(cfg['sim']['gravity'],device=args.device)
    dynamic={'mass_matrix':'get_generalized_mass_matrices','gravity':'get_gravity_compensation_forces',
        'coriolis':'get_coriolis_and_centrifugal_compensation_forces','jacobian':'get_jacobians',
        'com_velocity':'get_link_velocities','dof_velocity':'get_dof_velocities','actuation':'get_dof_actuation_forces'}
    rows=[]
    for scenario in SCENARIOS:
        root=torch.zeros(1,13,device=args.device); root[:,2]=2. if scenario=='free_fall' else .42; root[:,6]=1
        if scenario=='moving_contact': root[:,10:13]=torch.tensor([.4,.2,.7],device=args.device)
        q=env.controller.default[None].clone(); qd=torch.zeros_like(q)
        if scenario=='moving_contact': qd[:,:12]=.5
        env.reset(torch.tensor([0],device=args.device),root,q,qd)
        # Box is deliberately moved away from all three ground-only scenarios.
        env.reset_box(torch.tensor([0],device=args.device),torch.tensor([[10.,10.,.21]],device=args.device))
        warmup=200 if scenario=='static_contact' else 1
        for step in range(warmup+40):
            q,qd=env.joints()
            env.last_torques=env.controller.torques(q,qd,env.motor_strength)
            env.robot.set_joint_effort_target(env.last_torques,joint_ids=env.joint_ids)
            for name in ('instantaneous_wrench_composer','permanent_wrench_composer'):
                if getattr(env.robot,name).active: raise ValueError('Unexpected external wrench '+name)
            env.scene.write_data_to_sim()
            pre_pose=d.body_com_pose_w.torch.clone()
            pre={key:t(method) for key,method in dynamic.items()}
            pre.update(com_position=pre_pose[...,:3],link_rotation=matrix_from_quat(d.body_link_quat_w.torch).clone(),external_wrench_terms=torch.zeros(1,device=args.device))
            if not torch.equal(pre['actuation'][:,env.joint_ids],env.last_torques): raise ValueError('Native torque differs from submitted command')
            if pre['actuation'].shape[-1]!=20: raise ValueError('Expected all 20 DOFs')
            predicted,qdd_free,bias=model.predict(**pre)
            env.sim.step(render=False); env.scene.update(dt); env.time+=dt
            reported=t('get_link_accelerations')
            candidate=model.approximate_sensor_wrench(reported,predicted,link_rotation=pre['link_rotation'],com_offset_b=com_b,applied_force_w=gravity_force)
            residual=model.approximate_sensor_wrench(reported,predicted,link_rotation=pre['link_rotation'],com_offset_b=com_b,applied_force_w=torch.zeros_like(gravity_force))
            state=env.state()
            if step<warmup: continue
            rows.append(dict(scenario=scenario,step=step-warmup,candidate=n(candidate[0,ids]),solver_only=n(residual[0,ids]),
                link_pose=np.concatenate((n(d.body_link_pos_w.torch[0,ids]),n(d.body_link_quat_w.torch[0,ids])),axis=-1),
                post_velocity=n(d.body_com_vel_w.torch[0,ids]),reported_acceleration=n(reported),free_acceleration=n(predicted),free_qdd=n(qdd_free),bias=n(bias),
                native_wrench=n(state['foot_wrenches'][0]),joint_pos=n(state['dof_pos'][0]),joint_vel=n(state['dof_vel'][0]),
                torques=n(state['torques'][0]),contact_force=n(state['contact_forces'][0,ids]),**{'pre_'+k:n(v) for k,v in pre.items()}))
    np.savez(args.output/'samples.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    (args.output/'metadata.json').write_text(json.dumps(dict(engine='official Isaac Lab/PhysX',dt=dt,feet=FEET,body_names=env.body_names,joint_names=env.joint_names,usd_path=str(args.usd_path),static_warmup=200,other_warmup=1,record_steps=40,randomization=False,inertia_convention='get_inertias COM inertia in prim axes; column-major; no COM quaternion rotation',candidate='PRE body axes: m*(a_reported-a_free)+mg; I_PRE*(alpha_reported-alpha_free)+r_COM cross F',sampling='each 5ms physics step; no 20ms averaging'),indent=2)+'\n')
except BaseException:
    if 'pre' in locals(): np.savez(args.output/'failure-frame.npz',**{k:n(v) for k,v in pre.items()})
    traceback.print_exc()
    raise
finally: launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
