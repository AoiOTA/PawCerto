"""One-environment original default Gym force sensor probe; GPU owner only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from isaacgym import gymapi, gymtorch
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pawcerto.methods.deepwbc.config import default_config
from pawcerto.methods.deepwbc.controller import DeepWBCController
from analyze_deepwbc_wrench import FEET, SCENARIOS, rotation, net_wrench
p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); p.add_argument('--friction-probe-only',action='store_true'); p.add_argument('--friction-slide',action='store_true'); p.add_argument('--plane-friction',type=float,default=1.); args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
cfg=default_config(); dt=cfg['sim']['dt']; gym=gymapi.acquire_gym(); params=gymapi.SimParams(); params.dt=dt; params.up_axis=gymapi.UP_AXIS_Z; params.gravity=gymapi.Vec3(*cfg['sim']['gravity']); params.use_gpu_pipeline=False
for k in ('solver_type','num_position_iterations','num_velocity_iterations','contact_offset','rest_offset','bounce_threshold_velocity','max_depenetration_velocity'):
    setattr(params.physx,k,cfg['sim']['physx'][k])
params.physx.use_gpu=True; params.substeps=cfg['sim']['substeps']
sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params)
if sim is None: raise RuntimeError('Gym create_sim failed')
try:
    plane=gymapi.PlaneParams(); plane.normal=gymapi.Vec3(0,0,1); plane.static_friction=args.plane_friction if args.friction_slide else cfg['terrain']['static_friction']; plane.dynamic_friction=args.plane_friction if args.friction_slide else cfg['terrain']['dynamic_friction']; gym.add_ground(sim,plane)
    if args.friction_slide:
        values=(-.5,-.25,0.,.25,.5,1.,None)
        slide_env=gym.create_env(sim,gymapi.Vec3(-4,-4,0),gymapi.Vec3(4,4,4),1)
        records=[]
        for i,value in enumerate(values):
            box_options=gymapi.AssetOptions(); box_options.density=1000.; box_options.linear_damping=0.; box_options.angular_damping=0.
            box=gym.create_box(sim,.1,.1,.1,box_options)
            props=gym.get_asset_rigid_shape_properties(box)
            original=[float(prop.friction) for prop in props]
            if value is not None:
                for prop in props: prop.friction=value
                setter_result=gym.set_asset_rigid_shape_properties(box,props)
            else: setter_result=None
            actor=gym.create_actor(slide_env,box,gymapi.Transform(gymapi.Vec3(0,float(i),.055)),f'friction_{i}',0,0)
            records.append({'requested':value,'default_asset_friction':original,'setter_return':setter_result,'asset_readback':[float(prop.friction) for prop in gym.get_asset_rigid_shape_properties(box)],'actor_readback':[float(prop.friction) for prop in gym.get_actor_rigid_shape_properties(slide_env,actor)],'mass':float(gym.get_actor_rigid_body_properties(slide_env,actor)[0].mass)})
        metadata={'dt':dt,'gravity':cfg['sim']['gravity'],'plane_static_friction':plane.static_friction,'plane_dynamic_friction':plane.dynamic_friction,'plane_combine_mode':'not exposed by Gym PlaneParams; infer only from measured pairs','shape_field_documentation':'RigidShapeProperties.friction documented as static friction; no separate dynamic-friction field','box_size':[.1,.1,.1],'initial_slide_velocity':[2.,0.,0.],'values':values,'materials':records,'phase_steps':{'settle':40,'slide':160,'push':200},'push_force':'world x = mass*9.81*(step/199)*2, freshly reset stationary box; no torque','boundary':'Simple box contact experiment; does not prove all contact or robot material semantics'}
        (args.output/'slide-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        gym.prepare_sim(sim)
        roots=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)); contacts=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
        rows=[]
        def slide_refresh():
            gym.refresh_actor_root_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
        for phase in ('slide','push'):
            roots.zero_(); roots[:,1]=torch.arange(len(values),dtype=torch.float32); roots[:,2]=.055; roots[:,6]=1.
            gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(roots))
            for step in range(40): gym.simulate(sim); gym.fetch_results(sim,True)
            slide_refresh()
            if phase=='slide':
                roots[:,7]=2.; gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(roots))
            for step in range(160 if phase=='slide' else 200):
                pre=roots.numpy().copy(); applied=np.zeros((len(values),3),np.float32)
                if phase=='push':
                    applied[:,0]=np.array([r['mass'] for r in records])*9.81*(step/199)*2
                    forces=torch.from_numpy(applied.copy()); torques=torch.zeros_like(forces)
                    gym.apply_rigid_body_force_tensors(sim,gymtorch.unwrap_tensor(forces),gymtorch.unwrap_tensor(torques),gymapi.GLOBAL_SPACE)
                gym.simulate(sim); gym.fetch_results(sim,True); slide_refresh()
                rows.append(dict(phase=phase,step=step,pre_state=pre,post_state=roots.numpy().copy(),contact_force=contacts.numpy().copy(),applied_force=applied))
        np.savez(args.output/'slide-samples.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        print('FRICTION_SLIDE_PROBE_COMPLETE',flush=True)
        sys.exit(0)
    opts=gymapi.AssetOptions()
    for k in ('default_dof_drive_mode','collapse_fixed_joints','replace_cylinder_with_capsule','flip_visual_attachments','fix_base_link','density','angular_damping','linear_damping','max_angular_velocity','max_linear_velocity','armature','thickness','disable_gravity'): setattr(opts,k,cfg['asset'][k])
    opts.use_mesh_materials=True
    urdf=ROOT/'third_party/deepwbc-reference/legged_gym/resources/robots/widowGo1/urdf/widowGo1.urdf'
    if args.friction_probe_only:
        # Original shape setter path, no step or randomization repair/clamping.
        records=[]
        friction_env=gym.create_env(sim,gymapi.Vec3(-4,-4,0),gymapi.Vec3(4,4,4),1)
        for value in (-.25,0.,1.):
            friction_asset=gym.load_asset(sim,str(urdf.parent),urdf.name,opts)
            shape_props=gym.get_asset_rigid_shape_properties(friction_asset)
            row={'requested_friction':value,'shape_count':len(shape_props),'stage':'before_set_asset'}; records.append(row)
            for prop in shape_props: prop.friction=value
            (args.output/'friction-properties.json').write_text(json.dumps(records,indent=2)+'\n')
            setter_result=gym.set_asset_rigid_shape_properties(friction_asset,shape_props)
            row.update(setter_return=setter_result,stage='asset_set_returned',asset_friction=[float(prop.friction) for prop in gym.get_asset_rigid_shape_properties(friction_asset)])
            (args.output/'friction-properties.json').write_text(json.dumps(records,indent=2)+'\n')
            friction_actor=gym.create_actor(friction_env,friction_asset,gymapi.Transform(gymapi.Vec3(0,0,2)),str(value),0,cfg['asset']['self_collisions'])
            row.update(stage='actor_created',actor_friction=[float(prop.friction) for prop in gym.get_actor_rigid_shape_properties(friction_env,friction_actor)])
            (args.output/'friction-properties.json').write_text(json.dumps(records,indent=2)+'\n')
        print('FRICTION_ASSET_PROBE_COMPLETE',flush=True)
        sys.exit(0)
    asset=gym.load_asset(sim,str(urdf.parent),urdf.name,opts); bodies=gym.get_asset_rigid_body_names(asset); names=gym.get_asset_dof_names(asset); foot_ids=[bodies.index(n) for n in FEET]
    sensor_modes = {'default_omitted': None, 'explicit_both': (True,True), 'fd_only': (True,False), 'constraint_only': (False,True)}
    for mode, flags in sensor_modes.items():
        for i in foot_ids:
            pose=gymapi.Transform(gymapi.Vec3(0,0,0))
            if flags is None:
                gym.create_asset_force_sensor(asset,i,pose)
            else:
                sp=gymapi.ForceSensorProperties(); sp.enable_forward_dynamics_forces=flags[0]; sp.enable_constraint_solver_forces=flags[1]; sp.use_world_frame=False
                gym.create_asset_force_sensor(asset,i,pose,sp)
    env=gym.create_env(sim,gymapi.Vec3(-4,-4,0),gymapi.Vec3(4,4,4),1)
    actor=gym.create_actor(env,asset,gymapi.Transform(gymapi.Vec3(0,0,.4)),'robot',0,cfg['asset']['self_collisions'])
    props=gym.get_asset_dof_properties(asset); props['driveMode'].fill(gymapi.DOF_MODE_EFFORT); props['stiffness'].fill(0); props['damping'].fill(0); gym.set_actor_dof_properties(env,actor,props)
    dof_report = {'joint_names': names, 'joint_types': [int(gym.get_asset_dof_type(asset,i)) for i in range(len(names))], 'asset': {k: gym.get_asset_dof_properties(asset)[k].tolist() for k in ('lower','upper','effort','velocity','hasLimits')}, 'actor': {k: gym.get_actor_dof_properties(env,actor)[k].tolist() for k in ('lower','upper','effort','velocity','hasLimits')}}
    (args.output/'dof-properties.json').write_text(json.dumps(dof_report,indent=2)+'\n')
    bp=gym.get_actor_rigid_body_properties(env,actor)
    gym.set_actor_rigid_body_properties(env,actor,bp,recomputeInertia=True)
    bp=gym.get_actor_rigid_body_properties(env,actor)
    mass=np.array([bp[i].mass for i in foot_ids]); com=np.array([[bp[i].com.x,bp[i].com.y,bp[i].com.z] for i in foot_ids]); inertia=np.array([[[getattr(getattr(bp[i].inertia,c),r) for c in 'xyz'] for r in 'xyz'] for i in foot_ids])
    np.savez(args.output/'static.npz',mass=mass,com=com,inertia=inertia)
    gym.prepare_sim(sim)
    sensor=gymtorch.wrap_tensor(gym.acquire_force_sensor_tensor(sim)); body=gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)); dof=gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)); root=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)); contact=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
    ctl=DeepWBCController(1,names,props['effort'],cfg,'cpu'); q0=ctl.default.clone(); rows=[]
    def refresh():
        gym.refresh_rigid_body_state_tensor(sim); gym.refresh_dof_state_tensor(sim); gym.refresh_actor_root_state_tensor(sim); gym.refresh_force_sensor_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
    def snapshot(): return body[foot_ids].numpy().copy()
    for scenario in SCENARIOS:
        root.zero_(); root[:,2]=2. if scenario=='free_fall' else .42; root[:,6]=1
        if scenario=='moving_contact': root[:,10:13]=torch.tensor([.4,.2,.7])
        dof[:,0]=q0; dof[:,1]=0
        if scenario=='moving_contact': dof[:12,1]=.5
        gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(root)); gym.set_dof_state_tensor(sim,gymtorch.unwrap_tensor(dof)); ctl.reset(torch.tensor([0])); refresh()
        warmup=200 if scenario=='static_contact' else 1
        for step in range(warmup+40):
            pre=snapshot(); tau=ctl.torques(dof[:,0][None],dof[:,1][None],torch.ones(1,18)).contiguous(); gym.set_dof_actuation_force_tensor(sim,gymtorch.unwrap_tensor(tau)); gym.simulate(sim); gym.fetch_results(sim,True); refresh(); post=snapshot()
            if step<warmup: continue
            # Gym raw rigid-body velocity; COM interpretation below is sensitivity only.
            accel=(post[:,7:13]-pre[:,7:13])/dt
            rows.append(dict(scenario=scenario,step=step-warmup,link_pose=post[:,:7],pre_velocity=pre[:,7:13],post_velocity=post[:,7:13],raw_wrench=sensor.numpy().reshape(4,4,6)[0].copy(),sensor_modes_wrench=sensor.numpy().reshape(4,4,6).copy(),fd_candidate=net_wrench(mass,inertia,post[:,3:7],post[:,10:13],accel,com),joint_pos=dof[:,0].numpy().copy(),joint_vel=dof[:,1].numpy().copy(),torques=tau.numpy()[0],contact_force=contact[foot_ids].numpy().copy()))
    np.savez(args.output/'samples.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    (args.output/'metadata.json').write_text(json.dumps(dict(engine='Isaac Gym Preview 4',dt=dt,feet=FEET,body_names=bodies,joint_names=names,sensor_properties=sensor_modes,sensor_mode_order=list(sensor_modes),static_warmup=200,other_warmup=1,record_steps=40,randomization=False,protocol='default pose; static/moving z=.42, free z=2; default PD; moving initial angular velocity (.4,.2,.7) and leg joint velocity .5'),indent=2)+'\n')
finally: gym.destroy_sim(sim)
