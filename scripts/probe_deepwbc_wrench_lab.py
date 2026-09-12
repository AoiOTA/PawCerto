"""Read-only adapter probe of default net-body wrench hypothesis; GPU owner only."""
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
    from analyze_deepwbc_wrench import FEET, SCENARIOS, net_wrench
    cfg=default_config()
    ground=AssetBaseCfg(prim_path='/World/WrenchGround',spawn=sim_utils.CuboidCfg(size=(200.,200.,.1),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=cfg['sim']['physx']['contact_offset'],rest_offset=cfg['sim']['physx']['rest_offset']),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=cfg['terrain']['static_friction'],dynamic_friction=cfg['terrain']['dynamic_friction'],restitution=0.)),init_state=AssetBaseCfg.InitialStateCfg(pos=(0,0,-.05)))
    env=Go1WidowXIsaac(terrain_cfg=ground,terrain_contact_paths=['/World/WrenchGround/geometry/mesh'],num_envs=1,device=args.device,usd_path=args.usd_path,config=cfg)
    d=env.robot.data; ids=env.feet_indices; dt=env.dt
    def n(t): return t.detach().cpu().numpy().copy()
    mass=n(d.body_mass.torch[0,ids]); inertia=n(d.body_inertia.torch[0,ids]).reshape(4,3,3).swapaxes(-1,-2); com=n(d.body_com_pose_b.torch[0,ids,:3])
    if not all(np.isfinite(x).all() for x in (mass,inertia,com)): raise ValueError('Nonfinite imported mass/inertia/COM; asset repair required')
    np.savez(args.output/'static.npz',mass=mass,inertia=inertia,com=com,all_com=n(d.body_com_pose_b.torch),all_mass=n(d.body_mass.torch),all_inertia=n(d.body_inertia.torch))
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
            pre=n(d.body_com_vel_w.torch[0,ids]); state=env.step_control()
            post=n(d.body_com_vel_w.torch[0,ids]); pose=np.concatenate((n(d.body_link_pos_w.torch[0,ids]),n(d.body_link_quat_w.torch[0,ids])),axis=-1); acc=n(d.body_com_acc_w.torch[0,ids])
            if not np.isfinite(pose).all() or not np.isfinite(acc).all(): raise ValueError('Nonfinite physics state')
            if step<warmup: continue
            rows.append(dict(scenario=scenario,step=step-warmup,link_pose=pose,com_pose=n(d.body_com_pose_w.torch[0,ids]),pre_velocity=pre,post_velocity=post,reported_acceleration=acc,candidate=net_wrench(mass,inertia,pose[:,3:],post[:,3:],acc,com),fd_candidate=net_wrench(mass,inertia,pose[:,3:],post[:,3:],(post-pre)/dt,com),native_wrench=n(state['foot_wrenches'][0]),joint_pos=n(state['dof_pos'][0]),joint_vel=n(state['dof_vel'][0]),torques=n(state['torques'][0]),contact_force=n(state['contact_forces'][0,ids])))
    np.savez(args.output/'samples.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    (args.output/'metadata.json').write_text(json.dumps(dict(engine='official Isaac Lab/PhysX',dt=dt,feet=FEET,body_names=env.body_names,joint_names=env.joint_names,usd_path=str(args.usd_path),static_warmup=200,other_warmup=1,record_steps=40,randomization=False,inertia_convention='get_inertias COM inertia in prim axes; column-major; no COM quaternion rotation',candidate='m*aCOM; Iw*alpha+omega cross Iw*omega + rCOM cross F; body axes',sampling='each 5ms physics step; no 20ms averaging'),indent=2)+'\n')
except BaseException:
    traceback.print_exc()
    raise
finally: launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
