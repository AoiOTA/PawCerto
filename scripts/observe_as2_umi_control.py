"""Record one nominal AS2 UMI zero-action PD run: 3 s, 600 original 5 ms steps."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    from pawcerto.methods.umi_on_legs.robot_binding import robot_binding,joint_order
    binding=robot_binding(config)
    if binding['name']!='as2_piper':
        raise ValueError('This bounded observation is assigned to nominal AS2 only')
    if config['env']['cfg']['sim']['dt']!=.005 or config['env']['controller']['decimation_count']!=4:
        raise ValueError('This observation requires the original four 5 ms UMI steps')
    usd_path=Path(config.get('pawcerto_asset',{}).get('usd_path') or binding['usd_path'])
    if usd_path.suffix=='.txt':
        usd_path=Path(usd_path.read_text().strip())
    urdf_path=Path(config.get('pawcerto_asset',{}).get('urdf_path') or binding['urdf_path'])
    args.output.mkdir(parents=True,exist_ok=False)
    from pawcerto.artifacts import file_identity
    metadata=dict(config=file_identity(args.config),usd=file_identity(usd_path),urdf=file_identity(urdf_path),
                  engine='Isaac Lab PhysX',robot=binding,training=False,domain_randomization=False,
                  policy='constant zero action, no actor',physics_dt=.005,policy_dt=.02,seconds=3.,
                  controller=config['env']['controller'],init_state=config['env']['cfg']['init_state'],
                  force_signal='reconstructed-solver',
                  sampling='after each existing physics step and Lab scene update; initial state recorded before first step',
                  evidence='Nominal deterministic basic control observation; not learned WBC or force-reference equivalence')
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    launcher=AppLauncher(args)
    try:
        import torch
        import warp as wp
        from pawcerto.isaac.runtime import Go2Arx5Isaac
        from pawcerto.methods.umi_on_legs import UmiController
        torch.manual_seed(0)
        env=Go2Arx5Isaac(config,joint_order(config),1,args.device,usd_path=usd_path,
                         urdf_path=urdf_path,training=False,ground_contact_diagnostics=True,
                         force_signal='reconstructed-solver')
        controller=UmiController(config,1,args.device)
        # Exercise the same real reset owner used by training, without randomization.
        env.reset();controller.reset()
        zero=torch.zeros((1,18),device=args.device)
        metadata['body_names']=list(env.body_names)
        metadata['runtime_joint_names']=list(env.robot.joint_names)
        metadata['controlled_joint_ids']=env.joint_ids
        metadata['body_masses_kg']=env.robot.data.body_mass.torch[0].cpu().tolist()
        (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        rows=[]
        def sample(index,torque):
            state=env.state();raw=env.training_state()
            effort=wp.to_torch(env.robot.root_view.get_dof_actuation_forces())[:,env.joint_ids]
            values=dict(root_pos=raw['root_pos'],root_quat_wxyz=env.robot.data.root_link_quat_w.torch[:,[3,0,1,2]],
                        root_up_dot=-state.local_root_gravity[:,2],q=state.dof_pos,qd=state.dof_vel,
                        commanded_tau=torque,applied_tau=effort,tcp_pose=state.ee_pose,
                        contact_forces_w=raw['contact_forces'],feet_normal_net_z=raw['feet_force_z'],
                        feet_ground_z=env.feet_ground_force_z())
            if 'feet_solver_force_z' in raw:
                values['feet_solver_force_z']=raw['feet_solver_force_z']
            record=dict(physics_step=index,time_s=float(env.time[0].item()),
                        solver_force_valid=bool(raw['feet_solver_force_valid'][0]))
            for key,value in values.items():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f'Nonfinite {key} at physics step {index}')
                record[key]=value[0].detach().cpu().tolist()
            record['ground_supported_feet']=sum(v>1. for v in record['feet_ground_z'])
            return record
        with (args.output/'steps.jsonl').open('x') as stream:
            first=sample(0,zero);stream.write(json.dumps(first)+'\n');stream.flush();rows.append(first)
            for step in range(600):
                substep=step%4
                if substep==0:
                    controller.push(zero)
                q,qd=env.joints()
                torque=controller.torque(q,qd,substep)
                env.step_torque(torque)
                record=sample(step+1,torque)
                stream.write(json.dumps(record)+'\n');stream.flush();rows.append(record)
        summary=dict(completed_physics_steps=600,seconds=rows[-1]['time_s'],finite_recorded_values=True,
                     minimum_root_height_m=min(r['root_pos'][2] for r in rows),
                     minimum_root_up_dot=min(r['root_up_dot'] for r in rows),
                     maximum_abs_joint_speed_rad_s=max(abs(v) for r in rows for v in r['qd']),
                     final_ground_supported_feet=rows[-1]['ground_supported_feet'],final=rows[-1],
                     evidence=metadata['evidence'])
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k!='final'}),flush=True)
    finally:
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__=='__main__':
    main()
