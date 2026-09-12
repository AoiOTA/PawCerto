"""Run the released UMI actor on the matching Go2+ARX5 PhysX articulation."""
import argparse
import json
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, default=ROOT / 'reference/checkpoints/tossing/ours')
parser.add_argument('--compare-checkpoint', type=Path, action='append', default=[])
parser.add_argument('--trajectory', type=Path, default=ROOT / 'reference/data/tossing.pkl')
parser.add_argument('--joint-names', type=Path, required=True, help='JSON list from the original Gym asset query')
parser.add_argument('--num-envs', type=int, default=4)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--domain-randomization', action='store_true', help='Paired original physical/reset/PD/noise/push/transport distribution')
parser.add_argument('--contact-trace', type=Path, help='Optional 5 ms existing sensor readout JSONL; no physics changes')
parser.add_argument('--trace-case', type=int, default=6)
parser.add_argument('--output', type=Path, default=ROOT / 'reference/isaac/rollout.json')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
trace_stream = None
try:
    import torch
    from pawcerto.methods.umi_on_legs import UmiPolicy
    from pawcerto.isaac.runtime import Go2Arx5Isaac
    policy = UmiPolicy(args.checkpoint, device=args.device)
    torch.manual_seed(args.seed)
    env = Go2Arx5Isaac(policy.config, json.loads(args.joint_names.read_text()), args.num_envs, args.device,
                      training=args.domain_randomization, ground_contact_diagnostics=True)
    print('LAB_JOINT_NAMES', env.robot.joint_names, flush=True)
    print('LAB_BODY_NAMES', env.robot.body_names, flush=True)
    print('LAB_BODY_MASSES', env.robot.data.body_mass.torch[0].tolist(), flush=True)
    print('LAB_PHYSX_SHAPES', env.robot.root_view.max_shapes, flush=True)
    reports = []
    if args.contact_trace is not None:
        if not 0 <= args.trace_case < args.num_envs:
            raise ValueError('trace-case is outside the nominal batch')
        args.contact_trace.parent.mkdir(parents=True, exist_ok=True)
        trace_stream = args.contact_trace.open('x')

    for checkpoint in [args.checkpoint, *args.compare_checkpoint]:
        policy = UmiPolicy(checkpoint, device=args.device)
        torch.manual_seed(args.seed)
        env.reset()
        pos, rot = policy.trajectories(args.trajectory).sample(args.num_envs, args.seed)
        observer = policy.observer(pos, rot)
        controller = policy.controller(args.num_envs)
        noise_generator = None
        if args.domain_randomization:
            rand = policy.config['env']['cfg']['domain_rand']
            for gain in ('kp', 'kd'):
                low, high = rand[gain + '_ratio_range']
                value = getattr(controller, gain)
                setattr(controller, gain, value * (torch.rand((args.num_envs, 18), device=args.device) * (high-low) + low))
            noise_generator = torch.Generator(device=args.device).manual_seed(args.seed)
        rows = []
        def physics_step(substep, policy_step, raw_action):
            q, qd = env.joints()
            torque = controller.torque(q, qd, substep)
            if trace_stream is not None:
                i = args.trace_case
                indices = torch.ceil((controller.delay_steps - substep) / controller.decimation).long()
                executed = controller.buffer.permute(2, 1, 0)[torch.arange(len(indices), device=args.device), indices].T
                target = controller.offset + controller.scale * executed
                row = {'checkpoint': str(checkpoint), 'case': i, 'policy_step': policy_step,
                       'substep': substep, 'time_before_s': float(env.time[i]),
                       'q_before': q[i].tolist(), 'qd_before': qd[i].tolist(),
                       'raw_action': raw_action[i].tolist(), 'clipped_action': controller.last_action[i].tolist(),
                       'executed_action': executed[i].tolist(), 'joint_target': target[i].tolist(),
                       'torque_before_limit': (controller.kp * (target - q) - controller.kd * qd)[i].tolist(),
                       'torque_applied': torque[i].tolist()}
            env.step_torque(torque)
            if trace_stream is not None:
                readout = env.training_state()
                state = env.state()
                row.update({'time_after_s': float(env.time[i]), 'body_names': env.body_names,
                            'sensor_phase': 'ContactSensor net normal forces after existing simulation and scene update; body totals, not geom pairs or full friction force',
                            'contact_force_world_N': readout['contact_forces'][i].tolist(),
                            'foot_ground_z_N': env.feet_ground_force_z()[i].tolist(),
                            'q_after': state.dof_pos[i].tolist(), 'qd_after': state.dof_vel[i].tolist(),
                            'root_pos': readout['root_pos'][i].tolist(),
                            'up_dot': float(-state.local_root_gravity[i, 2]),
                            'ee_pose': state.ee_pose[i].tolist()})
                trace_stream.write(json.dumps(row) + '\n')

        with torch.inference_mode():
            # Original Env.reset() returns step(zeros): one full policy period.
            for substep in range(controller.decimation):
                physics_step(substep, 0, torch.zeros((args.num_envs, 18), device=args.device))
            observer.advance_pose(env.state().ee_pose)
            for step in range(args.steps - 1):
                state = env.state()
                action = policy.actor(observer.observe(state, controller.last_action, noise_generator))
                controller.push(action)
                for substep in range(controller.decimation):
                    physics_step(substep, step + 1, action)
                state = env.state()
                observer.advance_pose(state.ee_pose)
                target_pos, target_rot = observer.targets(state.episode_time)
                error = (state.ee_pose[:, :3, 3] - target_pos).norm(dim=-1)
                relative = state.ee_pose[:, :3, :3].transpose(-1, -2) @ target_rot
                angle = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos()
                height = env.robot.data.root_link_pos_w.torch[:, 2].cpu()
                feet_force = env.training_state()['feet_force_z']
                if not torch.isfinite(state.dof_pos).all():
                    raise FloatingPointError('Nonfinite physics joint state')
                rows.append({'step': step, 'ee_error_m': error.tolist(), 'ee_error_rad': angle.tolist(),
                             'root_height_m': height.tolist(), 'up_dot': (-state.local_root_gravity[:, 2]).tolist()})
                rows[-1]['supported_feet'] = (feet_force > 1).sum(-1).tolist()
                ground_force = env.feet_ground_force_z()
                rows[-1]['feet_ground_force_z'] = ground_force.tolist()
                rows[-1]['ground_supported_feet'] = (ground_force > 1).sum(-1).tolist()
                if args.domain_randomization:
                    env.apply_domain_randomization(step + 2)
        report = {'engine': 'Isaac Lab PhysX', 'checkpoint': str(checkpoint),
                  'joint_names': env.joint_names, 'num_envs': args.num_envs, 'steps': args.steps,
                  'physics_dt': env.dt, 'learning_performed_in_this_run': False, 'seed': args.seed,
                  'domain_randomization': args.domain_randomization, 'trajectory': str(args.trajectory),
                  'foot_order': ['FR', 'FL', 'RR', 'RL'],
                  'support_definitions': {'supported_feet': 'foot net normal Fz > 1 N, including self-contact',
                      'ground_supported_feet': 'foot normal Fz against /World/Ground/geometry/mesh > 1 N'},
                  'rows': rows}
        if trace_stream is not None:
            import warp as wp
            report['contact_trace'] = {'case': args.trace_case, 'path': str(args.contact_trace),
                'sample_period_s': env.dt,
                'native_joint_velocity_limits_rad_s': wp.to_torch(env.robot.root_view.get_dof_max_velocities()).cpu().tolist()}
        reports.append(report)
    if trace_stream is not None:
        trace_stream.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports if args.compare_checkpoint else reports[0], indent=2))
    print('UMI_ISAAC_RESULT', args.output, flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    if trace_stream is not None:
        trace_stream.close()
    import sys
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
