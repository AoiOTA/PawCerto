"""Validate the stock torch FD diagnostic, then shadow one nominal 17 s rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT/'outputs/isaac/stock-solver-force-shadow')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
if (args.output/'summary.json').exists():
    raise FileExistsError('Preserve the existing diagnostic result; use a new output directory')
launcher = AppLauncher(args)
report = {'status': 'initializing', 'training': False, 'reward_or_observation_changed': False,
          'diagnostic_equivalence_boundary': 'Current stock solver residual, not original Gym forceSensor equivalence'}
raw = {}
rows = []
failure_frame = {}
try:
    import numpy as np
    import torch
    import warp as wp
    from pxr import UsdPhysics
    from isaaclab.utils.math import matrix_from_quat
    from pawcerto.isaac.solver_force import StockSolverForce
    from pawcerto.isaac.runtime import Go2Arx5Isaac
    from pawcerto.methods.umi_on_legs import UmiPolicy

    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('This bounded diagnostic requires the assigned CUDA device')

    def tensor(value):
        if hasattr(value, 'torch'):
            value = value.torch
        return value if isinstance(value, torch.Tensor) else wp.to_torch(value)

    def numpy(value):
        return value.detach().cpu().numpy().copy()

    def properties_and_model(source, static):
        names = source['body_names']
        indices = source['metatype']['link_parent_indices']
        parents = [-1] + [indices[name] for name in names[1:]]
        body_records = {row['path']: row for row in source['usd_physics']}
        paths = source['link_paths'][0]
        def field(name, radians=False):
            values = []
            for path in paths:
                row = body_records[path]
                if 'PhysxRigidBodyAPI' not in row['schemas'] or name not in row['attributes']:
                    raise ValueError(f'Missing composed body property {path}: {name}')
                values.append(row['attributes'][name])
            result = torch.tensor([values], dtype=torch.float32, device=device)
            return torch.deg2rad(result) if radians else result
        model = StockSolverForce(parents, static['get_masses'].to(device), dt=source['dt'],
            body_inertias=static['get_inertias'].to(device).reshape(-1, len(names), 3, 3),
            dof_max_velocity=static['get_dof_max_velocities'].to(device),
            body_max_linear_velocity=field('physxRigidBody:maxLinearVelocity'),
            body_max_angular_velocity=field('physxRigidBody:maxAngularVelocity', radians=True),
            body_linear_damping=field('physxRigidBody:linearDamping'),
            body_angular_damping=field('physxRigidBody:angularDamping'),
            armature=static['get_dof_armatures'].to(device),
            disable_gravity=static['get_disable_gravities'].to(device),
            retain_accelerations=field('physxRigidBody:retainAccelerations'),
            gyroscopic_forces=field('physxRigidBody:enableGyroscopicForces'),
            external_forces_every_iteration=source['simulation_config']['physics']['enable_external_forces_every_iteration'])
        return model, parents

    # First gate: CUDA float32 implementation against the independent float64
    # results and actual reported acceleration from all three saved FD samples.
    validation = []
    for sample in range(3):
        stem = ROOT/f'outputs/analysis/stock-solver-force-probe-free-dynamics-step{sample:02d}'
        reference = json.loads(stem.with_suffix('.json').read_text())
        with np.load(stem.with_suffix('.npz')) as saved:
            source = reference['summary_snapshot']
            static = {key[8:]: torch.as_tensor(saved[key], dtype=torch.float32, device=device)
                      for key in saved.files if key.startswith('static__')}
            model, _ = properties_and_model(source, static)
            pre = f'step{sample:02d}_pre__'
            get = lambda name: torch.as_tensor(saved[pre+name], dtype=torch.float32, device=device)
            wrench = torch.stack([
                get(composer+'_'+field) for composer in ('instantaneous_wrench_composer', 'permanent_wrench_composer')
                for field in ('global_force_w', 'global_torque_w', 'global_force_at_com_w', 'local_force_b', 'local_torque_b')])
            prediction, qdd, bias = model.predict(mass_matrix=get('get_generalized_mass_matrices'),
                coriolis=get('get_coriolis_and_centrifugal_compensation_forces'),
                gravity=get('get_gravity_compensation_forces'), jacobian=get('get_jacobians'),
                com_position=get('data_body_com_pose_w')[..., :3], com_velocity=get('get_link_velocities'),
                link_rotation=matrix_from_quat(get('get_link_transforms')[..., 3:]),
                dof_velocity=get('get_dof_velocities'), actuation=get('get_dof_actuation_forces'),
                external_wrench_terms=wrench)
            torch.cuda.synchronize(device)
            actual = numpy(prediction).astype(np.float64)
            failure_frame = {'gpu_prediction': actual, 'cpu_prediction': saved['predicted_link_acceleration'],
                             'reported': saved['reported_link_acceleration']}
            checks = {}
            for label, target in [('cpu_float64', saved['predicted_link_acceleration']),
                                  ('reported', saved['reported_link_acceleration'])]:
                error = np.abs(actual-target)
                tolerance = 1e-3+1e-4*np.abs(target)
                # The reported comparison uses prediction in the predeclared tolerance.
                if label == 'reported':
                    tolerance = 1e-3+1e-4*np.abs(actual)
                checks[label] = {'failed_elements': int(np.sum(error > tolerance)),
                    'linear_max_abs_error': float(error[..., :3].max()),
                    'angular_max_abs_error': float(error[..., 3:].max()),
                    'max_error_over_tolerance': float((error/tolerance).max())}
                if np.any(error > tolerance):
                    raise AssertionError(f'GPU saved-sample validation failed: {sample} {label} {checks[label]}')
            np.savez(args.output/f'gpu-validation-step{sample:02d}.npz', prediction=actual,
                     cpu_prediction=saved['predicted_link_acceleration'], reported=saved['reported_link_acceleration'],
                     qdd=numpy(qdd), bias=numpy(bias))
            validation.append({'sample': sample, 'dtype': str(prediction.dtype), 'device': str(prediction.device),
                               'checks': checks, 'source': str(stem)})
            failure_frame = {}
    (args.output/'gpu-validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print('GPU_SAVED_FD_VALIDATION_PASSED', flush=True)

    policy = UmiPolicy(ROOT/'reference/checkpoints/tossing/ours', device=args.device)
    torch.manual_seed(0)
    joint_names = json.loads((ROOT/'configs/umi_go2_arx5_joint_names.json').read_text())
    env = Go2Arx5Isaac(policy.config, joint_names, 1, args.device, training=False, ground_contact_diagnostics=True)
    env.reset()
    view = env.robot.root_view
    static_methods = ('get_masses', 'get_coms', 'get_inertias', 'get_disable_gravities',
                      'get_dof_armatures', 'get_dof_max_velocities', 'get_dof_limits',
                      'get_dof_stiffnesses', 'get_dof_dampings', 'get_dof_friction_coefficients',
                      'get_dof_friction_properties')
    static = {name: tensor(getattr(view, name)()).clone() for name in static_methods}
    usd = []
    for prim in env.sim.stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and path.startswith('/World/envs/'):
            attrs = {str(a.GetName()): a.Get() for a in prim.GetAttributes()
                     if str(a.GetName()).startswith('physxRigidBody:')}
            usd.append({'path': path, 'schemas': list(prim.GetAppliedSchemas()), 'attributes': attrs})
    source = {'dt': env.dt, 'body_names': env.body_names, 'joint_names': env.robot.joint_names,
              'gym_to_lab': env.joint_ids, 'link_paths': view.link_paths, 'dof_paths': view.dof_paths,
              'metatype': {'link_parent_indices': dict(view.shared_metatype.link_parent_indices)},
              'usd_physics': usd, 'simulation_config': env.sim.cfg.to_dict()}
    if view.shared_metatype.fixed_base:
        raise ValueError('Expected the original floating base')
    model, parents = properties_and_model(source, static)
    (args.output/'metadata.json').write_text(json.dumps(source, indent=2)+'\n')
    np.savez(args.output/'static.npz', **{name: numpy(value) for name, value in static.items()})
    foot_names = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
    foot_ids = [env.body_names.index(name) for name in foot_names]
    positions, rotations = policy.trajectories(ROOT/'reference/data/tossing.pkl').sample(1, 0)
    observer = policy.observer(positions, rotations)
    controller = policy.controller(1)
    physics_steps = round(17.0/env.dt)
    if physics_steps != 3400 or controller.decimation != 4:
        raise ValueError('The assigned nominal protocol is 3400 physics steps, decimation 4')
    report.update(status='running', dt=env.dt, nominal_seconds=17.0, physics_steps_requested=physics_steps,
                  sample=[1, 0], seed=0, foot_order=foot_names, warmup_physics_steps=4,
                  checkpoint=str(ROOT/'reference/checkpoints/tossing/ours/model.pt'),
                  gpu_validation=validation, dtype=str(model.masses.dtype),
                  force_definition='m*(reported post-step COM acceleration - pre-state predicted free COM acceleration), world linear xyz',
                  branch='body damping zero, external wrench zero, retain acceleration false, every TGS external force false, armature zero, no pre DOF/body clamp; source PRE gyroscopic clamp correction',
                  joint_friction_note='Joint friction and implicit drive forces belong to solver and are not subtracted from Qfree')

    dynamic = {'mass_matrix': 'get_generalized_mass_matrices',
               'gravity': 'get_gravity_compensation_forces',
               'coriolis': 'get_coriolis_and_centrifugal_compensation_forces',
               'jacobian': 'get_jacobians',
               'com_velocity': 'get_link_velocities', 'root_velocity': 'get_root_velocities',
               'dof_position': 'get_dof_positions', 'dof_velocity': 'get_dof_velocities',
               'actuation': 'get_dof_actuation_forces'}
    started = time.perf_counter()
    with torch.inference_mode(), (args.output/'rows.jsonl').open('w') as stream:
        for step in range(physics_steps):
            whole_start = time.perf_counter()
            substep = step % controller.decimation
            if step >= 4 and substep == 0:
                action = policy.actor(observer.observe(env.state(), controller.last_action))
                controller.push(action)
            q, qd = env.joints()
            command = controller.torque(q, qd, substep)
            wrench_terms = []
            for name in ('instantaneous_wrench_composer', 'permanent_wrench_composer'):
                composer = getattr(env.robot, name)
                if composer.active:
                    raise ValueError(f'Unsupported active {name} before write_data_to_sim')
                for field in ('global_force_w', 'global_torque_w', 'global_force_at_com_w', 'local_force_b', 'local_torque_b'):
                    wrench_terms.append(tensor(getattr(composer, field)).clone())
            wrench = torch.stack(wrench_terms)
            env.robot.set_joint_effort_target_index(target=command.contiguous(), joint_ids=env.joint_ids)
            env.scene.write_data_to_sim()
            # Refresh COM/actor kinematics before reading state-dependent engine buffers.
            com_pose = env.robot.data.body_com_pose_w.torch.clone()
            pre = {name: tensor(getattr(view, method)()).clone() for name, method in dynamic.items()}
            pre['com_position'] = com_pose[..., :3]
            pre['link_rotation'] = matrix_from_quat(env.robot.data.body_link_quat_w.torch).clone()
            pre['external_wrench_terms'] = wrench
            failure_frame = {name: numpy(value) for name, value in pre.items()}
            if not torch.equal(pre['actuation'][:, env.joint_ids], command):
                raise ValueError('Actual same-step torque differs from submitted command')
            sample_end = time.perf_counter()
            event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            prediction_start = time.perf_counter()
            event_start.record()
            predicted, qdd, bias = model.predict(**{key: pre[key] for key in (
                'mass_matrix', 'coriolis', 'gravity', 'jacobian', 'com_position',
                'com_velocity', 'link_rotation', 'dof_velocity', 'actuation', 'external_wrench_terms')})
            event_end.record()
            torch.cuda.synchronize(device)
            prediction_end = time.perf_counter()
            failure_frame.update(predicted=numpy(predicted), qdd=numpy(qdd), bias=numpy(bias))
            physics_start = time.perf_counter()
            env.sim.step(render=False)
            env.scene.update(env.dt)
            env.time += env.dt
            torch.cuda.synchronize(device)
            physics_end = time.perf_counter()
            reported = tensor(view.get_link_accelerations()).clone()
            failure_frame['reported'] = numpy(reported)
            residual = model.residual_force(reported, predicted)
            post_velocity = tensor(view.get_link_velocities()).clone()
            post_qd = tensor(view.get_dof_velocities()).clone()
            failure_frame.update(post_com_velocity=numpy(post_velocity), post_dof_velocity=numpy(post_qd))
            # Free-force clamps consumed PRE state; POST is checked for finiteness only.
            if not bool(torch.isfinite(post_velocity).all() and torch.isfinite(post_qd).all()):
                raise FloatingPointError('Nonfinite stock dynamics post-step velocity')
            linear_speed = torch.linalg.vector_norm(post_velocity[..., :3], dim=-1)
            angular_speed = torch.linalg.vector_norm(post_velocity[..., 3:], dim=-1)
            state = env.state()
            target_position, target_rotation = observer.targets(state.episode_time)
            position_error = (state.ee_pose[:, :3, 3]-target_position).norm(dim=-1)
            relative = state.ee_pose[:, :3, :3].transpose(-1, -2) @ target_rotation
            angle_error = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1).acos()
            normal = env.training_state()['feet_force_z']
            ground = env.feet_ground_force_z()
            nonfoot = torch.stack([env.scene['contact_'+name].data.net_forces_w.torch[:, 0].norm(dim=-1)
                                  for name in env.contact_names if name not in foot_names], dim=-1).amax(-1)
            post_q = tensor(view.get_dof_positions()).clone()
            limits = static['get_dof_limits'].to(device)
            margin = torch.minimum(post_q-limits[..., 0], limits[..., 1]-post_q).amin(-1)
            values = dict(predicted=predicted, reported=reported, residual=residual,
                          qdd=qdd, bias=bias, post_com_velocity=post_velocity, post_dof_velocity=post_qd,
                          post_dof_position=post_q, normal_foot_fz=normal, ground_foot_fz=ground,
                          ee_pose=state.ee_pose, target_position=target_position, target_rotation=target_rotation)
            captured = {name: numpy(value) for name, value in {**pre, **values}.items()}
            for name, array in captured.items():
                if not np.isfinite(array).all():
                    raise FloatingPointError(f'Nonfinite captured {name} at physics step {step}')
            for name, array in captured.items():
                raw.setdefault(name, []).append(array)
            row = {'physics_step': step, 'policy_period': step//4, 'substep': substep,
                   'time_s': (step+1)*env.dt, 'warmup': step < 4,
                   'shadow_foot_fz_N': numpy(residual[:, foot_ids, 2])[0].tolist(),
                   'normal_foot_fz_N': numpy(normal)[0].tolist(), 'ground_foot_fz_N': numpy(ground)[0].tolist(),
                   'position_error_m': position_error.item(), 'orientation_error_rad': angle_error.item(),
                   'root_height_m': env.robot.data.root_link_pos_w.torch[0, 2].item(),
                   'root_up_dot': (-state.local_root_gravity[0, 2]).item(),
                   'joint_limit_min_margin_rad': margin.item(),
                   'max_abs_dof_velocity_rad_s': post_qd.abs().max().item(),
                   'max_body_linear_speed_m_s': linear_speed.max().item(),
                   'max_body_angular_speed_rad_s': angular_speed.max().item(),
                   'nonfoot_normal_contact_max_N': nonfoot.item(), 'all_state_finite': True,
                   'pre_sampling_wall_ms': (sample_end-whole_start)*1000,
                   'prediction_wall_ms': (prediction_end-prediction_start)*1000,
                   'prediction_cuda_ms': event_start.elapsed_time(event_end),
                   'physics_wall_ms': (physics_end-physics_start)*1000,
                   'post_diagnostics_wall_ms': (time.perf_counter()-physics_end)*1000}
            row['whole_step_wall_ms'] = (time.perf_counter()-whole_start)*1000
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            failure_frame = {}
            if substep == 3:
                observer.advance_pose(state.ee_pose)
            if (step+1) % 200 == 0:
                print(json.dumps({'physics_steps': step+1, 'time_s': (step+1)*env.dt,
                                  'max_abs_shadow_foot_fz': float(residual[:, foot_ids, 2].abs().max())}), flush=True)
    report.update(status='completed', physics_steps_completed=len(rows), elapsed_seconds=time.perf_counter()-started,
                  loaded_physx_libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                               if 'libPhysX' in line}))
except BaseException as exc:
    report.update(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                  physics_steps_completed=len(rows))
    print(report['traceback'], flush=True)
    raise
finally:
    if raw:
        np.savez_compressed(args.output/'capture.npz', **{name: np.stack(values) for name, values in raw.items()})
    if failure_frame:
        np.savez(args.output/'failed-frame.npz', **failure_frame)
    report['module_sha256'] = hashlib.sha256((ROOT/'pawcerto/isaac/solver_force.py').read_bytes()).hexdigest()
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    launcher.app.close(exit_code=int(report['status'] != 'completed'))
