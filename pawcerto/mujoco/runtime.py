"""CPU physics and reference PD path; no dependency on Isaac or PawWeaver."""
import json
from pathlib import Path
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
from .asset import DEFAULT_MODEL, ROOT
from pawcerto.methods.umi_on_legs.robot_binding import robot_binding
from pawcerto.methods.umi_on_legs.actuation import actuation_mode, AS2_NATIVE_SERVO, piper_target_limits
from pawcerto.methods.umi_on_legs.training.semantics import runtime_contract

JOINT_NAMES = tuple(f'{leg}_{part}_joint' for leg in ('FL', 'FR', 'RL', 'RR')
                    for part in ('hip', 'thigh', 'calf')) + tuple(f'joint{i}' for i in range(1, 7))


NUMERICAL_WARNINGS = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                      mujoco.mjtWarning.mjWARN_BADQACC)


class SimulationInstability(FloatingPointError):
    def __init__(self, time_before, time_after, warnings):
        self.details = {'kind': 'mujoco_numerical_failure', 'time_before_step_s': time_before,
                        'engine_time_after_step_s': time_after, 'new_warnings': warnings,
                        'engine_time_restarted': time_after < time_before}
        super().__init__('MuJoCo numerical failure: ' + json.dumps(self.details))


class Go2Arx5Mujoco:
    def __init__(self, config_path=ROOT / 'reference/checkpoints/tossing/ours/config.json', model_path=None):
        config = config_path if isinstance(config_path, dict) else json.loads(Path(config_path).read_text())
        self.binding = robot_binding(config)
        self.actuation_mode = actuation_mode(config)
        self.native_servo = self.actuation_mode == AS2_NATIVE_SERVO
        self.cfg = config['env']
        model_path = model_path or self.binding['mujoco_path']
        if self.native_servo:
            self.runtime_metadata = runtime_contract('normal-contact', config.get('joint_velocity_limit_override_rad_s'), self.actuation_mode)
            if config.get('pawcerto_runtime', self.runtime_metadata) != self.runtime_metadata:
                raise ValueError('Native MuJoCo execution requires the saved normal-contact servo contract')
            if not hasattr(mujoco.mjtIntegrator, 'mjINT_DISCRETE'):
                raise ValueError('AS2 native servo requires MuJoCo with the discrete integrator (3.13+)')
            tree = ET.parse(model_path)
            root = tree.getroot()
            # AS2 assets are a single source XML. Keep relative mesh paths anchored to that source.
            if root.find('include') is not None:
                raise ValueError('Native AS2 requires the existing single-file robot XML')
            for element in root.iter():
                if element.get('file') is not None:
                    element.set('file', str((Path(model_path).resolve().parent / element.get('file')).resolve()))
            option = root.find('option')
            if option is None:
                option = ET.SubElement(root, 'option')
            option.set('integrator', 'discrete'); option.set('solver', 'Newton'); option.set('noslip_iterations', '0')
            old_actuators = root.find('actuator')
            if old_actuators is not None:
                root.remove(old_actuators)
            actuators = ET.SubElement(root, 'actuator')
            controller = self.cfg['controller']
            for i, name in enumerate(self.binding['joint_names']):
                limit = controller['torque_limit']['data'][i]
                ET.SubElement(actuators, 'position', name='servo_'+name, joint=name,
                              kp=str(controller['kp']['data'][i]), kv=str(controller['kd']['data'][i]),
                              forcelimited='true', forcerange=f'{-limit} {limit}', ctrllimited='false')
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        else:
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = self.cfg['cfg']['sim']['dt']
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(self.binding['joint_names'])
        joints = [self.model.joint(name) for name in self.joint_names]
        self.qadr = np.array([int(j.qposadr[0]) for j in joints])
        self.vadr = np.array([int(j.dofadr[0]) for j in joints])
        c = self.cfg['controller']
        for name in ('offset', 'scale', 'kp', 'kd', 'torque_limit'):
            setattr(self, name, np.asarray(c[name]['data'], dtype=float))
        if self.native_servo:
            self.piper_limits = np.asarray(piper_target_limits(config))
            # Existing source converter writes six significant digits; retain its
            # native limits while clipping references to the full URDF precision.
            np.testing.assert_allclose(self.model.jnt_range[[j.id for j in joints[12:]]], self.piper_limits.T, rtol=0, atol=1e-5)
            np.testing.assert_array_equal(self.model.jnt_actfrcrange[[j.id for j in joints]], np.stack([-self.torque_limit, self.torque_limit], axis=1))
            np.testing.assert_array_equal(self.model.actuator_forcerange, np.stack([-self.torque_limit, self.torque_limit], axis=1))
            if not (self.model.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE
                    and self.model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
                    and self.model.opt.noslip_iterations == 0):
                raise ValueError('Native MuJoCo solver readback differs from the explicit candidate')
        self.decimation = c['decimation_count']
        self.delay = np.rint(np.asarray(self.cfg['ctrl_delay']['data']) / self.model.opt.timestep).astype(int)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        init = self.cfg['cfg']['init_state']
        self.data.qpos[:3] = init['pos']
        self.data.qpos[3:7] = np.asarray(init['rot'])[[3, 0, 1, 2]]
        self.data.qpos[self.qadr] = self.offset
        self.data.qvel[:3] = init['lin_vel']
        self.data.qvel[3:6] = init['ang_vel']
        self.previous_action = np.zeros(18)
        self.last_torque = np.zeros(18)
        if self.native_servo:
            self.data.ctrl[:] = self.offset
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        result = {'joint_pos': self.data.qpos[self.qadr].copy(),
                  'joint_vel': self.data.qvel[self.vadr].copy(), 'time': float(self.data.time)}
        for prefix, name in [('root', self.binding['root_body']), ('ee', 'end_effector' if self.binding['name'] == 'go2_arx5' else self.binding['tcp_body'])]:
            body = self.data.body(name)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                    self.model.body(name).id, velocity, 0)
            result[prefix + '_pos'] = body.xpos.copy()
            result[prefix + '_quat_xyzw'] = body.xquat[[1, 2, 3, 0]].copy()
            result[prefix + '_rotmat'] = body.xmat.reshape(3, 3).copy()
            result[prefix + '_ang_vel_world'] = velocity[:3].copy()
            # mj_objectVelocity BODY returns velocity at the inertial center.
            # Our state pose is the link origin, including massless EE links.
            result[prefix + '_lin_vel_world'] = velocity[3:] + np.cross(velocity[:3], body.xpos - body.xipos)
        if self.binding['name'] == 'as2_piper':
            displacement = result['ee_rotmat'] @ np.asarray(self.binding['tcp_xyz'])
            result['ee_pos'] += displacement
            result['ee_lin_vel_world'] += np.cross(result['ee_ang_vel_world'], displacement)
        result['root_ang_vel_body'] = result['root_rotmat'].T @ result['root_ang_vel_world']
        result['gravity_body'] = result['root_rotmat'].T @ np.array([0., 0., -1.])
        return result

    def step(self, action, *, trace=None):
        """Advance one policy step; optional callback reads each existing solve."""
        action = np.asarray(action, dtype=float)
        if action.shape != (18,) or not np.all(np.isfinite(action)):
            raise ValueError('Policy action must contain 18 finite values')
        raw_action = action.copy() if trace is not None else None
        action = np.clip(action, -self.cfg['max_action_value'], self.cfg['max_action_value'])
        for substep in range(self.decimation):
            pre_q = self.data.qpos.copy() if trace is not None else None
            pre_qd = self.data.qvel.copy() if trace is not None else None
            delayed = np.where(substep < self.delay, self.previous_action, action)
            target = self.offset + self.scale * delayed
            target_before_joint_clip = target.copy() if trace is not None else None
            if self.native_servo:
                target[12:] = np.clip(target[12:], *self.piper_limits)
            torque = self.kp * (target - self.data.qpos[self.qadr]) - self.kd * self.data.qvel[self.vadr]
            self.last_torque = np.clip(torque, -self.torque_limit, self.torque_limit)
            if self.native_servo:
                self.data.ctrl[:] = target
                self.data.qfrc_applied[self.vadr] = 0.
            else:
                self.data.qfrc_applied[self.vadr] = self.last_torque
            time_before = float(self.data.time)
            warning_before = {w: self.data.warning[w].number for w in NUMERICAL_WARNINGS}
            mujoco.mj_step(self.model, self.data)
            warnings = {w.name: {'count': int(self.data.warning[w].number),
                                  'lastinfo': int(self.data.warning[w].lastinfo)}
                        for w in NUMERICAL_WARNINGS if self.data.warning[w].number > warning_before[w]}
            if warnings or self.data.time < time_before:
                # MuJoCo can auto-reset to finite qpos after BADQACC. Checking
                # only qpos at the end of a policy step hides this real failure.
                raise SimulationInstability(time_before, float(self.data.time), warnings)
            if trace is not None:
                from .contacts import contact_snapshot
                foot_z, ground_z, contacts = contact_snapshot(self)
                sample = {'time_before_s': time_before, 'time_after_s': float(self.data.time),
                       'substep': substep, 'raw_action': raw_action.copy(),
                       'clipped_action': action.copy(), 'previous_action': self.previous_action.copy(),
                       'executed_action': delayed.copy(), 'joint_target': target.copy(),
                       'torque_before_limit': torque.copy(), 'torque_applied': self.last_torque.copy(),
                       'qpos_before': pre_q, 'qvel_before': pre_qd,
                       'qpos_after': self.data.qpos.copy(), 'qvel_after': self.data.qvel.copy(),
                       'solver_state': self.state(), 'foot_net_z_N': foot_z,
                       'foot_ground_z_N': ground_z, 'contacts': contacts}
                if self.native_servo:
                    sample['joint_target_before_source_clip'] = target_before_joint_clip
                    sample['servo_effort_prestate_estimate'] = sample.pop('torque_applied')
                    sample['servo_effort_prestate_before_limit'] = sample.pop('torque_before_limit')
                    sample['native_actuator_force'] = self.data.actuator_force.copy()
                    sample['external_effort'] = self.data.qfrc_applied[self.vadr].copy()
                trace(sample)
        self.previous_action = action.copy()
        mujoco.mj_forward(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)):
            raise FloatingPointError('MuJoCo produced nonfinite positions')
        return self.state()
