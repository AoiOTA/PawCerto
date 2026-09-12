"""Default no-vision RoboDuet observations and caller-driven histories.

Adapted from the pinned automatic/__init__.py, legged_robot.py and utils/common.py
(RoboDuet MIT, legged_gym BSD-3-Clause). Quaternions use xyzw throughout.
Commands, gait, physics and reset timing remain owned by the calling environment.
"""
import torch


def quat_mul(a, b):
    av, bv = a[..., :3], b[..., :3]
    xyz = a[..., 3:] * bv + b[..., 3:] * av + torch.cross(av, bv, dim=-1)
    w = a[..., 3:] * b[..., 3:] - (av * bv).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def quat_conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def quat_apply(q, vector):
    t = 2. * torch.cross(q[..., :3], vector, dim=-1)
    return vector + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


quat_rotate = quat_apply


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll * .5), torch.sin(roll * .5)
    cp, sp = torch.cos(pitch * .5), torch.sin(pitch * .5)
    cy, sy = torch.cos(yaw * .5), torch.sin(yaw * .5)
    return torch.stack((sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy), dim=-1)


def quaternion_to_rpy(q):
    x, y, z, w = q.unbind(-1)
    sinp = 2 * (w * y - z * x)
    return torch.stack((torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                        torch.where(sinp.abs() >= 1, torch.sign(sinp) * (torch.pi / 2),
                                    torch.asin(sinp)),
                        torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))), dim=-1)


def quat_to_angle(q):
    """Original projected-axis alpha/beta/gamma, which differs from Euler RPY."""
    basis = torch.eye(3, dtype=q.dtype, device=q.device)
    x = quat_apply(q, basis[0].expand(q.shape[0], -1))
    y = quat_apply(q, basis[1].expand(q.shape[0], -1))
    z = quat_apply(q, basis[2].expand(q.shape[0], -1))
    return torch.stack((torch.atan2(y[:, 2], y[:, 1]), torch.atan2(z[:, 0], z[:, 2]),
                        torch.atan2(x[:, 1], x[:, 0])), dim=-1)


def get_lpy_in_base_coord(state):
    """Original +0.1 m grasp point and terrain-relative z minus 0.38 m."""
    q, ee = state['base_quat'], state['end_effector_state']
    forward = torch.zeros_like(ee[:, :3])
    forward[:, 0] = 1.
    world_forward = quat_apply(q, forward)
    yaw = torch.atan2(world_forward[:, 1], world_forward[:, 0])
    grasp = ee[:, :3] + quat_apply(ee[:, 3:7], forward * .1)
    delta = grasp[:, :2] - state['root_states'][:, :2]
    x = torch.cos(yaw) * delta[:, 0] + torch.sin(yaw) * delta[:, 1]
    y = -torch.sin(yaw) * delta[:, 0] + torch.cos(yaw) * delta[:, 1]
    z = (grasp[:, 2:3] - state['measured_heights']).mean(dim=1) - .38
    return torch.stack((torch.sqrt(x.square() + y.square() + z.square()),
                        torch.atan2(z, torch.sqrt(x.square() + y.square())),
                        torch.atan2(y, x)), dim=-1)


def _privileged(state, cfg):
    # These are the only two enabled consumers in the effective default path.
    # Other enabled but unused config flags are not additional observations.
    chunks = []
    for name, key in (('friction', 'friction_coeffs'), ('restitution', 'restitutions')):
        if cfg['env']['priv_observe_' + name]:
            low, high = cfg['normalization'][name + '_range']
            chunks.append((state[key][:, :1] - (high + low) / 2.) * (2. / (high - low)))
    return torch.cat(chunks, dim=-1)


class RoboDuetObservations:
    """Mapping inputs use original env field names; every call appends one frame.

    Common: dof_pos/default_dof_pos/actions/base_quat/commands_arm_obs,
    friction_coeffs/restitutions. Dog additionally uses dof_vel,
    projected_gravity/commands_dog/clock_inputs. Arm additionally uses
    root_states/end_effector_state/measured_heights (world frame, xyzw).
    No observation noise is applied: it is commented out in both source methods.
    """
    def __init__(self, config, num_envs, device='cpu'):
        self.cfg = config['Cfg']
        cfg = self.cfg
        if cfg['use_rot6d'] or cfg['hybrid']['use_vision'] or cfg['hybrid']['plan_vel']:
            raise ValueError('This implementation serves default no-vision, no-rot6d Go1 automatic training')
        env = cfg['env']
        unsupported = ('observe_two_prev_actions', 'observe_timing_parameter', 'observe_vel',
                       'observe_only_ang_vel', 'observe_only_lin_vel', 'observe_yaw', 'observe_contact_states',
                       'priv_observe_ground_friction', 'priv_observe_base_mass', 'priv_observe_com_displacement',
                       'priv_observe_motor_strength', 'priv_observe_motor_offset', 'priv_observe_body_height',
                       'priv_observe_body_velocity', 'priv_observe_gravity', 'priv_observe_clock_inputs',
                       'priv_observe_desired_contact_states', 'priv_observe_vel', 'priv_observe_high_freq_goal')
        active = [name for name in unsupported if env.get(name, False)]
        if active or not env['observe_clock_inputs']:
            raise ValueError(f'Nondefault observation configuration is not implemented: {active}')
        if not env['priv_observe_friction'] or not env['priv_observe_restitution']:
            raise ValueError('Default adaptation targets require friction and restitution')
        if (cfg['dog']['dog_num_observations'], cfg['arm']['arm_num_observations'],
            cfg['dog']['dog_num_privileged_obs'], cfg['arm']['arm_num_privileged_obs']) != (56, 20, 2, 9):
            raise ValueError('The default observation assembly requires dog 56/2 and arm 20/9')
        self.dog_obs_history = torch.zeros(num_envs, cfg['dog']['dog_num_obs_history'], device=device)
        self.arm_obs_history = torch.zeros(num_envs, cfg['arm']['arm_num_obs_history'], device=device)

    def clear(self, ids=None):
        index = slice(None) if ids is None else ids
        self.dog_obs_history[index] = 0
        self.arm_obs_history[index] = 0

    def _append(self, side, obs, privileged):
        dims = self.cfg[side]
        expected = dims[side + '_num_observations']
        if obs.shape[1] != expected or privileged.shape[1] != dims[side + '_num_privileged_obs']:
            raise ValueError(f'{side} observation dimensions differ from configuration')
        limit = self.cfg['normalization']['clip_observations']
        obs, privileged = obs.clamp(-limit, limit), privileged.clamp(-limit, limit)
        name = side + '_obs_history'
        history = torch.cat((getattr(self, name)[:, expected:], obs), dim=-1)
        setattr(self, name, history)
        return {'obs': obs, 'privileged_obs': privileged, 'obs_history': history}

    def arm(self, state):
        scales = self.cfg['obs_scales']
        rpy = quaternion_to_rpy(state['base_quat'])
        obs = torch.cat(((state['dof_pos'][:, 12:18] - state['default_dof_pos'][:, 12:18]) * scales['dof_pos'],
                         state['actions'][:, 12:18], state['commands_arm_obs'][:, :6], rpy[:, :2]), dim=-1)
        yaw = rpy[:, 2]
        yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        # Preserve source's positive-yaw multiplication (not inverse), even
        # though its local variable is named quat_ee_in_base.
        ee_quat = quat_mul(yaw_quat, state['end_effector_state'][:, 3:7])
        privileged = torch.cat((_privileged(state, self.cfg), get_lpy_in_base_coord(state), ee_quat), dim=-1)
        return self._append('arm', obs, privileged)

    def dog(self, state, stage2):
        scales = self.cfg['obs_scales']
        command_scale = state['commands_dog'].new_tensor([scales['lin_vel'], scales['lin_vel'],
                                                        scales['ang_vel'], scales['body_pitch_cmd'],
                                                        scales['body_roll_cmd']])
        arm_commands = state['commands_arm_obs'][:, :6]
        if not stage2:
            arm_commands = torch.zeros_like(arm_commands)
        obs = torch.cat((state['projected_gravity'],
                         (state['dof_pos'][:, :12] - state['default_dof_pos'][:, :12]) * scales['dof_pos'],
                         state['dof_vel'][:, :12] * scales['dof_vel'], state['actions'][:, :12],
                         state['commands_dog'][:, :5] * command_scale, arm_commands,
                         quaternion_to_rpy(state['base_quat'])[:, :2], state['clock_inputs']), dim=-1)
        return self._append('dog', obs, _privileged(state, self.cfg))
