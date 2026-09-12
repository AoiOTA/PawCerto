"""Original automatic plan scaling and mixed M-controller arithmetic.

Adapted from pinned RoboDuet automatic environment (MIT/legged_gym BSD-3-Clause).
The mixed array's arm entries are position targets, not physical torque values.
The runtime must verify the engine drive behavior before applying these outputs.
"""
import torch


def plan(commands_dog, guidance, config):
    """Mutate dog pitch/roll commands and return the original unclipped plan_actions."""
    cfg = config['Cfg']
    if cfg['hybrid']['plan_vel'] or guidance.shape[-1] != 2:
        raise ValueError('Default RoboDuet guidance has exactly pitch and roll')
    rescaled = guidance * .4
    pitch, roll = cfg['commands']['limit_body_pitch'], cfg['commands']['limit_body_roll']
    commands_dog[:, 3] = torch.clip(rescaled[:, 0], pitch[0], pitch[1] / 4 * 3.)
    commands_dog[:, 4] = torch.clip(rescaled[:, 1], roll[0], roll[1])
    return rescaled


class RoboDuetController:
    """Name-mapped gains in the verified original simulator DOF order.

    Runtime supplies the measured order and all 20 URDF torque limits. This
    class does not assume XML order proves Gym import order. The default
    source indexes four consecutive hip/thigh/calf blocks and zarx_j1..8.
    """
    def __init__(self, config, joint_names, torque_limits, device='cpu'):
        self.cfg = config['Cfg']
        if self.cfg['control']['control_type'] != 'M':
            raise ValueError('This controller implements the effective automatic M branch')
        names = list(joint_names)
        legs = names[:12]
        valid_legs = (len(legs) == 12 and len(set(names)) == 20
                      and set(legs) == {f'{leg}_{part}_joint' for leg in ('FR', 'FL', 'RR', 'RL')
                                        for part in ('hip', 'thigh', 'calf')}
                      and all(legs[i:i+3] == [f'{legs[i].split("_")[0]}_{part}_joint'
                                               for part in ('hip', 'thigh', 'calf')]
                              for i in (0, 3, 6, 9)))
        if not valid_legs or names[12:] != [f'zarx_j{i}' for i in range(1, 9)]:
            raise ValueError('Measured DOF order does not match the original controller indexing')
        self.joint_names = names
        self.default_dof_pos = torch.tensor([[self.cfg['init_state']['default_joint_angles'][n] for n in names]],
                                            device=device)
        self.torque_limits = torch.as_tensor(torque_limits, dtype=torch.float32, device=device)
        if self.torque_limits.shape != (20,):
            raise ValueError('Expected 20 measured URDF effort limits')
        self.p_gains, self.d_gains = torch.zeros(20, device=device), torch.zeros(20, device=device)
        # Preserve original matching order, including the source's generic
        # zarx gains in this arithmetic branch. Runtime drive gains are separate.
        for i, name in enumerate(names[:18]):
            for key in self.cfg['control']['stiffness']:
                if key in name:
                    side, suffix = ('dog', 'leg') if i < 12 else ('arm', 'arm')
                    control = self.cfg[side]['control']
                    self.p_gains[i] = control['stiffness_' + suffix][key]
                    self.d_gains[i] = control['damping_' + suffix][key]
        self.decimation = self.cfg['control']['decimation']
        self.dt = self.cfg['sim']['dt']

    def compute(self, actions, dof_pos, dof_vel, *, kp_factors=1., kd_factors=1.,
                motor_offsets=0., motor_strengths=1.):
        """Return clipped actions, target, original combined20, leg effort and arm target.

        Recompute each physical substep from current q/qd. No lag buffer is
        inserted: the effective M branch does not consume lag_timesteps.
        """
        if actions.shape[-1] != 18 or dof_pos.shape[-1] != 20 or dof_vel.shape != dof_pos.shape:
            raise ValueError('Expected 18 policy actions and 20 simulator DOFs')
        clip = self.cfg['normalization']['clip_actions']
        actions = actions.clamp(-clip, clip)
        scaled = actions * self.cfg['control']['action_scale']
        scaled[:, [0, 3, 6, 9]] *= self.cfg['control']['hip_scale_reduction']
        target = torch.nn.functional.pad(scaled, (0, 2)) + self.default_dof_pos
        torque = self.p_gains * kp_factors * (target - dof_pos + motor_offsets) - self.d_gains * kd_factors * dof_vel
        torque = (torque * motor_strengths).clamp(-self.torque_limits, self.torque_limits)
        position = ((target + motor_offsets) * motor_strengths).clamp(-10., 10.)
        combined = torch.cat((torque[:, :12], position[:, 12:]), dim=-1)
        return {'actions': actions, 'joint_pos_target': target, 'combined': combined,
                'leg_effort': combined[:, :12], 'arm_position': combined[:, 12:]}
