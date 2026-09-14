"""Explicit AS2 native-servo candidate; the released external PD stays default."""
import math
from .robot_binding import robot_binding

EXTERNAL_PD = 'external-pd'
AS2_NATIVE_SERVO = 'as2-native-servo'
SERVO_EFFORT_ESTIMATE = 'servo_effort_prestate_estimate'


def actuation_mode(config):
    mode = config.get('actuation_mode', EXTERNAL_PD)
    if mode not in (EXTERNAL_PD, AS2_NATIVE_SERVO):
        raise ValueError(f'Unsupported UMI actuation_mode: {mode}')
    if mode == AS2_NATIVE_SERVO and robot_binding(config)['name'] != 'as2_piper':
        raise ValueError('as2-native-servo requires the explicit AS2/Piper robot binding')
    return mode


def piper_target_limits(config):
    """Saved source bounds, verified against the actual asset by each runtime."""
    if actuation_mode(config) != AS2_NATIVE_SERVO:
        return None
    names = robot_binding(config)['joint_names']
    if names[12:] != [f'piper_joint{i}' for i in range(1, 7)]:
        raise ValueError('Native servo requires the source six-Piper-joint order')
    joint = config['env']['constraints']['joint_limit']
    lower, upper = (joint[k]['data'][12:] for k in ('lower', 'upper'))
    if len(lower) != 6 or len(upper) != 6 or any(
            not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi
            for lo, hi in zip(lower, upper)):
        raise ValueError('Native servo requires six finite ordered source Piper position bounds')
    return lower, upper


def require_same_actuation(config, source_config):
    """Execution/resume may not reinterpret a checkpoint's control semantics."""
    mode = actuation_mode(config)
    if mode != actuation_mode(source_config):
        raise ValueError('Checkpoint actuation mode differs; use explicit weight initialization for a new run')
    if mode == AS2_NATIVE_SERVO:
        keys = ('offset', 'scale', 'kp', 'kd', 'torque_limit')
        def contract(cfg):
            env = cfg['env']
            return (piper_target_limits(cfg), [env['controller'][k]['data'] for k in keys],
                    env['ctrl_delay']['data'], env['controller']['decimation_count'],
                    env['cfg']['sim']['dt'], env['max_action_value'],
                    cfg.get('joint_velocity_limit_override_rad_s'), cfg.get('pawcerto_runtime'))
        if contract(config) != contract(source_config):
            raise ValueError('Checkpoint native-servo control/reward contract differs')
