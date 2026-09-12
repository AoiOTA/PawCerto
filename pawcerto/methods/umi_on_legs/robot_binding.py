"""Concrete UMI bindings for the released Go2/ARX5 and nominal AS2/Piper robots."""
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from pawcerto.robots.urdf import read_joint_limits
from pawcerto.artifacts import file_identity

ROOT = Path(__file__).resolve().parents[3]
FEET = [f'{leg}_foot' for leg in ('FR', 'FL', 'RR', 'RL')]


def robot_binding(config):
    binding = config.get('pawcerto_robot')
    if binding is not None:
        if binding['name'] != 'as2_piper':
            raise ValueError(f'Unknown UMI robot: {binding["name"]}')
        return binding
    return dict(name='go2_arx5', root_body='base', tcp_body='link6',
                tcp_xyz=[.22, 0., 0.], tcp_rotation=[[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]],
                joint_names=[f'{leg}_{part}_joint' for leg in ('FL', 'FR', 'RL', 'RR')
                             for part in ('hip', 'thigh', 'calf')] + [f'joint{i}' for i in range(1, 7)],
                feet=FEET, head_bodies=['Head_upper', 'Head_lower'],
                usd_path=str(ROOT/'reference/isaac/go2_arx5/usd_path.txt'),
                urdf_path=str(ROOT/'reference/isaac/go2_arx5/go2_arx5_merged.urdf'),
                mujoco_path=str(ROOT/'pawcerto/mujoco/go2_arx5.xml'))


def joint_order(config, path=None):
    expected = robot_binding(config)['joint_names']
    actual = expected if path is None else json.loads(Path(path).read_text())
    if actual != expected:
        raise ValueError('Joint order differs from the selected UMI robot binding')
    return actual


def require_same_robot(config, source_config):
    if robot_binding(config) != robot_binding(source_config):
        raise ValueError('Checkpoint robot binding differs; AS2 requires its own training, not Go2 weight reuse')


def as2_config(source):
    """Adapt physical inputs only; task, PPO, delays and reward formulas are retained."""
    if source.get('pawcerto_robot'):
        if robot_binding(source)['name'] != 'as2_piper':
            raise ValueError('Cannot adapt an unknown robot')
        return deepcopy(source)
    config = deepcopy(source)
    nominal_path = ROOT/'configs/as2_piper.json'
    nominal = json.loads(nominal_path.read_text())
    urdf = ROOT/'reference/as2_piper/robot.urdf'
    asset = ET.parse(urdf)
    names = nominal['controlled_joint_order']
    limits = read_joint_limits(urdf, names)
    collision_names = [link.get('name') for link in asset.findall('link') if link.find('collision') is not None]
    shape_count = sum(len(link.findall('collision')) for link in asset.findall('link'))
    # Nominal simulation choices, not identified actuator parameters.
    offset = nominal['default_joint_positions']
    kp = [60., 60., 60.]*4 + [80., 80., 60., 15., 10., 8.]
    kd = [1., 1., .25]*4 + [2., 2., 1., .03, .15, .02]
    binding = dict(name='as2_piper', root_body='base_link', tcp_body=nominal['tcp']['body'],
                   tcp_xyz=nominal['tcp']['xyz'], tcp_rotation=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                   joint_names=names, feet=FEET, head_bodies=[],
                   usd_path=str(ROOT/'reference/isaac/as2_piper/usd_path.txt'),
                   urdf_path=str(ROOT/'reference/isaac/as2_piper/merged.urdf'),
                   mujoco_path=str(ROOT/'reference/as2_piper/robot.xml'),
                   nominal_input=file_identity(nominal_path), source_urdf=file_identity(urdf),
                   simulation_choices='Nominal uncalibrated PD; source effort and velocity bounds; stock gripper fixed 40 mm; no hardware identification',
                   collision_shape_count=shape_count)
    config['pawcerto_robot'] = binding
    config.pop('pawcerto_asset', None)
    # This nominal adaptation selects source limits, including when its recipe
    # was copied from an earlier Go2 experiment with a 1000 rad/s override.
    config.pop('joint_velocity_limit_override_rad_s', None)
    config.pop('pawcerto_runtime', None)
    env = config['env']
    c = env['controller']
    for key, values in dict(offset=offset, kp=kp, kd=kd, torque_limit=[limits[n]['effort'] for n in names]).items():
        c[key]['data'] = values
    for group in ('state_obs', 'privileged_state_obs'):
        env[group]['dof_pos']['offset']['data'] = offset
    env['cfg']['init_state']['pos'] = [-.5, 0., nominal['default_base_xyz'][2]]
    env['cfg']['init_state']['rot'] = [0., 0., 0., 1.]
    cfg_asset = env['cfg']['asset']
    cfg_asset.update(name='as2_piper_nominal', file=binding['urdf_path'], collapse_fixed_joints=False,
                     terminate_after_contacts_on=['base_link', 'hip', 'thigh'] + [f'piper_link{i}' for i in range(1,6)])
    rand = env['cfg']['domain_rand']
    bodies = ['base_link', 'piper_link1', 'piper_link3', 'piper_link5', 'piper_link6']
    rand['randomize_rigid_body_masses'] = bodies
    rand['randomize_rigid_body_com'] = bodies
    constraints = env['constraints']
    constraints['joint_limit']['lower']['data'] = [limits[n]['lower'] for n in names]
    constraints['joint_limit']['upper']['data'] = [limits[n]['upper'] for n in names]
    constraints['collision']['link_names'] = [n for n in collision_names if n not in FEET and not n.startswith('piper_gripper') and n != 'piper_link6']
    constraints['root_height']['target_height'] = .30  # observed nominal supported stance; .55 m is release height
    constraints['aligned_body_ee']['joint_names'] = ['piper_joint1','piper_joint5']
    constraints['aligned_body_ee']['default_config']['data'] = [offset[12],offset[16]]
    # Keep inactive source config references descriptive and coherent too.
    mapping = {f'joint{i}':f'piper_joint{i}' for i in range(1,7)}
    mapping.update({f'link{i}':f'piper_link{i}' for i in range(1,7)})
    mapping['end_effector'] = nominal['tcp']['body']
    def rename(value):
        if isinstance(value, dict): return {k:rename(v) for k,v in value.items()}
        if isinstance(value, list): return [rename(v) for v in value]
        return mapping.get(value,value) if isinstance(value,str) else value
    env['constraints'] = rename(constraints)
    env['constraints']['close_to_default']['default_config']['data'] = offset
    env['tasks']['reaching']['link_name'] = nominal['tcp']['body']
    setup = env['privileged_setup_obs']
    delta = shape_count - setup['rigid_shape_friction']['dim']
    setup['rigid_shape_friction']['dim'] = shape_count
    ac = config['runner']['alg']['actor_critic']
    ac['num_critic_obs'] += delta
    ac['critic']['_args_'][0]['in_features'] += delta
    env['cfg']['env']['num_privileged_obs'] += delta
    return config
