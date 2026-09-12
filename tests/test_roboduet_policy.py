"""CPU reference comparisons against pinned source, without importing Isaac Gym.

The reference loader executes unchanged selected class/function bodies. Only
declarative params-proto bases and simulator imports are excluded. Quaternion
operations in reference observation functions use independent SciPy rotations.
"""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import pytest
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.distributions import Normal

from pawcerto.methods.roboduet import (SOURCE_REVISION, RoboDuetController, RoboDuetObservations,
                                     build_models, coordinated_actions, default_config, plan)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'third_party/roboduet-reference'


class DeclarativeConfig:
    def __init_subclass__(cls, **kwargs):
        pass


def _ast(path):
    location = SOURCE / path
    if not location.exists():
        pytest.skip('Run python scripts/fetch_roboduet.py for pinned upstream comparisons')
    tree = json.loads((SOURCE / '.source-tree.json').read_text())
    assert tree['sha'] == SOURCE_REVISION and not tree.get('truncated')
    entry = next(item for item in tree['tree'] if item['path'] == path)
    content = location.read_bytes()
    assert hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == entry['sha']
    return ast.parse(content, filename=str(location))


def _execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<unchanged pinned RoboDuet bodies>', 'exec'), namespace)


def _definitions(path, namespace, names=None):
    nodes = [node for node in _ast(path).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and (names is None or node.name in names)]
    _execute(nodes, namespace)


def _method(path, class_name, method_name, namespace):
    owner = next(n for n in _ast(path).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    _execute([method], namespace)
    return namespace[method_name]


def _plain(value):
    if isinstance(value, type):
        return {k: _plain(v) for k, v in vars(value).items()
                if not k.startswith('_') and (not callable(v) or isinstance(v, type))}
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    return value


def _reference_config():
    ns = dict(torch=torch, nn=nn, Normal=Normal, PrefixProto=DeclarativeConfig,
              ParamsProto=DeclarativeConfig, Union=Union, Meta=type)
    _definitions('go1_gym/envs/automatic/legged_robot_config.py', ns)
    for name in ('go1_config', 'wtw_config', 'asset_config'):
        _definitions(f'go1_gym/envs/go1/{name}.py', ns)
    for name in ('dog_ac.py', 'arm_ac.py', 'ppo.py', '__init__.py'):
        nodes = [n for n in _ast('go1_gym_learn/ppo_cse_automatic/' + name).body
                 if isinstance(n, ast.ClassDef) and n.name.endswith(('Args', '_Args'))]
        _execute(nodes, ns)
    ns['args'] = SimpleNamespace(debug=False, offline=True, no_wandb=True, resume=False,
                                 wo_two_stage=False, num_envs=2048, use_rot6d=False)
    ns['global_switch'] = SimpleNamespace()
    function = next(n for n in _ast('scripts/auto_train.py').body if isinstance(n, ast.FunctionDef))
    start = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Expr)
                 and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                 and n.value.func.id == 'config_go1')
    end = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == 'init_sigmoid_lr')
    _execute(function.body[start:end], ns)
    return ns


@pytest.fixture(scope='module', autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_effective_config_matches_original_entrypoint():
    ns = _reference_config()
    config = default_config()
    # The only added default field is the measured invalid-mass source fix;
    # all original source values, including the randomization range, stay equal.
    source_comparison = deepcopy(config)
    assert source_comparison['Cfg']['domain_rand'].pop('base_mass_body') == 'trunk'
    for name in ('Cfg', 'DogAC_Args', 'ArmAC_Args', 'PPO_Args', 'RunnerArgs'):
        assert source_comparison[name] == _plain(ns[name])
    assert config['stage'] == vars(ns['global_switch'])
    assert json.loads(json.dumps(config, allow_nan=False)) == config
    config['Cfg']['dog']['dog_num_observations'] = -1
    assert default_config()['Cfg']['dog']['dog_num_observations'] == 56


def _reference_models():
    ns = _reference_config()
    for file in ('dog_ac.py', 'arm_ac.py'):
        _definitions('go1_gym_learn/ppo_cse_automatic/' + file, ns)
    validation_method = Normal.set_default_validate_args
    with redirect_stdout(io.StringIO()):
        dog = ns['DogActorCritic'](56, 2, 1680, 12)
        arm = ns['ArmActorCritic'](20, 9, 600, 8)
    # The source assigns this class method to False; avoid leaking that side
    # effect into unrelated tests while retaining the constructed networks.
    Normal.set_default_validate_args = validation_method
    return ns, dog, arm


@pytest.mark.parametrize('side,history_dim,priv_dim', [('dog', 1680, 2), ('arm', 600, 9)])
def test_same_weight_training_distribution_critic_and_gradients(side, history_dim, priv_dim):
    torch.manual_seed(37)
    _, source_dog, source_arm = _reference_models()
    dog, arm = build_models()
    source, port = (source_dog, dog) if side == 'dog' else (source_arm, arm)
    port.load_state_dict(source.state_dict(), strict=True)
    history, privileged = torch.randn(5, history_dim), torch.randn(5, priv_dim)
    for model in (source, port):
        model.update_distribution(history)
    torch.testing.assert_close(port.action_mean, source.action_mean, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(port.action_std, source.action_std, rtol=0, atol=0)
    for model in (source, port):
        loss = model.action_mean.square().mean() + model.evaluate(history, privileged).square().mean()
        loss.backward()
    torch.testing.assert_close(port.evaluate(history, privileged), source.evaluate(history, privileged),
                               rtol=1e-6, atol=1e-6)
    for (a_name, a), (b_name, b) in zip(source.named_parameters(), port.named_parameters()):
        assert a_name == b_name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)
    # Source samples after squashing the mean; samples themselves are not tanh.
    torch.manual_seed(91)
    expected = source.act(history)
    torch.manual_seed(91)
    actual = port.act(history)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_actual_official_playback_closures_and_training_mean_difference(tmp_path):
    ns, source_dog, source_arm = _reference_models()
    dog, arm = build_models()
    with torch.no_grad():
        source_arm.actor_body[-1].weight[-2:] = 0
        source_arm.actor_body[-1].bias[-2:] = torch.tensor([2., -2.])
    for name, model, port in [('dog', source_dog, dog), ('arm', source_arm, arm)]:
        directory = tmp_path / ('checkpoints_' + name)
        directory.mkdir()
        torch.save(model.state_dict(), directory / f'ac_weights_last_{name}.pt')
        port.load_state_dict(model.state_dict(), strict=True)
    _definitions('scripts/load_policy.py', ns, ['load_dog_policy', 'load_arm_policy'])
    validation_method = Normal.set_default_validate_args
    with redirect_stdout(io.StringIO()):
        dog_play = ns['load_dog_policy'](str(tmp_path), 'last', ns['Cfg'])
        arm_play = ns['load_arm_policy'](str(tmp_path), 'last', ns['Cfg'])
    Normal.set_default_validate_args = validation_method
    # Independent latest obs deliberately differs from history's final frame.
    dog_obs = {'obs': torch.randn(3, 56), 'obs_history': torch.randn(3, 1680)}
    arm_obs = {'obs': torch.randn(3, 20), 'obs_history': torch.randn(3, 600)}
    torch.testing.assert_close(dog.act_inference(dog_obs), dog_play(dog_obs), rtol=1e-6, atol=1e-6)
    raw = arm.act_inference(arm_obs, mode='official_play')
    torch.testing.assert_close(raw, arm_play(arm_obs), rtol=1e-6, atol=1e-6)
    mean = arm.act_inference(arm_obs, mode='training_mean')
    source_arm.update_distribution(arm_obs['obs_history'])
    torch.testing.assert_close(mean, source_arm.action_mean, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(mean[:, -2:], raw[:, -2:].tanh())
    assert not torch.allclose(mean[:, -2:], raw[:, -2:])
    assert not torch.allclose(mean[:, :-2], raw[:, :-2])


def _scipy_quat_namespace():
    def tensor(array, reference):
        return torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    def apply(q, v):
        return tensor(Rotation.from_quat(q.numpy()).apply(v.numpy()), q)
    def multiply(q, r):
        return tensor((Rotation.from_quat(q.numpy()) * Rotation.from_quat(r.numpy())).as_quat(), q)
    def from_euler(roll, pitch, yaw):
        angles = torch.stack((roll, pitch, yaw), dim=-1)
        return tensor(Rotation.from_euler('xyz', angles.numpy()).as_quat(), angles)
    return dict(torch=torch, quat_apply=apply, quat_rotate=apply, quat_mul=multiply,
                quat_from_euler_xyz=from_euler)


def _state(batch=3):
    from pawcerto.methods.roboduet.observations import quat_from_euler_xyz
    torch.manual_seed(902)
    base = quat_from_euler_xyz(*torch.randn(3, batch) * .3)
    ee = quat_from_euler_xyz(*torch.randn(3, batch) * .4)
    end = torch.cat((torch.randn(batch, 3), ee, torch.randn(batch, 6)), dim=-1)
    forward = torch.zeros(batch, 3)
    forward[:, 0] = 1
    return dict(dof_pos=torch.randn(batch, 20), dof_vel=torch.randn(batch, 20),
                default_dof_pos=torch.randn(1, 20), actions=torch.randn(batch, 18),
                base_quat=base, end_effector_state=end, root_states=torch.randn(batch, 13),
                projected_gravity=torch.randn(batch, 3), commands_dog=torch.randn(batch, 5),
                commands_arm_obs=torch.randn(batch, 6), clock_inputs=torch.randn(batch, 4),
                friction_coeffs=torch.rand(batch, 1), restitutions=torch.rand(batch, 1),
                measured_heights=torch.randn(batch, 4), forward_vec=forward)


@pytest.mark.parametrize('stage2', [False, True])
def test_observation_layout_privileged_math_and_history_against_source(stage2):
    config = default_config()
    cfg = _reference_config()['Cfg']
    ns = _scipy_quat_namespace()
    ns['global_switch'] = SimpleNamespace(switch_open=stage2)
    ns['get_scale_shift'] = lambda limits: (2. / (limits[1] - limits[0]), (limits[1] + limits[0]) / 2.)
    _definitions('go1_gym/utils/common.py', ns, ['quaternion_to_rpy'])
    lpy = _method('go1_gym/envs/automatic/legged_robot.py', 'LeggedRobot', 'get_lpy_in_base_coord', ns)
    state = _state()
    original = SimpleNamespace(**deepcopy(state), cfg=cfg, num_actions_loco=12, num_actions_arm=6,
                               obs_scales=cfg.obs_scales, num_envs=3, device='cpu')
    original.commands_scale_dog = torch.tensor([2., 2., .25, 1., 1.])
    original.get_lpy_in_base_coord = lambda ids: lpy(original, ids)
    observer = RoboDuetObservations(config, 3)
    for side in ('arm', 'dog'):
        function = _method('go1_gym/envs/automatic/__init__.py', 'VelocityTrackingEasyEnv',
                           'get_' + side + '_observations', ns)
        expected_obs, expected_priv = function(original)
        result = observer.arm(state) if side == 'arm' else observer.dog(state, stage2)
        torch.testing.assert_close(result['obs'], expected_obs, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result['privileged_obs'], expected_priv, rtol=1e-5, atol=1e-6)
        width = expected_obs.shape[1]
        assert torch.count_nonzero(result['obs_history'][:, :-width]) == 0
        torch.testing.assert_close(result['obs_history'][:, -width:], expected_obs)
        # Compare the actual wrapper append and clear functions independently.
        wrapper = SimpleNamespace(env=SimpleNamespace(cfg=cfg),
                                  dog_obs_history=torch.zeros(3, 1680), arm_obs_history=torch.zeros(3, 600),
                                  obs_history=torch.zeros(3, 1890))
        setattr(wrapper.env, 'get_' + side + '_observations', lambda: (expected_obs, expected_priv))
        push = _method('go1_gym/envs/automatic/__init__.py', 'HistoryWrapper',
                       'get_' + side + '_observations', ns)
        push(wrapper)
        expected = push(wrapper)
        actual = observer.arm(state) if side == 'arm' else observer.dog(state, stage2)
        torch.testing.assert_close(actual['obs_history'], expected['obs_history'], rtol=1e-5, atol=1e-6)
    observer.clear(torch.tensor([1]))
    assert torch.count_nonzero(observer.dog_obs_history[1]) == 0
    assert torch.count_nonzero(observer.arm_obs_history[1]) == 0
    assert torch.count_nonzero(observer.dog_obs_history[0]) > 0


def test_m_control_and_asymmetric_plan_match_original_functions():
    config = default_config()
    cfg = _reference_config()['Cfg']
    names = [f'{leg}_{part}_joint' for leg in ('FR', 'FL', 'RR', 'RL')
             for part in ('hip', 'thigh', 'calf')] + [f'zarx_j{i}' for i in range(1, 9)]
    controller = RoboDuetController(config, names, [23.7] * 12 + [15] * 3 + [3] * 5)
    state = _state()
    inputs = dict(kp_factors=torch.rand(3, 20), kd_factors=torch.rand(3, 20),
                  motor_offsets=torch.randn(3, 20), motor_strengths=torch.rand(3, 20) + .5)
    actions = torch.randn(3, 18) * 20
    result = controller.compute(actions, state['dof_pos'], state['dof_vel'], **inputs)
    reference = SimpleNamespace(cfg=cfg, num_actions=18, num_dof=20, num_actions_loco=12,
                                default_dof_pos=controller.default_dof_pos, dof_pos=state['dof_pos'],
                                dof_vel=state['dof_vel'], p_gains=torch.tensor([35.] * 12 + [50., 50., 70., 50., 50., 50., 0., 0.]),
                                d_gains=torch.tensor([1.] * 12 + [20., 20., 15., 20., 20., 20., 0., 0.]),
                                torque_limits=controller.torque_limits,
                                Kp_factors=inputs['kp_factors'], Kd_factors=inputs['kd_factors'],
                                motor_offsets=inputs['motor_offsets'], motor_strengths=inputs['motor_strengths'])
    ns = dict(torch=torch)
    compute = _method('go1_gym/envs/automatic/legged_robot.py', 'LeggedRobot', '_compute_torques', ns)
    expected = compute(reference, actions.clamp(-10, 10))
    torch.testing.assert_close(result['combined'], expected)
    torch.testing.assert_close(result['joint_pos_target'], reference.joint_pos_target)
    torch.testing.assert_close(result['leg_effort'], expected[:, :12])
    torch.testing.assert_close(result['arm_position'], expected[:, 12:])
    reference.commands_dog = torch.zeros(3, 5)
    reference.plan_actions = torch.zeros(3, 2)
    commands = reference.commands_dog.clone()
    guidance = torch.tensor([[10., -10.], [-10., 10.], [.3, -.7]])
    source_plan = _method('go1_gym/envs/automatic/__init__.py', 'VelocityTrackingEasyEnv', 'plan', ns)
    source_plan(reference, guidance)
    plan_actions = plan(commands, guidance, config)
    torch.testing.assert_close(commands, reference.commands_dog)
    torch.testing.assert_close(plan_actions, reference.plan_actions)
    assert commands[0, 3].item() == pytest.approx(.3)
    assert plan_actions[0, 0].item() == pytest.approx(4.)


def test_coordinated_actions_read_guidance_before_dog_observation():
    dog, arm = build_models()
    events = []
    class Env:
        num_envs = 2
        def get_arm_observations(self):
            events.append('arm')
            return {'obs': torch.zeros(2, 20), 'obs_history': torch.zeros(2, 600)}
        def plan(self, guidance):
            assert guidance.shape == (2, 2)
            events.append('plan')
        def get_dog_observations(self):
            events.append('dog')
            return {'obs': torch.zeros(2, 56), 'obs_history': torch.zeros(2, 1680)}
    actions = coordinated_actions(dog, arm, Env(), stage2=True)
    assert events == ['arm', 'plan', 'dog']
    assert [a.shape for a in actions] == [torch.Size([2, 12]), torch.Size([2, 6])]
    events.clear()
    _, arm_actions = coordinated_actions(dog, arm, Env(), stage2=False)
    assert events == ['dog'] and torch.count_nonzero(arm_actions) == 0


def test_nondefault_observation_paths_are_not_silently_reinterpreted():
    config = default_config()
    config['Cfg']['use_rot6d'] = True
    with pytest.raises(ValueError, match='no-rot6d'):
        RoboDuetObservations(config, 1)
