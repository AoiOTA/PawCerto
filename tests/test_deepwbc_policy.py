"""CPU comparisons to executable pinned DeepWBC source; no trained policy claim."""
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.distributions import Normal

from pawcerto.methods.deepwbc.config import default_config
from pawcerto.methods.deepwbc.policy import build_model
from pawcerto.methods.deepwbc.controller import DeepWBCController, POLICY_DOF_NAMES, wrap_to_pi
from pawcerto.methods.deepwbc.observer import DeepWBCObserver, sphere_to_cartesian, cartesian_to_sphere, goal_world

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'third_party/deepwbc-reference'
ENV = SOURCE / 'legged_gym/legged_gym/envs/widowGo1/widowGo1.py'
SIM_NAMES = tuple(f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'RL', 'RR')
                  for joint in ('hip', 'thigh', 'calf')) + POLICY_DOF_NAMES[12:]


def upstream_methods(*names):
    tree = ast.parse(ENV.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WidowGo1')
    mod = ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    ns = {'torch': torch, 'torch_wrap_to_pi_minuspi': wrap_to_pi}
    exec(compile(ast.fix_missing_locations(mod), str(ENV), 'exec'), ns)
    return ns


class DeepWBCCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ENV.exists():
            raise unittest.SkipTest('Run python3 scripts/fetch_deepwbc.py for pinned-source equivalence tests')
        torch.set_num_threads(1)

    def test_network_state_exchange_and_both_latents(self):
        spec = importlib.util.spec_from_file_location('deepwbc_fixed_actor', SOURCE / 'rsl_rl/rsl_rl/modules/actor_critic.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = default_config()
        validate = Normal.set_default_validate_args
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(49)
                original = module.ActorCritic(76, 76, 18, num_priv=24, num_hist=10, num_prop=76, **cfg['policy'])
        finally:
            Normal.set_default_validate_args = validate
        adapted = build_model(cfg)
        adapted.load_state_dict(original.state_dict(), strict=True)
        obs = torch.randn(5, 860)
        for history in (False, True):
            torch.testing.assert_close(adapted.act_inference(obs, history), original.act_inference(obs, history), rtol=0, atol=0)
        torch.testing.assert_close(adapted.evaluate(obs), original.evaluate(obs), rtol=0, atol=0)
        torch.testing.assert_close(adapted.actor.infer_hist_latent(obs), original.actor.infer_hist_latent(obs), rtol=0, atol=0)
        adapted.update_distribution(obs, True)
        original.update_distribution(obs, True)
        actions = torch.randn(5, 18)
        torch.testing.assert_close(adapted.get_actions_log_prob(actions), original.get_actions_log_prob(actions), rtol=0, atol=0)
        self.assertEqual(adapted.entropy.shape, (5, 2))
        # A training-compatible state can also be exchanged back to the actual source.
        adapted.act_inference(obs, True).square().mean().backward()
        with torch.no_grad():
            for p in adapted.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-0.001)
        original.load_state_dict(adapted.state_dict(), strict=True)
        torch.testing.assert_close(original.act_inference(obs, True), adapted.act_inference(obs, True), rtol=0, atol=0)

    def test_resolved_fresh_config_against_original_classes(self):
        ns = {'BaseConfig': type('BaseConfig', (), {}), 'np': np, 'torch': torch}
        for filename in ('base/legged_robot_config.py', 'widowGo1/widowGo1_config.py'):
            path = SOURCE / 'legged_gym/legged_gym/envs' / filename
            tree = ast.parse(path.read_text())
            tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
            for n in tree.body:
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RESUME' for t in n.targets):
                    n.value = ast.Constant(False)
            exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
        cfg = default_config()
        self.assertFalse(cfg['runner']['resume'])
        self.assertEqual(cfg['algorithm']['mixing_schedule'], ns['WidowGo1RoughCfgPPO'].algorithm.mixing_schedule)
        self.assertEqual(cfg['algorithm']['priv_reg_coef_schedual'], ns['WidowGo1RoughCfgPPO'].algorithm.priv_reg_coef_schedual)
        self.assertEqual(cfg['env']['num_observations'], 860)
        self.assertEqual(sum(s != 0 for s in cfg['control']['action_scale']), 15)
        paper = default_config('paper')
        self.assertEqual(paper['algorithm']['priv_reg_coef_schedual'], [0, 1, 5000, 5000])
        self.assertEqual(paper['runner']['max_iterations'], 10000)

    def test_torque_equation_and_delayed_reordering(self):
        cfg = default_config()
        controller = DeepWBCController(3, SIM_NAMES, torch.ones(20)*25)
        method = upstream_methods('_compute_torques')['_compute_torques']
        actions = torch.randn(3, 18)
        for _ in range(2):
            self.assertEqual(controller.push(actions).count_nonzero().item(), 0)
        expected = controller.push(actions)
        torch.testing.assert_close(expected, actions[:, controller.policy_to_sim])
        q, qd = torch.randn(3, 20), torch.randn(3, 20)
        q[:, 10] = 7  # expose original q18[-8] wrapping rather than silently fix it
        q[:, 12] = 7
        motor = torch.rand(3, 18)+0.5
        env = NS(cfg=NS(control=NS(**cfg['control'])), motor_strength=motor,
                 action_scale=controller.scale, dof_pos_wo_gripper=q[:, :18],
                 dof_pos_wo_gripper_wrapped=torch.zeros(3, 18), p_gains=controller.kp,
                 d_gains=controller.kd, default_dof_pos_wo_gripper=controller.default[:18],
                 dof_vel_wo_gripper=qd[:, :18], gripper_torques_zero=torch.zeros(3, 2),
                 torque_limits=controller.limits)
        actual = controller.torques(q, qd, motor)
        torch.testing.assert_close(actual, method(env, expected), rtol=0, atol=0)
        self.assertEqual(actual[:, -2:].count_nonzero().item(), 0)
        controller.reset([1])
        self.assertEqual(controller.history[1].count_nonzero().item(), 0)

    def test_observation_and_history_before_update(self):
        cfg = default_config()
        observer = DeepWBCObserver(3, SIM_NAMES)
        methods = upstream_methods('compute_observations', 'ig2raisim', 'ig2raisim_wo_gripper', 'ig2raisim_feet', 'get_foot_contacts')
        env = NS(cfg=NS(domain_rand=NS(observe_priv=True), env=NS(**cfg['env'])), num_envs=3,
                 dof_names=list(SIM_NAMES), obs_scales=NS(**cfg['normalization']['obs_scales']),
                 default_dof_pos=observer.default, obs_history_buf=observer.history.clone(),
                 commands_scale=torch.ones(3), dof_pos_wrapped=torch.zeros(3, 20))
        for name, fn in methods.items():
            if callable(fn) and name not in ('torch_wrap_to_pi_minuspi',):
                setattr(env, name, fn.__get__(env))
        env.get_body_orientation = lambda: env.roll_pitch
        for step in range(4):
            inputs = dict(roll_pitch=torch.randn(3, 2), base_ang_vel=torch.randn(3, 3),
                dof_pos=torch.randn(3, 20), dof_vel=torch.randn(3, 20), latest_sim_actions=torch.randn(3, 18),
                foot_wrenches=torch.randn(3, 4, 6), commands=torch.randn(3, 3), ee_goal_lpy=torch.randn(3, 3),
                ee_delta_orn=torch.zeros(3, 3), mass_params=torch.randn(3, 5), friction=torch.rand(3, 1),
                motor_strength=torch.rand(3, 18), episode_length=torch.tensor([step, step+4, step+10]))
            for name in ('roll_pitch', 'base_ang_vel', 'dof_pos', 'dof_vel', 'commands', 'motor_strength'):
                setattr(env, name, inputs[name])
            env.action_history_buf = inputs['latest_sim_actions'][:, None]
            env.force_sensor_tensor = inputs['foot_wrenches']
            env.curr_ee_goal = inputs['ee_goal_lpy']
            env.ee_goal_delta_orn_euler = inputs['ee_delta_orn']
            env.mass_params_tensor, env.friction_coeffs_tensor = inputs['mass_params'], inputs['friction']
            env.episode_length_buf = inputs['episode_length']
            env.compute_observations()
            actual = observer.observe(**inputs)
            torch.testing.assert_close(actual, env.obs_buf, rtol=0, atol=0)
            torch.testing.assert_close(observer.history, env.obs_history_buf, rtol=0, atol=0)

    def test_reconstructed_lpy_roundtrip_and_world_frame(self):
        lpy = torch.tensor([[0.6, 0.2, 0.5], [0.4, -0.5, -1.0]])
        torch.testing.assert_close(cartesian_to_sphere(sphere_to_cartesian(lpy)), lpy)
        world = goal_world(torch.tensor([[0.5, 0., 0.]]), torch.tensor([[2., 3.]]), torch.tensor([torch.pi/2]))
        torch.testing.assert_close(world, torch.tensor([[2., 3.5, 0.53]]))

    def test_official_urdf_mesh_closure(self):
        urdf = SOURCE / 'legged_gym/resources/robots/widowGo1/urdf/widowGo1.urdf'
        tree = ET.parse(urdf)
        meshes = {x.attrib['filename'] for x in tree.findall('.//mesh')}
        self.assertEqual(len(meshes), 15)
        for mesh in meshes:
            self.assertTrue((urdf.parent / mesh).is_file(), mesh)
        joints = [n for n in tree.findall('joint') if n.attrib['type'] != 'fixed']
        self.assertEqual(len(joints), 20)


if __name__ == '__main__':
    unittest.main()
