"""Compare the adapter to executable upstream CPU equations and real artifacts.

Run with PyTorch and PyTorch3D available; no simulator import or GPU is needed.
AST loading only avoids the upstream module's IsaacGym import-time dependency.
The extracted implementations themselves are executed unchanged.
"""
import ast
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
import pickle
import unittest
import tempfile
import json

import numpy as np
import torch
import pytorch3d.transforms as pt3d

from pawcerto.methods.umi_on_legs import UmiPolicy, RobotState, axis_angle_to_matrix

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/umi-on-legs/mani-centric-wbc/legged_gym'


def upstream_nodes(path, class_names=(), method_names=()):
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name in class_names:
                selected.append(node)
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and (node.name, method.name) in method_names:
                    selected.append(method)
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + selected
    namespace = dict(torch=torch, np=np, pt3d=pt3d, pickle=pickle, ABC=ABC, abstractmethod=abstractmethod)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace


class UmiPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_export_reload_original_and_training_checkpoint_paths(self):
        names = json.loads((ROOT / 'configs/umi_go2_arx5_joint_names.json').read_text())
        for variant in ('ours', 'ours-real'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                original = UmiPolicy(ROOT / 'reference/checkpoints/tossing' / variant)
                # A real saved actor state in the training filename/schema. This
                # checks checkpoint compatibility, not a training outcome.
                (directory / 'config.json').write_text(json.dumps(original.config))
                torch.save({'model_state_dict': {'actor.' + k: v for k, v in original.actor.state_dict().items()},
                            'iter': 3}, directory / 'model_3.pt')
                trained_path = UmiPolicy(directory / 'model_3.pt')
                report = trained_path.export(directory / 'export', names)
                self.assertEqual(report['max_actor_error'], 0.)
                loaded = UmiPolicy(directory / 'export')
                self.assertEqual(loaded.config, original.config)
                self.assertEqual(loaded.joint_names, names)
                pos, rot = original.trajectories(ROOT / 'reference/data/tossing.pkl').sample(2, 4)
                obs = original.observer(pos, rot)
                ctrl = original.controller(2)
                pose = torch.eye(4).repeat(2, 1, 1)
                pose[:, :3, 3] = torch.tensor([.5, 0., .5])
                obs.advance_pose(pose)
                obs.advance_pose(pose)
                state = RobotState(torch.zeros(2,3), torch.tensor([[0.,0.,-1.]]).repeat(2,1),
                                   ctrl.offset.repeat(2,1), torch.zeros(2,18), pose, torch.tensor([0., 1.]))
                observation = obs.observe(state, ctrl.last_action)
                with torch.inference_mode():
                    torch.testing.assert_close(loaded.actor(observation), original.actor(observation), rtol=0, atol=0)
                self.assertEqual(sorted(p.name for p in (directory / 'export').iterdir()),
                                 ['actor.ts', 'config.json', 'export.json', 'joint_names.json'])
                import hashlib
                provenance = json.loads((directory / 'export/export.json').read_text())
                self.assertEqual(provenance['source_weights']['sha256'],
                                 hashlib.sha256((directory / 'model_3.pt').read_bytes()).hexdigest())
                self.assertEqual(provenance['actor']['sha256'],
                                 hashlib.sha256((directory / 'export/actor.ts').read_bytes()).hexdigest())
                self.assertIsNone(loaded.training_selection)

    def test_real_trajectory_matches_upstream(self):
        ns = upstream_nodes(UPSTREAM / 'env/isaacgym/pose_sequence.py', class_names=('SequenceSampler', 'PicklePoseSequenceLoader'))
        policy = UmiPolicy(ROOT / 'reference/checkpoints/tossing/ours')
        cfg = policy.config['env']['tasks']['reaching']['sequence_sampler']
        source = ROOT / 'reference/data/tossing.pkl'
        reference = ns['PicklePoseSequenceLoader'](file_path=source, device='cpu', **{k: v for k, v in cfg.items() if k not in ('file_path', 'device', '_target_')})
        actual = policy.trajectories(source).sample(3, 17)
        expected = reference.sample(3, 17)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=5e-7)
        self.assertEqual(tuple(actual[0].shape), (3, 3400, 3))

    def test_observation_actor_and_pd_match_upstream(self):
        task = upstream_nodes(UPSTREAM / 'env/isaacgym/task.py', method_names=(('ReachingLinkTask', 'observe'),))['observe']
        attribute = upstream_nodes(UPSTREAM / 'env/obs.py', method_names=(('ObservationAttribute', '__call__'),))['__call__']
        controllers = upstream_nodes(UPSTREAM / 'env/isaacgym/control.py', class_names=('PDController', 'PositionController'))
        # ActorCritic imports are simulator independent; load the original class.
        import importlib.util
        spec = importlib.util.spec_from_file_location('upstream_actor', UPSTREAM / 'rsl_rl/modules/actor_critic.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(19)
        for variant, dim in [('ours', 132), ('ours-real', 96)]:
            with self.subTest(variant=variant):
                policy = UmiPolicy(ROOT / 'reference/checkpoints/tossing' / variant)
                pos, rot = policy.trajectories(ROOT / 'reference/data/tossing.pkl').sample(2, 7)
                obs = policy.observer(pos, rot)
                ctrl = policy.controller(2)
                state = RobotState(torch.randn(2, 3), torch.randn(2, 3), ctrl.offset + torch.randn(2, 18) * .1,
                                   torch.randn(2, 18), torch.eye(4).repeat(2, 1, 1), torch.tensor([0., 16.99]))
                state.sim_dt = ctrl.dt
                ac_cfg = policy.config['runner']['alg']['actor_critic']
                def network(config):
                    return torch.nn.Sequential(*[getattr(torch.nn, l['_target_'].split('.')[-1])(**{k:v for k,v in l.items() if not k.startswith('_')}) for l in config['_args_']])
                original = module.ActorCritic(actor=network(ac_cfg['actor']), critic=network(ac_cfg['critic']), num_actions=18)
                original.load_state_dict(torch.load(ROOT / 'reference/checkpoints/tossing' / variant / 'model.pt', map_location='cpu', weights_only=False)['model_state_dict'])
                cfg = obs.task
                def get_pose(times, sim_dt):
                    p, r = obs.targets(times)
                    result = torch.eye(4).repeat(2, 1, 1)
                    result[:, :3, 3], result[:, :3, :3] = p, r
                    return result
                reference = SimpleNamespace(num_envs=2, device='cpu', storage_device='cpu', target_obs_times=cfg['target_obs_times'],
                                            get_target_pose=get_pose, get_latency_scheduler=lambda: 1., pose_latency_frames=obs.frames,
                                            pose_latency_frame_variability=None, target_relative_to_base=False, position_noise=0., euler_noise=0.,
                                            position_obs_encoding='linear', pos_obs_scale=3., orn_obs_scale=1.5, pos_obs_clip=None)
                pd_cfg = policy.config['env']['controller']
                pd = controllers['PositionController'](control_dim=18, device='cpu', num_envs=2, decimation_count=4,
                                                       **{k:torch.tensor(pd_cfg[k]['data']) for k in ('kp','kd','scale','offset','torque_limit')})
                for step in range(4):
                    reference.link_pose_history = obs.pose_history
                    actual = obs.observe(state, ctrl.last_action)
                    # Execute the actual producer ordering from Env.__init__.
                    env_ast = ast.parse((UPSTREAM / 'env/isaacgym/env.py').read_text())
                    env_cls = next(n for n in env_ast.body if isinstance(n, ast.ClassDef) and n.name == 'IsaacGymEnv')
                    init = next(n for n in env_cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
                    assign = next(n for n in init.body if isinstance(n, ast.Assign)
                                  and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr == 'state_obs')
                    raw_attrs = {}
                    for key, a in policy.config['env']['state_obs'].items():
                        if key.startswith('_'): continue
                        a = {k:v for k,v in a.items() if not k.startswith('_')}
                        if a['offset'] is not None: a['offset'] = torch.tensor(a['offset']['data'])
                        raw_attrs[key] = SimpleNamespace(**a)
                    ordered = eval(compile(ast.Expression(assign.value), 'upstream_state_obs_constructor', 'eval'),
                                   {'state_obs': raw_attrs, 'ObservationAttribute': SimpleNamespace})
                    pieces = [attribute(a, state) for a in ordered.values()]
                    expected = torch.cat((*pieces, task(reference, state), ctrl.last_action), -1)
                    self.assertEqual(actual.shape[-1], dim)
                    # Exercise the constructor's sorted ordering, not JSON insertion order.
                    torch.testing.assert_close(actual[:, :18], state.dof_pos - ctrl.offset)
                    torch.testing.assert_close(actual[:, 18:36], state.dof_vel * .05)
                    torch.testing.assert_close(actual[:, 36:39], state.local_root_gravity)
                    torch.testing.assert_close(actual[:, 39:42], state.local_root_ang_vel * .25)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    with torch.no_grad():
                        torch.testing.assert_close(policy.actor(actual), original.act_inference(expected), rtol=1e-5, atol=1e-5)
                    ctrl.push(policy.actor(actual))
                    for substep in range(4):
                        idx = torch.ceil((ctrl.delay_steps - substep) / 4).long()
                        delayed = ctrl.buffer.permute(2,1,0)[torch.arange(18),idx].T
                        torch.testing.assert_close(ctrl.torque(state.dof_pos,state.dof_vel,substep), pd(delayed,state), rtol=0,atol=0)
                    state.ee_pose[:, :3, :3] = axis_angle_to_matrix(torch.randn(2,3))
                    state.ee_pose[:, :3, 3] = torch.randn(2,3)
                    obs.advance_pose(state.ee_pose)
                    state.episode_time += .02


if __name__ == '__main__':
    unittest.main()
