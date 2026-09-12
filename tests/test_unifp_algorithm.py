"""CPU equation/state parity against actual pinned upstream source, not learning."""
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from pawcerto.methods.unifp.algorithm import ActorCritic, PPO, RolloutStorage, OnPolicyRunner

REFERENCE = Path(__file__).resolve().parents[1] / 'third_party/unifp-reference/legged_gym/b2_gym_learn'


def reference_modules():
    if not (REFERENCE / 'ppo_cse_pf/actor_critic.py').exists():
        raise unittest.SkipTest('Fetch the pinned UniFP reference to run upstream parity')
    class PrefixProto:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()
    shim = types.ModuleType('params_proto')
    shim.PrefixProto = PrefixProto
    modules = {'params_proto': shim}
    for name in ['legged_gym', 'legged_gym.b2_gym_learn', 'legged_gym.b2_gym_learn.ppo_cse_pf']:
        modules[name] = types.ModuleType(name)
    with patch.dict(sys.modules, modules):
        for name, path in [('utils', 'utils/utils.py'), ('ppo_cse_pf.actor_critic', 'ppo_cse_pf/actor_critic.py'), ('ppo_cse_pf.rollout_storage', 'ppo_cse_pf/rollout_storage.py'), ('ppo_cse_pf.ppo', 'ppo_cse_pf/ppo.py')]:
            full = 'legged_gym.b2_gym_learn.' + name
            spec = importlib.util.spec_from_file_location(full, REFERENCE / path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            modules[full] = module
            spec.loader.exec_module(module)
    return modules


def model(cls=ActorCritic, *, full_size=False):
    # Upstream overwrites this torch method; contain that original side effect.
    original = torch.distributions.Normal.set_default_validate_args
    try:
        dimensions = (2336, 447, 12, 73, 17) if full_size else (18, 15, 12, 6, 3)
        hidden = [512, 256, 128] if full_size else [16, 8]
        with redirect_stdout(io.StringIO()):
            return cls(*dimensions, actor_hidden_dims=hidden, critic_hidden_dims=hidden)
    finally:
        torch.distributions.Normal.set_default_validate_args = original


class AlgorithmParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.upstream = reference_modules()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_actual_upstream_state_and_outputs(self):
        upstream_cls = self.upstream['legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic'].ActorCritic
        torch.manual_seed(731)
        upstream = model(upstream_cls)
        torch.manual_seed(731)
        local = model()
        self.assert_tree_equal(local.state_dict(), upstream.state_dict())
        local.load_state_dict(upstream.state_dict(), strict=True)
        upstream.load_state_dict(local.state_dict(), strict=True)
        obs, critic = torch.randn(5, 18), torch.randn(5, 15)
        self.assert_tree_equal(local.act_student(obs, {}), upstream.act_student(obs, {}))
        self.assert_tree_equal(local.get_student_latent(obs), upstream.get_student_latent(obs))
        self.assert_tree_equal(local.evaluate(critic), upstream.evaluate(critic))
        local.update_distribution(obs)
        upstream.update_distribution(obs)
        self.assert_tree_equal(local.action_mean, upstream.action_mean)
        self.assert_tree_equal(local.entropy, upstream.entropy)

    def test_full_b2_z1_upstream_state_and_outputs(self):
        upstream_cls = self.upstream['legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic'].ActorCritic
        torch.manual_seed(732)
        upstream = model(upstream_cls, full_size=True)
        torch.manual_seed(732)
        local = model(full_size=True)
        self.assert_tree_equal(local.state_dict(), upstream.state_dict())
        local.load_state_dict(upstream.state_dict(), strict=True)
        upstream.load_state_dict(local.state_dict(), strict=True)
        obs, critic = torch.randn(2, 2336), torch.randn(2, 447)
        with torch.no_grad():
            actions = local.act_student(obs, {})
            estimate = local.get_student_latent(obs)
            value = local.evaluate(critic)
            self.assertEqual(actions.shape, (2, 17))
            self.assertEqual(estimate.shape, (2, 12))
            self.assertEqual(value.shape, (2, 1))
            self.assert_tree_equal(actions, upstream.act_student(obs, {}))
            self.assert_tree_equal(estimate, upstream.get_student_latent(obs))
            self.assert_tree_equal(value, upstream.evaluate(critic))
            local.update_distribution(obs)
            upstream.update_distribution(obs)
            self.assert_tree_equal(local.action_mean, upstream.action_mean)
            self.assert_tree_equal(local.action_std, upstream.action_std)
            self.assert_tree_equal(local.entropy, upstream.entropy)

    def test_real_update_matches_upstream_and_checkpoint(self):
        reference_ppo = self.upstream['legged_gym.b2_gym_learn.ppo_cse_pf.ppo'].PPO
        reference_ac = self.upstream['legged_gym.b2_gym_learn.ppo_cse_pf.actor_critic'].ActorCritic
        torch.manual_seed(819)
        local_ac, upstream_ac = model(), model(reference_ac)
        upstream_ac.load_state_dict(local_ac.state_dict())
        kwargs = dict(num_learning_epochs=2, num_mini_batches=2, schedule='adaptive', entropy_coef=0.01)
        local, upstream = PPO(local_ac, **kwargs), reference_ppo(upstream_ac, **kwargs)
        initial = {key: value.clone() for key, value in local_ac.state_dict().items()}
        for alg in (local, upstream):
            alg.init_storage(4, 3, [18], [15], [12], [3])
        for step in range(3):
            obs, critic, targets = torch.randn(4, 18), torch.randn(4, 15), torch.randn(4, 12)
            rewards = torch.randn(4)
            dones = torch.tensor([step == 1, False, step == 2, False])
            timeouts = torch.tensor([False, False, step == 2, False])
            for alg in (local, upstream):
                torch.manual_seed(900 + step)
                with torch.no_grad():
                    alg.act(obs, critic, targets)
                    expected_reward = rewards + alg.gamma * alg.transition.values.squeeze(1) * timeouts
                    alg.process_env_step(rewards, dones, {'time_outs': timeouts})
                self.assert_tree_equal(alg.storage.rewards[step, :, 0], expected_reward)
        last_critic = torch.randn(4, 15)
        for alg in (local, upstream):
            alg.compute_returns(last_critic)
        self.assert_tree_equal(local.storage.returns, upstream.storage.returns)
        self.assert_tree_equal(local.storage.advantages, upstream.storage.advantages)
        torch.manual_seed(17)
        actual = local.update()
        torch.manual_seed(17)
        expected = upstream.update()
        self.assert_tree_equal(actual, expected)
        self.assert_tree_equal(local_ac.state_dict(), upstream_ac.state_dict())
        self.assert_tree_equal(local.optimizer.state_dict(), upstream.optimizer.state_dict())
        self.assert_tree_equal(local.adaptation_module_optimizer.state_dict(), upstream.adaptation_module_optimizer.state_dict())
        self.assertEqual(local.storage.step, 0)
        for prefix in ['actor_body', 'critic_body', 'adaptation_encoder_module', 'adaptation_decoder_module']:
            self.assertTrue(any(not torch.equal(initial[k], v) for k, v in local_ac.state_dict().items() if k.startswith(prefix)))
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in actual[:3]))
        # Exercise the runner checkpoint path using this actual update, no environment training.
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.alg, runner.device, runner.current_learning_iteration = local, 'cpu', 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_2.pt'
            runner.save(path, {'source': 'CPU equation test'})
            restored = OnPolicyRunner.__new__(OnPolicyRunner)
            restored.alg, restored.device = PPO(model(), **kwargs), 'cpu'
            self.assertEqual(restored.load(path), {'source': 'CPU equation test'})
            self.assertEqual(restored.current_learning_iteration, 2)
            self.assert_tree_equal(restored.alg.actor_critic.state_dict(), local_ac.state_dict())
            self.assert_tree_equal(restored.alg.optimizer.state_dict(), local.optimizer.state_dict())
            self.assert_tree_equal(restored.alg.adaptation_module_optimizer.state_dict(), local.adaptation_module_optimizer.state_dict())
            self.assertEqual(restored.alg.learning_rate, local.learning_rate)

    def test_runner_without_logging_or_environment_steps(self):
        class ObservationOnlyEnv:
            num_envs, num_obs, num_privileged_obs = 2, 18, 15
            num_pred_obs, num_single_obs, num_actions = 12, 6, 3

            def reset(self):
                return self.get_observations(), {}

            def get_observations(self):
                return {'obs': torch.zeros(2, 18), 'privileged_obs': torch.zeros(2, 15),
                        'obs_pred': torch.zeros(2, 12)}

            def step(self, actions):
                raise AssertionError('This constructor/no-op test must not train')

        cfg = {'runner': {'experiment_name': 'cpu', 'run_name': 'interface',
                          'policy_class_name': 'ActorCritic', 'algorithm_class_name': 'PPO',
                          'num_steps_per_env': 3, 'save_interval': 1},
               'policy': {'actor_hidden_dims': [16, 8], 'critic_hidden_dims': [16, 8]},
               'algorithm': {}}
        with patch.dict(sys.modules, {'wandb': None, 'torch.utils.tensorboard': None}), redirect_stdout(io.StringIO()):
            runner = OnPolicyRunner(ObservationOnlyEnv(), cfg)
            runner.learn(0)
        self.assertEqual(runner.current_learning_iteration, 0)
        self.assertIsNone(runner.writer)
        runner.alg.test_mode()
        self.assertFalse(runner.alg.actor_critic.training)

    def test_gae_equation_and_statistics_no_mutation(self):
        storage = RolloutStorage(2, 2, [1], [1], [1], [1])
        storage.rewards.copy_(torch.tensor([[[1.], [2.]], [[3.], [4.]]]))
        storage.values.copy_(torch.tensor([[[.2], [.3]], [[.4], [.5]]]))
        storage.dones[0, 0] = 1
        last = torch.tensor([[.6], [.7]])
        gamma, lam = .9, .8
        a1 = storage.rewards[1] + gamma * last - storage.values[1]
        a0 = storage.rewards[0] + torch.tensor([[0.], [1.]]) * gamma * storage.values[1] - storage.values[0] + torch.tensor([[0.], [1.]]) * gamma * lam * a1
        advantages = torch.stack([a0, a1])
        storage.compute_returns(last, gamma, lam)
        self.assert_tree_equal(storage.returns, advantages + storage.values)
        torch.testing.assert_close(storage.advantages, (advantages - advantages.mean()) / (advantages.std() + 1e-8))
        before = storage.dones.clone()
        storage.get_statistics()
        self.assert_tree_equal(storage.dones, before)


if __name__ == '__main__':
    unittest.main()
