"""Failure-accounting regression tests; these fixtures are not a simulator."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pawcerto.methods.unifp.evaluation import evaluate_fixed_policy


class UniFPEvaluationAccountingTests(unittest.TestCase):
    def policy(self):
        return SimpleNamespace(act_inference=lambda obs: torch.zeros(2, 17),
                               get_student_latent=lambda history: torch.zeros(2, 12))

    def test_invalid_environment_does_not_discard_other_finite_transitions(self):
        obs = {'obs': torch.zeros(2, 2336), 'obs_pred': torch.zeros(2, 12)}
        env = SimpleNamespace(num_envs=2, transition_observer=None,
                              reset=lambda: None, get_observations=lambda: obs)
        def step(action):
            env.transition_observer(env)
            return obs, None, None, None
        env.step = step
        snapshots = []
        for errors in ([1., 3.], [5., float('nan')]):
            snapshots.append({'ee_position_error_m': np.array(errors),
                              'ee_compliance_error_m': np.array(errors),
                              'base_up_dot': np.ones(2), 'base_velocity_error': np.zeros((2, 2)),
                              'done': np.zeros(2, dtype=bool), 'reason_mask': np.zeros((2, 4), dtype=bool)})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'evaluation'
            with patch('pawcerto.methods.unifp.evaluation.transition_snapshot', side_effect=snapshots):
                with self.assertRaises(FloatingPointError):
                    evaluate_fixed_policy(env, self.policy(), steps=2, output=output,
                                          metadata={}, exported_policy=self.policy())
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['finite_environment_transitions'], 3)
            self.assertEqual(report['finite_policy_steps'], 1)
            self.assertEqual(report['ee_position_error_m_mean_finite_transitions'], 3.)
            self.assertTrue(report['exported_policy_consumed'])
            self.assertEqual(report['exported_policy_observed_steps'], 2)
            with np.load(output / 'trajectory.npz') as saved:
                self.assertTrue(np.isnan(saved['ee_position_error_m'][1, 1]))

    def test_reset_failure_cannot_claim_export_consumption_or_zero_error(self):
        def fail_reset():
            raise RuntimeError('fixture reset failure')
        env = SimpleNamespace(num_envs=2, transition_observer=None, reset=fail_reset)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'evaluation'
            with self.assertRaisesRegex(RuntimeError, 'fixture reset failure'):
                evaluate_fixed_policy(env, self.policy(), steps=2, output=output,
                                      metadata={}, exported_policy=self.policy())
            report = json.loads((output / 'summary.json').read_text())
            self.assertFalse(report['exported_policy_consumed'])
            self.assertEqual(report['exported_policy_observed_steps'], 0)
            self.assertEqual(report['compared_policy_steps'], 0)
            self.assertIsNone(report['exported_action_max_abs_error'])
            self.assertEqual(report['saved_policy_steps'], 0)


if __name__ == '__main__':
    unittest.main()
