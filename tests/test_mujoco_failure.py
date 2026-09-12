"""Regression for real MuJoCo auto-reset hiding QACC failure behind finite qpos."""
import unittest
import numpy as np
from pawcerto.mujoco.runtime import Go2Arx5Mujoco, SimulationInstability
from pawcerto.mujoco.evaluate import evaluate, summarize_cases
from pawcerto.mujoco.asset import ROOT


class NumericalFailureTest(unittest.TestCase):
    def test_real_qacc_auto_reset_is_propagated(self):
        sim = Go2Arx5Mujoco()
        sim.step(np.zeros(18))
        sim.step(np.zeros(18))
        sim.data.qfrc_applied[0] = 1e20  # force the engine's actual BADQACC path
        with self.assertRaises(SimulationInstability) as caught:
            sim.step(np.zeros(18))
        failure = caught.exception.details
        self.assertIn('mjWARN_BADQACC', failure['new_warnings'])
        self.assertTrue(failure['engine_time_restarted'])
        self.assertAlmostEqual(failure['time_before_step_s'], .04)
        self.assertTrue(np.all(np.isfinite(sim.data.qpos)))  # old qpos-only check missed it

    def test_invalid_prefix_is_not_pooled_as_a_complete_episode(self):
        complete = evaluate(ROOT/'reference/checkpoints/tossing/ours', ROOT/'reference/data/tossing.pkl', seconds=.08)
        complete['case_index'] = 0
        invalid = dict(complete, case_index=1, completion_status='invalid', metrics_scope='pre_failure_prefix',
                       position_error_mean_m=1000., failure={'kind':'mujoco_numerical_failure'})
        result = summarize_cases([complete, invalid], 'test', 'test', .08, 0)
        self.assertEqual(result['requested_cases'], 2)
        self.assertEqual(result['complete_cases'], 1)
        self.assertEqual(result['invalid_cases'], 1)
        self.assertEqual(result['invalid_case_indices'], [1])
        self.assertNotIn('position_error_mean_m', result)
        self.assertEqual(result['complete_case_metrics']['position_error_mean_m'], complete['position_error_mean_m'])
        self.assertEqual(len(result['case_results']), 2)


if __name__ == '__main__':
    unittest.main()
