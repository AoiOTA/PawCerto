import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from pawcerto.mujoco.runtime import Go2Arx5Mujoco


class SubstepTraceTest(unittest.TestCase):
    def test_trace_preserves_integration_and_exposes_delay_and_torque(self):
        baseline, traced = Go2Arx5Mujoco(), Go2Arx5Mujoco()
        rows = []
        with patch('pawcerto.mujoco.runtime.mujoco.mj_forward', wraps=mujoco.mj_forward) as forward:
            for action in (np.zeros(18), np.linspace(-100, 100, 18), np.zeros(18)):
                baseline.step(action)
                traced.step(action, trace=rows.append)
                np.testing.assert_array_equal(baseline.data.qpos, traced.data.qpos)
                np.testing.assert_array_equal(baseline.data.qvel, traced.data.qvel)
                np.testing.assert_array_equal(baseline.data.qacc_warmstart, traced.data.qacc_warmstart)
            self.assertEqual(forward.call_count, 6)  # one per policy step, never per trace
        self.assertEqual(len(rows), 12)
        for row in rows:
            expected = np.where(row['substep'] < traced.delay, row['previous_action'], row['clipped_action'])
            np.testing.assert_array_equal(row['executed_action'], expected)
            target = traced.offset + traced.scale * expected
            np.testing.assert_array_equal(row['joint_target'], target)
            torque = traced.kp * (target - row['qpos_before'][traced.qadr]) - traced.kd * row['qvel_before'][traced.vadr]
            np.testing.assert_array_equal(row['torque_before_limit'], torque)
            np.testing.assert_array_equal(row['torque_applied'], np.clip(torque, -traced.torque_limit, traced.torque_limit))
            self.assertAlmostEqual(row['time_after_s'] - row['time_before_s'], .005)

    def test_callback_failure_is_not_hidden(self):
        sim = Go2Arx5Mujoco()
        def fail(row):
            raise RuntimeError('diagnostic output failed')
        with self.assertRaisesRegex(RuntimeError, 'diagnostic output failed'):
            sim.step(np.zeros(18), trace=fail)


if __name__ == '__main__':
    unittest.main()
