"""Actual-contact fixtures for the read-only policy-endpoint contact records."""
import unittest
import mujoco
import numpy as np
from pawcerto.mujoco.runtime import Go2Arx5Mujoco
from pawcerto.mujoco.contacts import contact_snapshot, geom_role


class ContactReadTest(unittest.TestCase):
    def test_reader_does_not_change_state_or_next_step(self):
        sampled, baseline = Go2Arx5Mujoco(), Go2Arx5Mujoco()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(sampled.model, spec))
        after = np.empty_like(before)
        for _ in range(4):
            mujoco.mj_getState(sampled.model, sampled.data, before, spec)
            forces = sampled.data.efc_force.copy()
            contact_snapshot(sampled)
            mujoco.mj_getState(sampled.model, sampled.data, after, spec)
            np.testing.assert_array_equal(before, after)
            np.testing.assert_array_equal(forces, sampled.data.efc_force)
            a, b = sampled.step(np.zeros(18)), baseline.step(np.zeros(18))
            np.testing.assert_array_equal(a['joint_pos'], b['joint_pos'])
            np.testing.assert_array_equal(sampled.data.qpos, baseline.data.qpos)
            np.testing.assert_array_equal(sampled.data.qvel, baseline.data.qvel)

    def test_actual_external_and_self_contacts_and_finger_exclusion(self):
        sim = Go2Arx5Mujoco()
        # A static lowered fixture produces actual base/ground solver contacts.
        sim.data.qpos[2] = .035
        mujoco.mj_forward(sim.model, sim.data)
        foot_z, ground_z, records = contact_snapshot(sim)
        self.assertTrue(any(r['body_names'] == ['world', 'base'] and r['kind'] == 'external'
                            and r['force_norm_gt_1N'] for r in records))
        self.assertTrue(np.all(foot_z > 0))
        np.testing.assert_allclose(foot_z, ground_z)
        self.assertTrue(any(r['foot_ground'] and not r['nonfoot_nongripper'] for r in records))
        self.assertFalse(any(set(r['geom_roles']) <= {'external', 'foot', 'gripper_finger'}
                             for r in records if r['nonfoot_nongripper']))
        link6_geoms = [i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == sim.model.body('link6').id]
        self.assertEqual([geom_role(sim.model, i) for i in link6_geoms],
                         ['nonfoot_nongripper', 'nonfoot_nongripper', 'gripper_finger', 'gripper_finger'])
        # Within-limit folded-arm fixture found from the actual model ranges.
        rng = np.random.default_rng(23)
        for _ in range(2):
            q = np.array([rng.uniform(*sim.model.joint(name).range) for name in sim.joint_names])
        sim.data.qpos[2] = 2.
        sim.data.qpos[sim.qadr] = q
        mujoco.mj_forward(sim.model, sim.data)
        _, ground_z, records = contact_snapshot(sim)
        self.assertTrue(records)
        np.testing.assert_array_equal(ground_z, np.zeros(4))
        self.assertTrue(all(r['kind'] == 'self' and 'world' not in r['body_names'] for r in records))
        for r in records:
            self.assertAlmostEqual(r['force_norm_N'], np.linalg.norm(r['force_world_on_geom2_N']))
            self.assertEqual(r['force_norm_gt_1N'], r['force_norm_N'] > 1.)

    def test_foot_self_contact_does_not_count_as_ground_support(self):
        sim = Go2Arx5Mujoco()
        sim.data.qpos[2] = 2.
        rng = np.random.default_rng(23)
        for _ in range(5):
            sim.data.qpos[sim.qadr] = [rng.uniform(*sim.model.joint(name).range) for name in sim.joint_names]
        mujoco.mj_forward(sim.model, sim.data)
        net_z, ground_z, records = contact_snapshot(sim)
        self.assertTrue(any(r['kind'] == 'self' and 'foot' in r['geom_roles'] for r in records))
        self.assertGreater(np.max(net_z), 1.)
        np.testing.assert_array_equal(ground_z, np.zeros(4))
        self.assertFalse(any(r['foot_ground'] for r in records))


if __name__ == '__main__':
    unittest.main()
