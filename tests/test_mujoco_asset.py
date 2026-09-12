"""Numerical checks against source URDF, plus actual floating-base PD stepping."""
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from pawcerto.mujoco import Go2Arx5Mujoco
from pawcerto.mujoco.asset import URDF


def urdf_fk(robot, q, base_pose):
    poses = {'base': base_pose}
    pending = list(robot.findall('joint'))
    while pending:
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent not in poses:
                continue
            origin = joint.find('origin')
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            motion = np.eye(4)
            if joint.get('type') == 'revolute':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                motion[:3, :3] = Rotation.from_rotvec(axis * q[joint.get('name')]).as_matrix()
            poses[joint.find('child').get('link')] = poses[parent] @ transform @ motion
            pending.remove(joint)
    return poses


class AssetTest(unittest.TestCase):
    def test_random_pose_fk_and_source_inertia(self):
        sim = Go2Arx5Mujoco()
        robot = ET.parse(URDF).getroot()
        mass = sum(float(x.get('value')) for x in robot.findall('./link/inertial/mass'))
        self.assertAlmostEqual(sim.model.body_mass.sum(), mass, places=10)
        # Check each explicit inertial tensor in its link frame, including off-diagonals.
        for link in robot.findall('link'):
            inertial = link.find('inertial')
            if inertial is None or float(inertial.find('mass').get('value')) == 0:
                continue
            i = inertial.find('inertia')
            xx, yy, zz, xy, xz, yz = [float(i.get(k)) for k in ('ixx','iyy','izz','ixy','ixz','iyz')]
            source = np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
            body = sim.model.body(link.get('name'))
            rotation = Rotation.from_quat(body.iquat[[1,2,3,0]]).as_matrix()
            np.testing.assert_allclose(rotation @ np.diag(body.inertia) @ rotation.T, source, atol=1e-7)
        rng = np.random.default_rng(0)
        max_error = 0.
        for _ in range(32):
            q = sim.offset + rng.uniform(-.4, .4, 18)
            pose = np.eye(4)
            pose[:3,:3] = Rotation.random(random_state=rng).as_matrix()
            pose[:3,3] = rng.normal(size=3)
            sim.data.qpos[:3] = pose[:3,3]
            sim.data.qpos[3:7] = Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
            sim.data.qpos[sim.qadr] = q
            mujoco.mj_forward(sim.model, sim.data)
            expected = urdf_fk(robot, dict(zip(sim.joint_names,q)), pose)
            for name, transform in expected.items():
                body = sim.data.body(name)
                max_error = max(max_error, float(np.max(np.abs(body.xpos-transform[:3,3]))),
                                float(np.max(np.abs(body.xmat.reshape(3,3)-transform[:3,:3]))))
            self.assertLess(max_error, 5e-6)
        print(f'URDF/MuJoCo FK: 32 random poses, all {len(expected)} links; maximum component error={max_error:.3g}; mass={mass:.6f} kg')

    def test_link_origin_velocity(self):
        sim = Go2Arx5Mujoco()
        sim.data.qvel[:] = np.random.default_rng(8).normal(size=sim.model.nv)
        mujoco.mj_forward(sim.model, sim.data)
        before = sim.state()
        dt = 1e-7
        mujoco.mj_integratePos(sim.model, sim.data.qpos, sim.data.qvel, dt)
        mujoco.mj_forward(sim.model, sim.data)
        after = sim.state()
        for name in ('root', 'ee'):
            finite_difference = (after[name + '_pos'] - before[name + '_pos']) / dt
            np.testing.assert_allclose(before[name + '_lin_vel_world'], finite_difference, atol=1e-6)

    def test_pd_physics_and_action_delay(self):
        baseline, changed = Go2Arx5Mujoco(), Go2Arx5Mujoco()
        start = baseline.data.qpos.copy()
        baseline.step(np.zeros(18))
        changed.step(np.ones(18))
        np.testing.assert_array_equal(baseline.data.qpos, changed.data.qpos)
        baseline.step(np.zeros(18))
        changed.step(np.ones(18))
        self.assertGreater(np.linalg.norm(baseline.data.qpos-changed.data.qpos), 1e-5)
        for _ in range(98):
            state = baseline.step(np.zeros(18))
        self.assertAlmostEqual(state['time'], 2., places=10)
        self.assertTrue(np.all(np.isfinite(baseline.data.qpos)))
        self.assertGreater(np.linalg.norm(start-baseline.data.qpos), 1e-3)
        self.assertTrue(np.all(abs(baseline.last_torque) <= baseline.torque_limit))
        print(f'PD diagnostic: simulated {state["time"]:.2f}s, root z={state["root_pos"][2]:.4f}, contacts={baseline.data.ncon}; not learned WBC evidence')

if __name__ == '__main__':
    unittest.main()
