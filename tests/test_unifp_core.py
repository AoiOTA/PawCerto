"""CPU tensor parity only; these fixtures do not simulate or establish learning."""
import ast
import copy
import inspect
from pathlib import Path
import unittest

import torch
from pawcerto.methods.unifp import B2Z1PosForceRoughCfg, UniFPEquations
from pawcerto.methods.unifp.history import initialize_history, reset_history, force_stage_active
from pawcerto.methods.unifp import equations
from pawcerto.methods.unifp.math import quat_apply, quat_rotate_inverse, quat_from_euler_xyz

REFERENCE = Path(__file__).resolve().parents[1] / 'third_party/unifp-reference'


def fixture(cls=UniFPEquations):
    s = cls()
    s.cfg = B2Z1PosForceRoughCfg()
    s.num_envs, s.num_actions, s.device, s.dt = 2, 17, 'cpu', .02
    s.obs_scales = s.cfg.normalization.obs_scales
    s.root_states = torch.zeros(2, 13)
    s.root_states[:, 2] = .6
    s.root_states[:, 6] = 1
    s.base_pos = s.root_states[:, :3]
    s.base_quat = s.root_states[:, 3:7]
    s.base_yaw_quat = s.base_quat.clone()
    s.base_euler_xyz = torch.zeros(2, 3)
    s.base_lin_vel = torch.tensor([[.2, -.1, .05], [0., .1, 0.]])
    s.base_ang_vel = torch.zeros(2, 3)
    s.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    names = [f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
    names += ['z1_waist','z1_shoulder','z1_elbow','z1_wrist_angle','z1_forearm_roll','z1_wrist_rotate','z1_jointGripper']
    s.default_dof_pos = torch.tensor([[s.cfg.init_state.default_joint_angles[n] for n in names]])
    s.dof_pos = s.default_dof_pos.repeat(2, 1)
    s.dof_vel = torch.zeros(2, 19)
    s.default_dof_pos_wo_gripper = s.default_dof_pos[:, :17]
    s.dof_pos_wo_gripper, s.dof_vel_wo_gripper = s.dof_pos[:, :17], s.dof_vel[:, :17]
    s.dof_pos_limits = torch.tensor([[-3., 3.]]).repeat(19, 1)
    s.motor_strength = torch.ones(2, 17)
    s.p_gains = torch.tensor([next(v for k,v in s.cfg.control.stiffness.items() if k in n) for n in names[:17]])
    s.d_gains = torch.tensor([next(v for k,v in s.cfg.control.damping.items() if k in n) for n in names[:17]])
    s.torque_limits = torch.full((19,), 100.)
    s.actions = torch.zeros(2, 17)
    s.last_actions = torch.zeros_like(s.actions)
    s.last_dof_vel = torch.zeros_like(s.dof_vel)
    s.torques = torch.zeros(2, 19)
    s.gait_indices = torch.tensor([.1, .7])
    s.commands = torch.zeros(2, 15)
    s.commands[0, 0] = .2
    o = s.obs_scales
    s.commands_scale = torch.tensor([o.lin_vel,o.lin_vel,o.ang_vel,o.ee_sphe_radius_cmd,o.ee_sphe_pitch_cmd,o.ee_sphe_yaw_cmd,o.end_effector_roll_cmd,o.end_effector_pitch_cmd,o.end_effector_yaw_cmd,*([o.ee_force]*3),*([o.base_force]*3)])
    s.arm_base_offset = torch.tensor([[.2, 0., .225]]).repeat(2,1)
    s.ee_goal_center_offset = torch.tensor([[.2, 0., .8]]).repeat(2,1)
    s.ee_pos = torch.tensor([[.9, .1, .8], [.8, -.1, .9]])
    s.curr_ee_goal_cart_world = torch.tensor([[.9, 0., .8]]).repeat(2,1)
    s.gripper_idx, s.robot_base_idx = 8, 9
    s.feet_indices = torch.arange(4)
    s.thigh_indices = torch.arange(4,8)
    s.hip_indices = torch.tensor([0,3,6,9])
    s.penalised_contact_indices = torch.tensor([4,5,6,7,9])
    s.termination_contact_indices = torch.tensor([], dtype=torch.long)
    s.contact_forces = torch.zeros(2,10,3)
    s.contact_forces[:, :4, 2] = 100.
    s.forces = torch.zeros_like(s.contact_forces)
    s.forces_local = torch.zeros_like(s.forces)
    s.current_Fxyz_gripper_cmd = torch.zeros(2,3)
    s.current_Fxyz_base_cmd = torch.zeros(2,3)
    s.gripper_force_kps = torch.full((2,3), 200.)
    s.base_force_kds = torch.full((2,3), 200.)
    s.mass_params_tensor = torch.zeros(2,22)
    s.friction_coeffs_tensor = torch.ones(2,1)
    s.rigid_state = torch.zeros(2,10,13)
    s.last_contacts = torch.zeros(2,4,dtype=torch.bool)
    s.feet_air_time = torch.zeros(2,4)
    s.episode_length_buf = torch.zeros(2,dtype=torch.long)
    s.max_episode_length = 1000
    s.rew_buf = torch.zeros(2)
    s.reward_scales = {k:getattr(s.cfg.rewards.scales,k) for k in dir(s.cfg.rewards.scales) if not k.startswith('_')}
    s._prepare_reward_function()
    s.noise_scale_vec = s._get_noise_scale_vec(s.cfg)
    s.add_noise = False
    initialize_history(s)
    return s


def upstream_equations():
    path = REFERENCE / 'legged_gym/envs/b2/legged_robot_b2z1_pos_force.py'
    if not path.exists():
        raise unittest.SkipTest('Fetch pinned UniFP reference for independent upstream parity')
    source = ast.parse(path.read_text())
    upstream = next(n for n in source.body if isinstance(n,ast.ClassDef))
    names = {k for k,v in UniFPEquations.__dict__.items() if callable(v)}
    upstream.bases = []
    upstream.body = [n for n in upstream.body if isinstance(n,ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[upstream],type_ignores=[])
    namespace = dict(vars(equations))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[upstream.name]


class UniFPCoreTest(unittest.TestCase):
    def test_observations_supervision_history_and_rewards_match_upstream(self):
        local, upstream = fixture(), fixture(upstream_equations())
        for s in (local, upstream):
            s.forces[:,8,0] = 40.
            s.current_Fxyz_gripper_cmd[:,0] = -20.
            s.compute_observations()
            s.check_termination()
            s.compute_reward()
        for attr in ('obs_buf','privileged_obs_buf','obs_pred','rew_buf','reset_buf'):
            torch.testing.assert_close(getattr(local,attr),getattr(upstream,attr),rtol=0,atol=0)
        self.assertEqual(local.obs_buf.shape,(2,2336))
        self.assertEqual(local.privileged_obs_buf.shape,(2,447))
        torch.testing.assert_close(local.obs_pred[:,6],torch.full((2,),.4))
        torch.testing.assert_close(local.obs_pred[:,9:],torch.zeros(2,3))
        self.assertEqual(torch.count_nonzero(local.obs_buf[:,:-73]),0)
        self.assertEqual(len(local.reward_names),27)
        reset_history(local,torch.tensor([0]))
        local.compute_observations()
        self.assertEqual(torch.count_nonzero(local.obs_buf[0,:-73]),0)
        self.assertGreater(torch.count_nonzero(local.obs_buf[1,:-73]),0)

    def test_pd_and_force_offset_equations(self):
        s = fixture()
        a = torch.ones(2,17)
        s.motor_strength[:] = .9
        t = s._compute_torques(a)
        torch.testing.assert_close(t[:,:17],torch.clamp(s.p_gains*.9*.25,max=100).repeat(2,1))
        torch.testing.assert_close(t[:,17:],torch.zeros(2,2))
        s.forces[:,8,0] = 40
        s.current_Fxyz_gripper_cmd[:,0] = -20
        s.ee_pos[:] = s.curr_ee_goal_cart_world + torch.tensor([.1,0,0])
        torch.testing.assert_close(s._reward_tracking_ee_force_world(),torch.ones(2))
        # Actual contact forces are a separate observation source, not these labels.
        s.contact_forces[:,8,0] = 999
        s.compute_observations()
        torch.testing.assert_close(s.obs_pred[:,6],torch.full((2,),.4))

    def test_quaternion_frame_and_stage_boundary(self):
        zero = torch.zeros(2)
        q = quat_from_euler_xyz(zero,zero,torch.full((2,),torch.pi/2))
        v = torch.tensor([[1.,0,0]]).repeat(2,1)
        torch.testing.assert_close(quat_apply(q,v),torch.tensor([[0.,1,0]]).repeat(2,1),atol=2e-7,rtol=0)
        torch.testing.assert_close(quat_rotate_inverse(q,quat_apply(q,v)),v,atol=3e-7,rtol=0)
        cfg = B2Z1PosForceRoughCfg()
        self.assertFalse(force_stage_active(192000,cfg))
        self.assertTrue(force_stage_active(192001,cfg))

    def test_goal_interpolation_collision_and_termination(self):
        s = fixture()
        s.curr_ee_goal_sphere = torch.zeros(2,3)
        s.curr_ee_goal_cart = torch.zeros(2,3)
        s.ee_start_sphere = torch.tensor([[.4,0,0]]).repeat(2,1)
        s.ee_goal_sphere = torch.tensor([[.8,0,0]]).repeat(2,1)
        s.ee_goal_orn_delta_rpy = torch.zeros(2,3)
        s.goal_timer = torch.full((2,),5.)
        s.traj_timesteps = torch.full((2,),10.)
        s.traj_total_timesteps = torch.full((2,),20.)
        s.stop_update_goal = False
        s.update_curr_ee_goal()
        torch.testing.assert_close(s.curr_ee_goal_sphere,torch.tensor([[.6,0,0]]).repeat(2,1))
        torch.testing.assert_close(s.curr_ee_goal_cart_world,torch.tensor([[.8,0,.8]]).repeat(2,1))
        torch.testing.assert_close(s.commands[:,3:6],s.curr_ee_goal_sphere)
        s.num_collision_check_samples = 10
        s.collision_check_t = torch.linspace(0,1,10)[None,None,:]
        s.collision_lower_limits = torch.tensor(s.cfg.goal_ee.collision_lower_limits)
        s.collision_upper_limits = torch.tensor(s.cfg.goal_ee.collision_upper_limits)
        s.underground_limit = s.cfg.goal_ee.underground_limit
        self.assertFalse(s.collision_check(torch.arange(2)).any())
        s.ee_start_sphere[:] = torch.tensor([1.,-torch.pi/2,0])
        s.ee_goal_sphere[:] = s.ee_start_sphere
        self.assertTrue(s.collision_check(torch.arange(2)).all())
        s.base_euler_xyz[0,0] = .81
        s.episode_length_buf[1] = 1001
        s.check_termination()
        self.assertEqual(s.reset_buf.tolist(),[True,True])
        self.assertEqual(s.time_out_buf.tolist(),[False,True])

    def test_all_copied_method_bodies_are_upstream(self):
        upstream_equations()  # checks whether the pinned reference is available
        path = REFERENCE / 'legged_gym/envs/b2/legged_robot_b2z1_pos_force.py'
        original = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
        copied = next(n for n in ast.parse(inspect.getsource(equations)).body if isinstance(n,ast.ClassDef))
        methods = {n.name:ast.dump(n) for n in original.body if isinstance(n,ast.FunctionDef)}
        for node in copied.body:
            if isinstance(node,ast.FunctionDef):
                self.assertEqual(ast.dump(node),methods[node.name],node.name)

    def test_force_ramp_hold_release_upstream_parity(self):
        local, original = fixture(), fixture(upstream_equations())
        for s in (local, original):
            s.settling_time_force_gripper = 2
            for kind in ('cmd','ext'):
                setattr(s,f'freed_envs_gripper_{kind}',torch.zeros(2,dtype=torch.bool))
                setattr(s,f'selected_env_ids_gripper_{kind}',torch.ones(2,dtype=torch.int32))
                setattr(s,f'push_interval_gripper_{kind}',torch.full((2,1),100))
                setattr(s,f'push_end_time_gripper_{kind}',torch.full((2,),4.))
                setattr(s,f'push_duration_gripper_{kind}',torch.full((2,),4.))
                setattr(s,f'force_target_gripper_{kind}',torch.full((2,3),40.))
                setattr(s,f'push_interval_gripper_{kind}_min',100)
                setattr(s,f'push_interval_gripper_{kind}_max',101)
        # Strict inequalities leave the previous sample at the plateau boundary.
        for step, expected in [(1,10.),(2,20.),(3,30.),(4,30.),(6,30.),(7,30.),(8,20.),(9,10.),(10,0.)]:
            for s in (local,original):
                s.episode_length_buf[:] = step
                s._push_gripper(torch.arange(2))
            torch.testing.assert_close(local.current_Fxyz_gripper_cmd,original.current_Fxyz_gripper_cmd,rtol=0,atol=0)
            torch.testing.assert_close(local.forces,original.forces,rtol=0,atol=0)
            torch.testing.assert_close(local.current_Fxyz_gripper_cmd,torch.full((2,3),expected))
            torch.testing.assert_close(local.forces[:,8],torch.full((2,3),expected))



if __name__ == '__main__':
    unittest.main()
