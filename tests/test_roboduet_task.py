"""CPU semantic comparisons against pinned upstream bodies, without Isaac Gym."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from pawcerto.methods.roboduet.config import default_config
from pawcerto.methods.roboduet.training import task as port

REFERENCE = Path(__file__).resolve().parents[1] / "third_party/roboduet-reference/go1_gym"


def source_class(path, name, namespace, methods=None):
    """Execute the actual pinned class bodies; omit simulator imports only."""
    tree = ast.parse((REFERENCE / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.bases = [] if name == "LeggedRobot" else node.bases
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    code = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(REFERENCE / path), "exec"), namespace)
    return namespace[name]


def task_fixture(n=16):
    torch.manual_seed(721)
    cfg = default_config()
    cfg["stage"] = {"pretrained_to_hybrid_start": 2, "pretrained_to_hybrid_end": 4}
    t = port.RoboDuetTask(cfg, n)
    root = torch.randn(n, 13) * 0.2
    root[:, 2] = 0.38 + torch.linspace(-0.1, 0.1, n)
    root[:, 3:7] = port.quat_from_euler_xyz(*(torch.randn(n) * 0.3 for _ in range(3)))
    ee = root.clone()
    ee[:, :3] += torch.randn(n, 3) * 0.3
    ee[:, 3:7] = port.quat_from_euler_xyz(*(torch.randn(n) for _ in range(3)))
    fields = {k: torch.randn(n, 20) * 0.2 for k in
              ("dof_pos", "dof_vel", "torques", "last_dof_vel", "joint_pos_target",
               "last_joint_pos_target", "last_last_joint_pos_target")}
    fields.update({k: torch.randn(n, 18) * 0.2 for k in ("actions", "last_actions", "last_last_actions")})
    fields.update({k: torch.randn(n, 2) * 0.2 for k in ("plan_actions", "last_plan_actions")})
    fields.update(root_states=root, end_effector_state=ee,
                  foot_positions=torch.randn(n, 4, 3) * 0.2,
                  foot_velocities=torch.randn(n, 4, 3) * 0.2,
                  contact_forces=torch.randn(n, 8, 3) * 4,
                  feet_indices=torch.arange(4), penalised_contact_indices=torch.tensor([4, 5]),
                  dof_pos_limits=torch.tensor([[-0.1, 0.1]]).repeat(20, 1),
                  default_dof_pos=torch.zeros(1, 20))
    t.update_physics(**fields)
    t.reset(torch.arange(n))
    t.advance()
    return t


def upstream_task(t):
    namespace = dict(vars(port))
    switch = SimpleNamespace(switch_open=t.switch_open, get_reward_scales=t.get_reward_scales)
    namespace["global_switch"] = switch
    namespace["LeggedRobot"] = object
    rewards = source_class("envs/rewards/rewards.py", "Rewards", namespace)
    selected = {"compute_reward", "check_termination", "quat_to_angle", "get_lpy_in_base_coord",
                "get_alpha_beta_gamma_in_base_coord", "_resample_arm_commands", "_resample_Traj_commands",
                "_resample_commands", "_step_contact_targets"}
    robot = source_class("envs/automatic/legged_robot.py", "LeggedRobot", namespace, selected)
    original = robot()
    original.__dict__.update(deepcopy(t.__dict__))
    original.reward_container = rewards(original)
    original.reward_functions = [getattr(original.reward_container, "_reward_" + k) for k in original.reward_names]
    return original


class TaskSemanticsTests(unittest.TestCase):
    def test_every_active_reward_and_aggregate_matches_source(self):
        for iteration in (0, 2, 3, 4):
            t = task_fixture()
            t.set_iteration(iteration)
            if t.switch_open:
                t._resample_arm_commands(torch.arange(t.num_envs))
            t.check_termination()
            original = upstream_task(t)
            for name in t.reward_names:
                expected = getattr(original.reward_container, "_reward_" + name)()
                actual = getattr(t.reward_container, "_reward_" + name)()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
            t.compute_reward()
            original.compute_reward()
            for key in ("rew_buf_dog", "rew_buf_arm", "rew_buf_pos_dog", "rew_buf_neg_dog"):
                torch.testing.assert_close(getattr(t, key), getattr(original, key), rtol=0, atol=0)
            for key in t.command_sums:
                torch.testing.assert_close(t.command_sums[key], original.command_sums[key], rtol=0, atol=0)
            self.assertEqual(len(t.reward_names), 33)

    def test_arm_sampling_and_target_geometry_matches_source(self):
        t = task_fixture()
        t.set_iteration(4)
        original = upstream_task(t)
        ids = torch.arange(t.num_envs)
        torch.manual_seed(19)
        np.random.seed(29)
        t._resample_arm_commands(ids)
        torch.manual_seed(19)
        np.random.seed(29)
        original._resample_arm_commands(ids)
        for key in ("commands_arm", "commands_arm_obs", "target_abg", "obj_quats", "T_trajs", "arm_time_buf"):
            torch.testing.assert_close(getattr(t,key), getattr(original,key), rtol=0, atol=0)
        torch.testing.assert_close(t.get_lpy_in_base_coord(ids), original.get_lpy_in_base_coord(ids), rtol=0, atol=0)
        torch.testing.assert_close(t.get_alpha_beta_gamma_in_base_coord(ids), original.get_alpha_beta_gamma_in_base_coord(ids), rtol=0, atol=0)

    def test_commands_and_standing_gait_match_source(self):
        t = task_fixture()
        original = upstream_task(t)
        ids = torch.arange(t.num_envs)
        for key in t.command_sums:
            t.command_sums[key].fill_(1.0)
            original.command_sums[key].fill_(1.0)
        torch.manual_seed(15)
        t._resample_commands(ids)
        torch.manual_seed(15)
        original._resample_commands(ids)
        torch.testing.assert_close(t.commands_dog, original.commands_dog, rtol=0, atol=0)
        t.commands_dog[:5] = 0
        original.commands_dog[:5] = 0
        t._step_contact_targets()
        original._step_contact_targets()
        for key in ("foot_indices", "clock_inputs", "desired_contact_states", "gait_indices"):
            torch.testing.assert_close(getattr(t,key), getattr(original,key), rtol=0, atol=0)
        torch.testing.assert_close(t.clock_inputs[:5], torch.ones(5,4))

    def test_curriculum_matches_source_and_checkpoint_continues(self):
        namespace = {"np": np, "torch": torch}
        source_class("envs/automatic/curriculum.py", "Curriculum", namespace)
        cls = source_class("envs/automatic/curriculum.py", "RewardThresholdCurriculum", namespace)
        args = dict(seed=31, x=(-1,1,5), y=(-1,1,3))
        old, new = cls(**args), port.RewardThresholdCurriculum(**args)
        for c in (old,new):
            c.set_to(np.array([-.5,-.5]), np.array([.5,.5]))
            c.update(np.array([5,8]), [torch.tensor([0.9,0.1])], [0.8], local_range=np.array([.5,.5]))
        np.testing.assert_array_equal(old.weights, new.weights)
        for a,b in zip(old.sample(24), new.sample(24)):
            np.testing.assert_array_equal(a,b)
        t = task_fixture()
        t.set_iteration(4)
        t._resample_arm_commands(torch.arange(t.num_envs))
        t.check_termination()
        t.compute_reward()
        checkpoint = t.state_dict()
        restored = task_fixture()
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.switch_open, t.switch_open)
        torch.manual_seed(11)
        np.random.seed(9)
        t.reset(torch.arange(t.num_envs))
        torch.manual_seed(11)
        np.random.seed(9)
        restored.reset(torch.arange(t.num_envs))
        for name in t._state_names:
            torch.testing.assert_close(getattr(t,name), getattr(restored,name), rtol=0, atol=0, msg=name)
        np.testing.assert_array_equal(t.curricula[0].weights, restored.curricula[0].weights)

    def test_stage_timing_and_termination_boundaries_match_source(self):
        t = task_fixture(4)
        namespace = {"np": np}
        switch_cls = source_class("utils/global_switch.py", "GlobalSwitch", namespace)
        namespace["sigmoid"] = lambda x: 1/(1+np.exp(-x))
        switch = switch_cls()
        switch.pretrained_to_hybrid_start = t.pretrained_to_hybrid_start
        switch.pretrained_to_hybrid_end = t.pretrained_to_hybrid_end
        switch.init_sigmoid_lr()
        switch.set_reward_scales(t.hybrid_reward_scales,t.pretrained_reward_scales)
        for count in range(6):
            switch.count = count
            t.set_iteration(count)
            self.assertEqual(t.get_reward_scales(), switch.get_reward_scales())
            self.assertEqual(t.switch_open, count > 2)
        t.T_trajs.fill_(2)
        t.arm_time_buf[:] = torch.tensor([60,61,61,61])
        t.episode_length_buf[:] = torch.tensor([0,0,t.max_episode_length,t.max_episode_length+1])
        t.root_states[:,2] = 0.38
        t.root_states[:,3:7] = port.quat_from_euler_xyz(torch.zeros(4),torch.ones(4)*-.5,torch.zeros(4))
        t.commands_arm[:,0] = .5
        t.commands_arm[:,1] = -.5
        original = upstream_task(t)
        t.check_termination()
        original.check_termination()
        torch.testing.assert_close(t.reset_buf,original.reset_buf)
        torch.testing.assert_close(t.time_out_buf,original.time_out_buf)
        self.assertEqual(t.time_out_buf.tolist(), [False,False,False,True])
        self.assertEqual(t.reverse_buf.tolist(), [False,True,True,True])


if __name__ == "__main__":
    unittest.main()
