"""CPU semantic evidence only: explicit state fixtures are not simulated physics."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from pawcerto.methods.roboduet import default_config, build_models
from pawcerto.methods.roboduet.evaluation import (
    FixedCommands, EpisodeRecorder, evaluate_seed, issued_targets,
    nominal_config, physical_sample, validate_protocol,
)
from pawcerto.methods.roboduet.export import ExportedRoboDuet, export_checkpoint
from pawcerto.methods.roboduet.training.isaac_env import RoboDuetIsaacTrainingEnv

PROTOCOL = Path(__file__).resolve().parents[1]/'configs/roboduet_evaluation.json'


class ExplicitState:
    """Prescribed states with a forced second-control-step fall, no integrator."""
    device, num_envs, dt = 'cpu', 3, .005
    joint_names = [f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR')
                   for joint in ('hip','thigh','calf')] + [f'zarx_j{i}' for i in range(1,9)]
    body_names = ['base','FL_foot','FR_foot','RL_foot','RR_foot','zarx_body6','Head']
    contact_names = body_names
    torque_limits = torch.full((20,),30.)
    dof_pos_limits = torch.tensor([[-3.,3.]]*20)

    def __init__(self, config):
        self.q = torch.tensor([[config['Cfg']['init_state']['default_joint_angles'][name]
                                for name in self.joint_names]]).repeat(3,1)
        self.qd = torch.zeros_like(self.q)
        self.pose = torch.tensor([[0.,0.,.34,1.,0.,0.,0.]]).repeat(3,1)
        self.velocity = torch.zeros(3,6)
        self.substep = 0

    def set_material_properties(self, *args): pass
    def set_base_mass_delta(self, *args, **kwargs): pass
    def set_gravity(self, *args): pass
    def set_stage(self, *args): pass

    def reset(self, ids, joint_pos, root_pose, root_velocity):
        self.q[ids], self.pose[ids], self.velocity[ids] = joint_pos, root_pose, root_velocity
        self.qd[ids] = 0

    def joints(self): return self.q, self.qd

    def step_control(self, leg_torque, arm_target, *, capture_state=True):
        self.substep += 1
        if self.substep == 8:
            self.pose[0,2] = .1

    def state(self):
        body_pos = torch.zeros(3,7,3)
        body_pos[:,0] = self.pose[:,:3]
        body_pos[:,-2] = torch.tensor([.4,0.,.38])
        return dict(root_pos=self.pose[:,:3],root_quat=self.pose[:,3:],
            root_lin_vel=self.velocity[:,:3],root_ang_vel=self.velocity[:,3:],
            joint_pos=self.q.clone(),joint_vel=self.qd.clone(),body_pos=body_pos,
            body_lin_vel=torch.zeros(3,7,3),contact_forces=torch.zeros(3,7,3),
            ee_pos=body_pos[:,-2],ee_quat=torch.tensor([1.,0.,0.,0.]).repeat(3,1),
            ee_lin_vel=torch.zeros(3,3),ee_ang_vel=torch.zeros(3,3))


def setup_env():
    config = nominal_config(default_config(),3)
    env = RoboDuetIsaacTrainingEnv(ExplicitState(config),config)
    env.set_stage(2)
    protocol = json.loads(PROTOCOL.read_text())
    return env, protocol


def test_frozen_commands_preserve_original_clock_and_abg():
    env, protocol = setup_env()
    assert validate_protocol(protocol,env.config) == 1000
    assert protocol['seeds'] == [2027,2028,2029]
    fixed = FixedCommands(env.task,protocol)
    ids = torch.arange(3)
    env.task.reset(ids)
    torch.testing.assert_close(env.task.commands_arm_obs[0,:3],torch.tensor([.5,0.,0.]))
    for _ in range(124): env.task.advance()
    assert fixed.phase == [0,0,0]
    env.task.advance()
    assert fixed.phase == [1,1,1]
    assert env.task.arm_time_buf.tolist() == [0,0,0]
    torch.testing.assert_close(env.task.commands_arm_obs[0,3:6],env.task.quat_to_angle(env.task.obj_quats)[0])
    torch.testing.assert_close(env.task.commands_dog[1,:3],torch.tensor([0.,.2,0.]))
    changed = deepcopy(protocol)
    changed['cases'][0]['arm_lpy_rpy'][0][0] = 8
    with pytest.raises(ValueError, match='outside supported'):
        validate_protocol(changed,env.config)


def test_real_step_callback_copies_terminal_state_before_reset_and_failure_propagates():
    env, protocol = setup_env()
    FixedCommands(env.task,protocol)
    env._reset_indices(torch.arange(3))
    recorder = EpisodeRecorder(['a','b','c'],.02,2)
    target = issued_targets(env.task)
    for _ in range(2):
        env.step(torch.zeros(3,12),torch.zeros(3,6),pre_reset_callback=lambda e,d,t:
                 recorder.append(physical_sample(e,target,d,t)))
    assert recorder.summary()[0]['non_timeout_termination']
    assert recorder.summary()[0]['survival_s'] == .04
    assert recorder.records[0][-1]['root_states'][2].item() == pytest.approx(.1)
    assert env.task.root_states[0,2].item() == pytest.approx(.34)
    # Later autoreset episodes cannot append over a failed first prefix.
    recorder.append(physical_sample(env,target,torch.zeros(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool)))
    assert len(recorder.records[0]) == 2
    def broken(*args): raise RuntimeError('readout failed')
    with pytest.raises(RuntimeError,match='readout failed'):
        env.step(torch.zeros(3,12),torch.zeros(3,6),pre_reset_callback=broken)


def test_geometry_uses_grasp_point_and_target_at_issued_step():
    env, protocol = setup_env()
    FixedCommands(env.task,protocol)
    env.task.reset(torch.arange(3))
    env.runtime.pose[:] = torch.tensor([0.,0.,.34,1.,0.,0.,0.])
    env._refresh_physics()
    done, timeout = env.task.check_termination()
    target = issued_targets(env.task)
    # Advance next target independently: errors must still use issued target.
    env.task._resample_arm_commands(torch.arange(3))
    sample = physical_sample(env,target,done,timeout)
    torch.testing.assert_close(sample['ee_position_error_m'],torch.zeros(3),atol=1e-6,rtol=0)
    torch.testing.assert_close(sample['ee_orientation_error_rad'],torch.zeros(3))
    assert sample['up_dot'].tolist() == [1.,1.,1.]


def test_five_module_actual_evaluator_consumption_and_prefix_output(tmp_path):
    torch.set_num_threads(1)
    env, protocol = setup_env()
    for side in ('DogAC_Args','ArmAC_Args'):
        env.config[side].update(actor_hidden_dims=[24,16,8],critic_hidden_dims=[24,16,8],
                                adaptation_module_branch_hidden_dims=[16])
    dog, arm = build_models(env.config)
    checkpoint = tmp_path/'weights.pt'
    torch.save(dict(format='pawcerto.roboduet.training.v1',config=env.config,
        dog_model_state_dict=dog.state_dict(),arm_model_state_dict=arm.state_dict(),stage=2,next_iteration=10001),checkpoint)
    export_checkpoint(checkpoint,tmp_path/'export',env.runtime.joint_names)
    policy = ExportedRoboDuet(tmp_path/'export')
    # Shorten only this semantic test; the delivered protocol remains 20 s.
    protocol['duration_s'] = .06
    recorder = evaluate_seed(env,policy,protocol)
    result = recorder.save(tmp_path/'result',{'evidence':'CPU explicit-state fixture, no physics'})
    substeps = torch.load(tmp_path/'result/substeps.pt',weights_only=False)['cases']
    failed = substeps['standing_ee_motion']
    assert len(failed) == 8  # four reset and four first-policy substeps
    assert failed[-1]['time_s'] == .04
    assert failed[-1]['root_height'].item() == pytest.approx(.1)
    assert failed[-1]['sent_effort_command'].shape == (20,)
    assert len(substeps['locomotion_fixed_ee']) == 16
    assert result['cases'][0]['samples'] == 1
    assert result['cases'][0]['non_timeout_termination']
    assert result['cases'][1]['samples'] == 3
    assert result['cases'][1]['full_duration']
    saved = torch.load(tmp_path/'result/policy_inputs_outputs.pt',weights_only=False)
    assert [s['step'] for s in saved] == [0,2]
    for entry in saved:
        torch.testing.assert_close(entry['arm_output'],arm.act_inference(entry['arm_input'],mode='official_play'))
        torch.testing.assert_close(entry['dog_output'],dog.act_inference(entry['dog_input']))
        # Body guidance reaches the dog observation before the dog graph.
        assert entry['dog_input']['obs'].shape == (3,56)


def test_initialization_failure_is_invalid_prefix():
    env, protocol = setup_env()
    FixedCommands(env.task,protocol)
    env.task.reset(torch.arange(3))
    env.runtime.pose[:,2] = .1
    env._refresh_physics()
    done, timeout = env.task.check_termination()
    recorder = EpisodeRecorder(['a','b','c'],.02,1000)
    recorder.append(physical_sample(env,issued_targets(env.task),done,timeout),initialization=True)
    assert not any(recorder.active)
    assert all(row['invalid_prefix'] and not row['full_duration'] for row in recorder.summary())
