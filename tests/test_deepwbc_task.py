"""CPU task semantics and executable pinned-source parity, not physics tests."""
import ast
from copy import deepcopy
from pathlib import Path
from types import MethodType
import pytest
import torch
import pawcerto.methods.deepwbc.task as module
from pawcerto.methods.deepwbc.task import DeepWBCTask, evaluation_config

torch.set_num_threads(1)
REFERENCE = Path(__file__).resolve().parents[1]/'third_party/deepwbc-reference/legged_gym/legged_gym/envs'


def bound(n=3, config=None):
    task=DeepWBCTask(n,config)
    root=torch.zeros(n,13); root[:,2]=.42; root[:,6]=1
    task.bind_state(root_states=root,dof_pos=torch.zeros(n,20),dof_vel=torch.ones(n,20),
        torques=torch.ones(n,20),actions=torch.ones(n,18),ee_pos=torch.tensor([.5,0,.53]).repeat(n,1),
        ee_orn=torch.tensor([0.,0,0,1]).repeat(n,1),contact_forces=torch.zeros(n,5,3),foot_wrenches=torch.zeros(n,4,6))
    return task


def test_copied_equations_ast_and_executable_reward_parity():
    task=bound()
    original=deepcopy(task)
    local_ast={n.name: n for n in ast.walk(ast.parse(Path(module.__file__).read_text())) if isinstance(n,ast.FunctionDef)}
    count=0
    for path in (REFERENCE/'base/legged_robot.py',REFERENCE/'widowGo1/widowGo1.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node,ast.FunctionDef) or node.name not in local_ast: continue
            if node.name.startswith('_reward_') or node.name in ('compute_reward','collision_check','update_curr_ee_goal','check_termination','_resample_ee_goal','_resample_commands','update_command_curriculum'):
                # Child overrides base methods; compare only actual owner.
                if path.name=='legged_robot.py' and (not node.name.startswith('_reward_') or node.name in ('_reward_torques','_reward_tracking_lin_vel')): continue
                assert ast.dump(node,include_attributes=False)==ast.dump(local_ast[node.name],include_attributes=False)
                ns=vars(module).copy()
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
                setattr(original,node.name,MethodType(ns[node.name],original))
                count+=1
    assert count >= 45
    # Compare compiled original functions, including their metric side effects.
    for name in task.reward_names+task.arm_reward_names:
        torch.testing.assert_close(getattr(task,'_reward_'+name)(),getattr(original,'_reward_'+name)())
    task.compute_reward(); original.compute_reward()
    torch.testing.assert_close(task.rew_buf,original.rew_buf)
    torch.testing.assert_close(task.arm_rew_buf,original.arm_rew_buf)
    for name in task.episode_metric_sums:
        torch.testing.assert_close(task.episode_metric_sums[name],original.episode_metric_sums[name])


def test_dual_rewards_divide_100_without_dt_and_timeout_penalty():
    task=bound()
    task.curr_ee_goal_sphere[:]=torch.tensor([.5,0,0])
    task.commands[:,0]=.5; task.base_lin_vel[:,0]=.5
    task.compute_reward()
    # 12 leg power squares, four hip actions; no foot force; yaw matches.
    assert task.rew_buf[0].item()==pytest.approx((-.00006*12-.01*4+.2+.15+.5*.5)/100)
    assert task.arm_rew_buf[0].item()==pytest.approx((-.004*6+.55)/100)
    task.reset_buf[:]=True; task.time_out_buf[1]=True
    assert task._reward_termination().tolist()==[True,False,True]


def test_goal_interpolation_collision_and_exhaustion_are_original():
    task=bound(1)
    task.ee_start_sphere[:]=torch.tensor([.4,0,0]); task.ee_goal_sphere[:]=torch.tensor([.6,0,0])
    task.traj_timesteps[:]=2; task.traj_total_timesteps[:]=4
    for expected in (.4,.5,.6):
        task.update_curr_ee_goal()
        assert task.curr_ee_goal_sphere[0,0].item()==pytest.approx(expected)
    # Strict interior box, including path samples, rejects this stationary point.
    task.ee_start_sphere[:]=module.cart2sphere(torch.tensor([[.1,0,-.2]]))
    task.ee_goal_sphere[:]=task.ee_start_sphere
    assert task.collision_check(torch.tensor([0])).item()
    calls=[]
    def sample(ids):
        calls.append(ids.clone()); task.ee_goal_sphere[ids]=torch.tensor([.1,0,0])
    task._resample_ee_goal_sphere_once=sample
    task.collision_check=lambda ids: torch.ones(len(ids),dtype=torch.bool)
    task._resample_ee_goal(torch.tensor([0]))
    assert len(calls)==10
    assert task.ee_goal_sphere[0,0].item()==pytest.approx(.1) # no invented fallback


def test_strict_timeout_and_goal_direction_termination_eval_quirk():
    task=bound(3,evaluation_config())
    task.episode_length_buf[:]=task.max_episode_length
    task.curr_ee_goal_sphere[:,2]=torch.tensor([1,-1,0.])
    roll=torch.full((3,),.3)
    task.base_quat[:]=torch.stack((torch.sin(roll/2),roll*0,roll*0,torch.cos(roll/2)),1)
    task.check_termination()
    assert task.time_out_buf.tolist()==[False]*3
    assert task.reset_buf.tolist()==[True,False,True] # eval r_threshold=1 ignored
    task.episode_length_buf+=1; task.check_termination()
    assert task.time_out_buf.all()


def test_reset_preserves_fall_command_and_goal_start_and_priv_randomization():
    task=bound(2)
    task.commands[:]=torch.tensor([.8,0,.7]); task.ee_goal_sphere[:]=torch.tensor([.4,0,.1])
    task.time_out_buf[1]=True
    writes=task.reset(torch.arange(2),default_dof_pos=torch.ones(20),env_origins=torch.zeros(2,3))
    torch.testing.assert_close(task.commands[0],torch.tensor([.8,0,.7]))
    assert task.commands[1].eq(0).all() # initial ranges zero
    torch.testing.assert_close(task.ee_start_sphere,torch.tensor([[.4,0,.1],[.4,0,.1]]))
    assert ((writes['dof_pos']>=.8)&(writes['dof_pos']<=1.2)).all()
    assert writes['dof_vel'].eq(0).all()
    rand=task.sample_creation_randomization()
    assert rand['mass_params'].shape==(2,5)
    assert rand['motor_strength'].shape==(2,18)
    assert rand['friction'].shape==(2,1)
    torch.manual_seed(17)
    offsets=DeepWBCTask(256).sample_creation_randomization()['box_delta_y']
    assert (offsets<0).any() and (offsets>0).any()
    assert ((offsets.abs()>=.1)&(offsets.abs()<=.3)).all()
    assert ((rand['motor_strength']>=.7)&(rand['motor_strength']<=1.3)).all()
    empty=task.reset([],default_dof_pos=torch.ones(20),env_origins=torch.zeros(2,3))
    assert empty['episode']=={} and empty['root_states'].shape==(0,13)
    task.update_command_curriculum()
    assert task.lin_vel_x_ranges.tolist()==[0,.9]


def test_real_external_transition_returns_dual_rewards_without_fabricating_state():
    task=bound(2)
    root=task.root_states.clone()
    task.traj_total_timesteps[:]=1000
    leg,arm,done,extras=task.after_physics()
    assert leg.shape==arm.shape==done.shape==(2,)
    assert torch.isfinite(leg).all() and torch.isfinite(arm).all()
    torch.testing.assert_close(task.root_states,root)
    assert task.episode_length_buf.tolist()==[1,1]
    assert not extras['time_outs'].any()
    task.finish_step()
    assert task.last_actions.eq(1).all()
