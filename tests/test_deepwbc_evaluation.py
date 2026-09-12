"""CPU-only interface evidence. Runtime doubles do not integrate physics."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from pawcerto.methods.deepwbc.config import default_config
from pawcerto.methods.deepwbc.evaluation import (
    fixed_task_config, transition_snapshot, TrajectoryWriter, evaluate_fixed_policy)
from pawcerto.methods.deepwbc.policy import build_model
from pawcerto.methods.deepwbc.training.isaac_env import DeepWBCIsaacTrainingEnv
from test_deepwbc_env import InterfaceRuntime

ROOT = Path(__file__).resolve().parents[1]
TASK = dict(commands=[.5, 0, 0], ee_goal_lpy=[.6, -.3, .3], traj_time_s=2., hold_time_s=1.)


def cfg():
    c = fixed_task_config(default_config(), TASK, num_envs=2, profile='checkpoint')
    c['domain_rand']['push_robots'] = False
    c['terrain']['init_vel_perturb_range'] = 0
    return c


def test_fixed_task_uses_original_sampler_deadband_and_goal_interpolation():
    c = cfg()
    assert c['terrain'] == default_config()['terrain'] | {'init_vel_perturb_range': 0}
    env = DeepWBCIsaacTrainingEnv(InterfaceRuntime(c), c)
    env.update_command_curriculum()
    env.reset()
    torch.testing.assert_close(env.task.commands, torch.tensor([[.5,0,0]]).repeat(2,1))
    torch.testing.assert_close(env.task.ee_goal_sphere, torch.tensor([TASK['ee_goal_lpy']]).repeat(2,1))
    assert env.task.curr_ee_goal_sphere.eq(0).all() # zero-start, not a teleported target
    env.step(torch.zeros(2,18))
    torch.testing.assert_close(env.task.curr_ee_goal_sphere, env.task.ee_goal_sphere/100)
    task = dict(TASK, commands=[.3,0,0])
    c = fixed_task_config(default_config(), task, num_envs=2)
    env = DeepWBCIsaacTrainingEnv(InterfaceRuntime(c), c)
    env.update_command_curriculum(); env.reset()
    assert env.task.commands.eq(0).all() # exact upstream strict deadband
    with pytest.raises(ValueError, match='lateral'):
        fixed_task_config(c, dict(TASK, commands=[.5,.1,0]))
    with pytest.raises(ValueError, match='collision|underground'):
        fixed_task_config(c, dict(TASK, ee_goal_lpy=[1.,1.5,0]))


def test_snapshot_preserves_terminal_state_and_does_not_change_rewards_or_history():
    c = cfg()
    torch.manual_seed(3)
    e = DeepWBCIsaacTrainingEnv(InterfaceRuntime(c), c); e.reset()
    # Both can terminate with different reasons in one vectorized transition.
    e.runtime.root[0,2] = .1
    e.task.episode_length_buf[1] = e.task.max_episode_length
    captured = []
    def callback(env, leg, arm, done, extras):
        before = {k: v.clone() for k,v in env.task.episode_metric_sums.items()}
        history = env.observer.history.clone()
        captured.append(transition_snapshot(env, leg, arm, done, extras))
        for key,value in before.items():
            torch.testing.assert_close(value, env.task.episode_metric_sums[key])
        torch.testing.assert_close(history, env.observer.history)
    _, _, leg, arm, done, _ = e.step(torch.zeros(2,18), transition_callback=callback)
    row = captured[0]
    assert done.all() and row['root_states'][0,2] == pytest.approx(.1)
    assert e.task.root_states[0,2] == pytest.approx(.42)
    assert row['reason_mask'].tolist() == [[False,False,False,True,False], [False,False,False,False,True]]
    np.testing.assert_array_equal(row['reward_leg'], leg.numpy())
    np.testing.assert_array_equal(row['reward_arm'], arm.numpy())
    assert row['episode_length_buf'][1] == 501 and e.episode_length_buf.eq(0).all()
    e.task.root_states.fill_(0)
    assert row['root_states'][0,2] == pytest.approx(.1) # real independent clone


def test_goal_signed_roll_reason_ignores_unused_config_thresholds():
    c = cfg(); c['termination']['r_threshold'] = 1
    e = DeepWBCIsaacTrainingEnv(InterfaceRuntime(c), c); e.reset()
    e.runtime.root[:,3] = np.sin(.3/2); e.runtime.root[:,6] = np.cos(.3/2)
    rows=[]
    e.step(torch.zeros(2,18), transition_callback=lambda *args: rows.append(transition_snapshot(*args)))
    assert rows[0]['reason_mask'][:,1].all()
    assert rows[0]['done'].all()


def test_cpu_recording_real_observations_export_comparison_and_partial_episode(tmp_path):
    from pawcerto.methods.deepwbc.export import load_checkpoint, export_policy
    c = cfg(); c['env']['episode_length_s'] = .04
    torch.manual_seed(5)
    model = build_model(c).eval()
    checkpoint=tmp_path/'fixture.pt'
    torch.save(dict(model_state_dict=model.state_dict(), config=c, iter=0), checkpoint)
    loaded=load_checkpoint(checkpoint)
    export_policy(loaded, tmp_path/'export')
    jit=torch.jit.load(str(tmp_path/'export'/'policy.pt')).eval()
    writer=TrajectoryWriter(tmp_path/'evaluation', {'evidence_source':'cpu_interface_double'}, chunk_steps=2)
    e=DeepWBCIsaacTrainingEnv(InterfaceRuntime(c), c)
    summary=evaluate_fixed_policy(e, lambda obs: model.act_inference(obs, hist_encoding=True),
        steps=5, writer=writer, exported_policy=jit)
    assert summary['status']=='completed' and summary['evidence_source']=='cpu_interface_double'
    assert summary['completed_episodes']==4 and summary['failure_events']==0
    assert summary['chunks']==3 and summary['steps']==5
    first=np.load(tmp_path/'evaluation'/'trajectory_00000.npz')
    assert first['observations'].shape==(2,2,860)
    assert first['policy_actions'].shape==(2,2,18)
    assert first['reason_mask'][-1,:,-1].all()
    assert first['export_action_max_abs_error'].max()<1e-6
    # Input goal is previous step's goal, terminal record is post-physics goal.
    np.testing.assert_allclose(first['observations'][0,:,70:73], 0)
    assert first['curr_ee_goal_sphere'][0].any()
    events=[json.loads(line) for line in (tmp_path/'evaluation'/'episodes.jsonl').read_text().splitlines()]
    assert all(event['reasons']==['timeout'] for event in events)
    with pytest.raises(FileExistsError):
        TrajectoryWriter(tmp_path/'evaluation', {'evidence_source':'cpu_interface_double'})


def test_nonfinite_transition_and_policy_failure_are_preserved(tmp_path):
    c=cfg(); e=DeepWBCIsaacTrainingEnv(InterfaceRuntime(c), c)
    # Invalid policy cannot reach push_actions after reset warmup.
    writer=TrajectoryWriter(tmp_path/'bad_policy', {'evidence_source':'cpu_interface_double'})
    with pytest.raises(FloatingPointError, match='policy action'):
        evaluate_fixed_policy(e, lambda obs: torch.full((2,18),float('nan')), steps=2, writer=writer)
    report=json.loads((tmp_path/'bad_policy'/'summary.json').read_text())
    assert report['status']=='failed' and report['steps']==0
    assert e.runtime.calls.count('push_actions')==1
    writer=TrajectoryWriter(tmp_path/'bad_state', {'evidence_source':'cpu_interface_double'})
    def corrupt(obs):
        e.runtime.q[0,0]=float('nan')
        return torch.zeros(2,18)
    with pytest.raises(FloatingPointError, match='Nonfinite transition saved'):
        evaluate_fixed_policy(e, corrupt, steps=2, writer=writer)
    row=np.load(tmp_path/'bad_state'/'trajectory_00000.npz')
    assert not row['finite'][0,0] and np.isnan(row['dof_pos'][0,0,0])
    report=json.loads((tmp_path/'bad_state'/'summary.json').read_text())
    assert report['status']=='failed' and 'FloatingPointError' in report['error']
    assert 'nonfinite_transition' in (tmp_path/'bad_state'/'episodes.jsonl').read_text()


def test_cli_preflight_without_simulator_imports(tmp_path):
    c=cfg(); model=build_model(c)
    checkpoint=tmp_path/'fixture.pt'; task=tmp_path/'task.json'; usd=tmp_path/'fixture.usd'
    torch.save(dict(model_state_dict=model.state_dict(), config=c, iter=0), checkpoint)
    task.write_text(json.dumps(TASK)); usd.write_text('#usda 1.0\n') # existence fixture, NOT robot verification
    from pawcerto.isaac.deepwbc_runtime import SIM_DOF_NAMES
    # Minimal metadata fixtures exercise preflight reads, not USD composition.
    (tmp_path/'source-manifest.json').write_text(json.dumps(dict(
        limits={name: dict(effort=100.,velocity=10.) for name in SIM_DOF_NAMES})))
    (tmp_path/'merged.urdf').write_text('<robot name="fixture"><link name="base"/></robot>')
    guard="""
import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('isaaclab','isaacgym','omni'):
            raise AssertionError('Simulator imported in CPU preflight: '+fullname)
sys.meta_path.insert(0,Guard())
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    args=[str(ROOT/'scripts/eval_deepwbc.py'), '--checkpoint',str(checkpoint),'--task',str(task),
        '--robot-usd',str(usd),'--force-mode','native_joint_wrench','--output',str(tmp_path/'preflight'),
        '--steps','3','--preflight']
    result=subprocess.run([sys.executable,'-c',guard,*args],capture_output=True,text=True)
    assert result.returncode==0, result.stdout+result.stderr
    report=json.loads((tmp_path/'preflight'/'preflight.json').read_text())
    assert report['physics_run'] is False and report['evaluation_success'] is False
    assert report['config']['env']['num_envs']==1
    # A pointer alias resolves to the same checkpoint asset, not a different
    # hash merely because the pointer text and physical USD bytes differ.
    import hashlib
    c['pawcerto_runtime']=dict(robot_usd=str(usd),robot_usd_sha256=hashlib.sha256(usd.read_bytes()).hexdigest(),
        force_mode='native_joint_wrench')
    torch.save(dict(model_state_dict=model.state_dict(), config=c, iter=0), checkpoint)
    pointer=tmp_path/'usd_path.txt';pointer.write_text(str(usd)+'\n')
    args[args.index('--robot-usd')+1]=str(pointer)
    args[args.index('--output')+1]=str(tmp_path/'pointer_preflight')
    result=subprocess.run([sys.executable,'-c',guard,*args],capture_output=True,text=True)
    assert result.returncode==0, result.stdout+result.stderr
    alias=json.loads((tmp_path/'pointer_preflight'/'preflight.json').read_text())['config']['pawcerto_runtime']
    assert alias['robot_usd']==str(usd.resolve())
    assert alias['robot_usd_sha256']==c['pawcerto_runtime']['robot_usd_sha256']
    (tmp_path/'merged.urdf').unlink()
    args[args.index('--output')+1] = str(tmp_path/'missing_asset')
    result=subprocess.run([sys.executable,'-c',guard,*args],capture_output=True,text=True)
    assert result.returncode != 0 and 'merged.urdf' in result.stderr
    assert not (tmp_path/'missing_asset').exists()
    # Runtime keyword compatibility is checked without constructing it.
    import inspect
    from pawcerto.isaac.deepwbc_runtime import Go1WidowXIsaac
    inspect.signature(Go1WidowXIsaac).bind(num_envs=1,device='cpu',usd_path=usd,config=c,
        force_signal='native_joint_wrench',terrain_cfg=None,terrain_contact_paths=[],env_origins=torch.zeros(1,3))


@pytest.mark.parametrize('failure_type', [None, RuntimeError, KeyboardInterrupt])
def test_cli_real_main_with_interface_double_preserves_shutdown_status(tmp_path, monkeypatch, capsys, failure_type):
    from types import ModuleType, SimpleNamespace
    from pawcerto.methods.deepwbc import terrain
    from pawcerto.isaac import deepwbc_runtime
    spec=importlib.util.spec_from_file_location('deepwbc_eval_cli', ROOT/'scripts/eval_deepwbc.py')
    cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    c=cfg(); c['pawcerto_runtime']=dict(robot_usd='unused.usda',force_mode='native_joint_wrench',
        force_signal_parity='native incoming reaction; differs from original Gym sensor')
    loaded=SimpleNamespace(model=build_model(c).eval(), checkpoint_path=tmp_path/'fixture.pt',
        checkpoint_sha256='cpu-fixture',iteration=0)
    monkeypatch.setattr(cli, 'prepare', lambda args: (loaded,c,TASK,None))
    seen={}
    class Launcher:
        def __init__(self, **kwargs):
            seen.update(kwargs); self.app=SimpleNamespace(close=self.close)
        def close(self, *, exit_code):
            seen['exit_code']=exit_code
            raise SystemExit(exit_code) # actual Kit may replace exception propagation
    module=ModuleType('isaaclab.app'); module.AppLauncher=Launcher
    monkeypatch.setitem(sys.modules, 'isaaclab.app', module)
    def build(**kwargs):
        if failure_type is not None:
            raise failure_type('preserved startup failure')
        return None
    monkeypatch.setattr(terrain, 'build_terrain', build)
    monkeypatch.setattr(terrain, 'make_terrain_cfg', lambda t: (None, []))
    monkeypatch.setattr(deepwbc_runtime, 'Go1WidowXIsaac', lambda **kwargs: InterfaceRuntime(kwargs['config']))
    with pytest.raises(SystemExit) as error:
        cli.main(['--checkpoint','fixture.pt','--task','unused.json','--output',str(tmp_path/'run'),
                  '--steps','3','--device','cpu','--num-envs','2'])
    expected=int(failure_type is not None)
    assert error.value.code==seen['exit_code']==expected
    assert seen['visualizer']=='none' and 'headless' not in seen
    report=json.loads((tmp_path/'run'/'summary.json').read_text())
    assert report['status']==('failed' if expected else 'completed')
    if expected:
        assert 'preserved startup failure' in report['error']
        assert 'preserved startup failure' in capsys.readouterr().err
    else:
        assert report['steps']==3
