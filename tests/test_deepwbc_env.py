"""Interface-double tests: no simulated physics or learning claims."""
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from pawcerto.methods.deepwbc.config import default_config
from pawcerto.methods.deepwbc.controller import DeepWBCController
from pawcerto.isaac.deepwbc_runtime import SIM_DOF_NAMES
from pawcerto.methods.deepwbc.training.isaac_env import DeepWBCIsaacTrainingEnv

torch.set_num_threads(1)


class InterfaceRuntime:
    """Records physical API calls; never integrates robot motion."""
    def __init__(self, config):
        self.device='cpu'; self.num_envs=2; self.dt=config['sim']['dt']
        self.joint_names=list(SIM_DOF_NAMES); self.body_names=['base','FL_foot','FR_foot','RL_foot','RR_foot']
        self.controller=DeepWBCController(2,self.joint_names,torch.ones(20)*100,config)
        self.calls=[]; self.efforts=[]
        self.root=torch.zeros(2,13); self.root[:,2]=.42; self.root[:,6]=1
        self.q=self.controller.default.repeat(2,1); self.qd=torch.zeros(2,20)
        self.torques=torch.zeros(2,20); self.motor_strength=torch.ones(2,18)
    def apply_creation_randomization(self,**kwargs):
        self.calls.append('randomize'); self.motor_strength=kwargs['motor_strength']
    def push_actions(self,actions):
        self.calls.append('push_actions'); self.controller.push(actions)
    def step_control(self):
        self.calls.append('physics_step')
        self.torques=self.controller.torques(self.q,self.qd,self.motor_strength)
        self.efforts.append(self.torques.clone())
        return self.state()
    def state(self):
        self.calls.append('state')
        return dict(root_states=self.root,dof_pos=self.q,dof_vel=self.qd,torques=self.torques,
            actions=self.controller.actions,latest_sim_actions=self.controller.history[:,-1],
            ee_pos=torch.tensor([[.4,0,.53]]).repeat(2,1),ee_orn=self.root[:,3:7],
            contact_forces=torch.zeros(2,5,3),foot_wrenches=torch.zeros(2,4,6))
    def reset(self,ids,root_states,joint_pos,joint_vel):
        self.calls.append('reset'); self.root[ids]=root_states; self.q[ids]=joint_pos; self.qd[ids]=joint_vel
        self.controller.reset(ids)
    def reset_box(self,ids,positions):
        self.calls.append('reset_box')
        assert positions.shape==(len(ids),3)
    def push_base_velocity_xy(self,velocity):
        self.calls.append('push_velocity'); self.root[:,7:9]=velocity


def config():
    c=default_config(); c['env']['num_envs']=2
    c['domain_rand']['push_robots']=False
    c['terrain']['init_vel_perturb_range']=0
    return c


def test_original_zero_step_reset_and_observation_cache():
    c=config(); r=InterfaceRuntime(c); e=DeepWBCIsaacTrainingEnv(r,c)
    obs,priv=e.reset()
    assert obs.shape==(2,860) and priv is None
    assert r.calls.count('physics_step')==4
    assert r.calls.index('reset')<r.calls.index('push_actions')<r.calls.index('physics_step')
    assert e.episode_length_buf.tolist()==[1,1]
    assert ((e.origins[:,0]>=-3.75)&(e.origins[:,0]<=-3)).all()
    assert ((e.origins[:,1]>=-115)&(e.origins[:,1]<=115)).all()
    history=e.observer.history.clone()
    assert e.get_observations() is obs and e.get_privileged_observations() is None
    torch.testing.assert_close(history,e.observer.history) # getters don't advance history


def test_delay_mapping_and_dual_return_autoreset_timeout():
    c=config(); r=InterfaceRuntime(c); e=DeepWBCIsaacTrainingEnv(r,c); e.reset()
    action=torch.zeros(2,18); action[:,0]=.4 # policy FR hip -> sim3
    r.calls.clear()
    for i in range(3):
        obs,priv,leg,arm,done,info=e.step(action)
        assert leg.shape==arm.shape==done.shape==(2,)
        assert priv is None and not done.any()
        assert r.controller.actions[:,3].eq(.4 if i==2 else 0).all()
        assert r.controller.actions[:,0].eq(0).all()
    assert r.calls.count('push_actions')==3 and r.calls.count('physics_step')==12
    e.task.episode_length_buf[:]=e.task.max_episode_length
    result=e.step(action)
    assert result[4].all() and result[5]['time_outs'].all()
    assert e.episode_length_buf.eq(0).all()
    assert r.controller.history.eq(0).all()
    assert e.observer.history.shape==(2,10,76)
    # Done is retained even though resetting clears episode lengths/history.
    assert result[0].shape==(2,860) and result[5]['episode']
    e.task.episode_length_buf[:]=0
    result=e.step(action)
    assert not result[5]['time_outs'].any()


def test_push_callback_and_curriculum_delegate():
    c=config(); c['domain_rand']['push_robots']=True
    c['domain_rand']['push_interval_s']=.02
    r=InterfaceRuntime(c); e=DeepWBCIsaacTrainingEnv(r,c)
    e.reset()
    assert 'push_velocity' in r.calls
    assert r.calls.index('physics_step') < r.calls.index('push_velocity')
    e.update_command_curriculum()
    assert e.task.update_counter==1 and e.task.lin_vel_x_ranges.tolist()==[0,.9]
    replacement=torch.tensor([10,11]); e.episode_length_buf=replacement
    assert e.task.episode_length_buf is replacement


def test_cli_config_and_named_schedule_are_explicit():
    path=Path(__file__).resolve().parents[1]/'scripts/train_deepwbc.py'
    spec=importlib.util.spec_from_file_location('deepwbc_cli',path)
    cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    args=cli.parser().parse_args(['--iterations','21','--output','/tmp/not-created','--schedule','paper',
                                 '--force-mode','native_joint_wrench','--robot-usd','robot.usd'])
    c=cli.resolve_config(args)
    assert c['pawcerto_schedule']=='paper'
    assert c['algorithm']['dagger_update_freq']==20
    assert c['runner']['max_iterations']==10000
    assert not Path('/tmp/not-created').exists()
    # Validate against the real runtime's Python API without constructing Sim.
    import inspect
    from pawcerto.isaac.deepwbc_runtime import Go1WidowXIsaac
    inspect.signature(Go1WidowXIsaac).bind(terrain_cfg=None,terrain_contact_paths=[],
        num_envs=2,device='cpu',usd_path='robot.usd',config=c,
        force_signal='native_joint_wrench',env_origins=torch.zeros(2,3))
    inspect.signature(Go1WidowXIsaac.apply_creation_randomization).bind(None,
        mass_params=None,friction=None,motor_strength=None,box_added_mass=None)
    inspect.signature(Go1WidowXIsaac.reset_box).bind(None,None,None)
    inspect.signature(Go1WidowXIsaac.push_base_velocity_xy).bind(None,None)


def cli_module():
    path=Path(__file__).resolve().parents[1]/'scripts/train_deepwbc.py'
    spec=importlib.util.spec_from_file_location('deepwbc_cli',path)
    cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    return cli


def cpu_runner(output):
    from pawcerto.methods.deepwbc.training.runner import OnPolicyRunner
    c=config()
    c['runner'].update(num_steps_per_env=2,save_interval=2)
    c['algorithm'].update(num_learning_epochs=1,num_mini_batches=1)
    env=DeepWBCIsaacTrainingEnv(InterfaceRuntime(c),c)
    return OnPolicyRunner(env,c,log_dir=output)


@pytest.mark.parametrize('iterations', [4,5])
def test_cli_streams_metrics_with_only_periodic_and_final_checkpoints(tmp_path,monkeypatch,iterations):
    runner=cpu_runner(tmp_path)
    saves=[]
    save=runner.save
    def record(path,*args,**kwargs):
        saves.append(Path(path).name)
        return save(path,*args,**kwargs)
    monkeypatch.setattr(runner,'save',record)
    cli_module().run_iterations(runner,iterations,tmp_path)
    expected=['model_0.pt','model_2.pt','model_4.pt']
    if iterations==5: expected.append('model_5.pt')
    assert saves==expected # no duplicate final write at periodic boundary
    assert sorted(p.name for p in tmp_path.glob('model_*.pt'))==expected
    metrics=[json.loads(line) for line in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert [row['iteration'] for row in metrics]==list(range(iterations))
    assert metrics[0]['stage']=='dagger' and metrics[1]['stage']=='ppo'
    checkpoint=torch.load(tmp_path/f'model_{iterations}.pt',weights_only=False)
    assert checkpoint['iter']==checkpoint['algorithm_counter']==iterations


@pytest.mark.parametrize('failure_type',[RuntimeError,KeyboardInterrupt])
def test_cli_preserves_midrun_failure_and_completed_metrics(tmp_path,monkeypatch,failure_type):
    runner=cpu_runner(tmp_path)
    step=runner.env.step
    count=0
    failure=failure_type('physical step failure preserved')
    def fail(actions):
        nonlocal count
        count+=1
        if count==3: raise failure # first iteration completed, second not
        return step(actions)
    monkeypatch.setattr(runner.env,'step',fail)
    with pytest.raises(failure_type) as error:
        cli_module().run_iterations(runner,5,tmp_path)
    assert error.value is failure
    assert runner.current_learning_iteration==1
    assert [p.name for p in tmp_path.glob('model_*.pt')]==['model_0.pt']
    rows=(tmp_path/'metrics.jsonl').read_text().splitlines()
    assert len(rows)==1 and json.loads(rows[0])['iteration']==0


@pytest.mark.parametrize('failure_type',[RuntimeError,KeyboardInterrupt])
def test_cli_launcher_uses_visualizer_and_preserves_exit_status(tmp_path,monkeypatch,capsys,failure_type):
    import sys
    from types import ModuleType, SimpleNamespace
    from pawcerto.methods.deepwbc import terrain
    seen={}
    class Launcher:
        def __init__(self,**kwargs):
            seen.update(kwargs)
            self.app=SimpleNamespace(close=self.close)
        def close(self,*,exit_code):
            seen['exit_code']=exit_code
            raise SystemExit(exit_code) # reproduces shutdown replacing propagation
    app_module=ModuleType('isaaclab.app'); app_module.AppLauncher=Launcher
    monkeypatch.setitem(sys.modules,'isaaclab.app',app_module)
    failure=failure_type('terrain failure must remain visible')
    def fail(**kwargs): raise failure
    monkeypatch.setattr(terrain,'build_terrain',fail)
    usd=tmp_path/'robot.usd'; usd.write_text('interface fixture only')
    with pytest.raises(SystemExit) as error:
        cli_module().main(['--iterations','1','--output',str(tmp_path/'run'),
                          '--robot-usd',str(usd),'--force-mode','native_joint_wrench','--device','cpu'])
    assert error.value.code==seen['exit_code']==1
    assert seen['visualizer']=='none' and 'headless' not in seen
    assert 'terrain failure must remain visible' in capsys.readouterr().err
