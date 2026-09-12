"""CPU consumer semantics; actual PhysX parity is a separate bounded probe."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
from test_roboduet_evaluation import ExplicitState
from pawcerto.methods.roboduet import default_config
from pawcerto.methods.roboduet.training.isaac_env import RoboDuetIsaacTrainingEnv


def test_runtime_optional_capture_keeps_all_physics_operations():
    path = Path(__file__).resolve().parents[1]/'pawcerto/isaac/roboduet_runtime.py'
    tree = ast.parse(path.read_text())
    method = next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Go1Arx5Isaac'
                  for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='step_control')
    ns = dict(torch=torch)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
    def run(capture):
        calls=[]
        def record(name): return lambda *a,**kw: calls.append(name)
        obj=SimpleNamespace(joint_ids=list(range(20)),default_pos=torch.zeros(20),stage=1,num_envs=3,
            time=torch.zeros(3),dt=.005,
            robot=SimpleNamespace(**{name:record(name) for name in ('set_joint_effort_target',
                'set_joint_position_target','write_joint_position_to_sim_index','write_joint_velocity_to_sim_index')}),
            scene=SimpleNamespace(write_data_to_sim=record('write'),update=record('update')),
            sim=SimpleNamespace(step=record('physics')),state=lambda: calls.append('state') or {'complete':True})
        result=ns['step_control'](obj,torch.zeros(3,12),torch.zeros(3,8),**capture)
        return calls,obj.time,result
    old,time_old,result=run({})
    new,time_new,empty=run({'capture_state':False})
    assert old == new+['state'] and result == {'complete':True} and empty is None
    assert torch.equal(time_old,time_new)


def test_training_skips_unused_capture_but_preserves_callbacks_and_terminal_data():
    class CountingState(ExplicitState):
        def __init__(self,cfg):
            super().__init__(cfg)
            self.captures=[]
        def step_control(self,*args,capture_state=True):
            self.captures.append(capture_state)
            super().step_control(*args)
            return self.state() if capture_state else None
    cfg=default_config()
    runtime=CountingState(cfg)
    env=RoboDuetIsaacTrainingEnv(runtime,cfg)
    env.set_stage(2)
    env._reset_indices(torch.arange(3))
    env.step(torch.zeros(3,12),torch.zeros(3,6))
    assert runtime.captures == [False]*4
    samples=[]
    terminal=[]
    env.step(torch.zeros(3,12),torch.zeros(3,6),
             substep_callback=lambda e,i,b,o,s:samples.append((i,s['root_pos'].clone())),
             pre_reset_callback=lambda e,d,t:terminal.append(e.task.root_states.clone()))
    assert runtime.captures == [False]*4+[True]*4
    assert len(samples)==4 and samples[-1][1][0,2] == .1
    assert terminal[0][0,2] == .1 and env.task.root_states[0,2] == .34
