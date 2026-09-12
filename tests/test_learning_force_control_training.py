"""Released-source configuration and tensor-equation checks, without a simulator."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from pawcerto.methods.learning_force_control.training.config import default_config,resolve_config,as_dict
from pawcerto.methods.learning_force_control.training.lifecycle import LFCLifecycle
from pawcerto.methods.learning_force_control.training import torch_math,math_utils

SOURCE=Path(__file__).resolve().parents[1]/'third_party/learning-force-control-reference'


def source_file(path):
    if not SOURCE.exists():pytest.skip('Fetch fixed LFC source for actual-source comparison')
    return (SOURCE/path).read_text()


def test_complete_released_configuration():
    ns={'np':np}
    tree=ast.parse(source_file('b1_gym/envs/base/legged_robot_config.py'))
    tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    for n in ast.walk(tree):
        if isinstance(n,ast.ClassDef):n.bases=[];n.keywords=[]
    exec(compile(tree,'original/base_config.py','exec'),ns)
    tree=ast.parse(source_file('b1_gym/envs/b1/b1_plus_z1_config.py'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    for node in tree.body:
        for arg in node.args.args:arg.annotation=None
    exec(compile(tree,'original/robot_config.py','exec'),ns)
    ns['AC_Args']=SimpleNamespace()
    tree=ast.parse(source_file('scripts/train.py'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure_env']
    exec(compile(tree,'original/train.py','exec'),ns)
    ns['configure_env']()
    cfg=default_config()
    assert cfg['env']==as_dict(ns['Cfg'])
    for key,value in vars(ns['AC_Args']).items():assert cfg['policy'][key]==value
    assert cfg['env']['env']['add_balls'] is False
    assert (cfg['env']['env']['num_actions'],cfg['env']['env']['num_observations'],cfg['runner']['num_steps_per_env'])==(19,87,48)


def test_original_lifecycle_tensor_statements_preserved():
    tree=ast.parse(source_file('b1_gym/envs/base/legged_robot.py'))
    original=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    port=ast.parse(Path(__import__('inspect').getfile(LFCLifecycle)).read_text())
    port=next(n for n in port.body if isinstance(n,ast.ClassDef))
    original={n.name:n for n in original.body if isinstance(n,ast.FunctionDef)}
    port={n.name:n for n in port.body if isinstance(n,ast.FunctionDef)}
    for name in ['compute_intermediate_ee_pos_command','check_termination','compute_reward',
                 '_randomize_dof_props','_resample_commands','_step_contact_targets','_compute_torques',
                 '_resample_force_or_position_control','_get_env_origins','_parse_cfg']:
        assert ast.dump(original[name])==ast.dump(port[name]),name


def test_actual_source_force_schedule_and_hybrid_mask():
    text=source_file('b1_gym/envs/base/legged_robot.py');tree=ast.parse(text)
    original=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    method=deepcopy(next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='_push_gripper'))
    class ProvenGymLookup(ast.NodeTransformer):
        def visit_Call(self,node):
            # Actual original Gym probe proved this lookup is -1, not an alias.
            if isinstance(node.func,ast.Attribute) and node.func.attr=='find_actor_rigid_body_handle':
                return ast.copy_location(ast.Constant(-1),node)
            return self.generic_visit(node)
    method=ProvenGymLookup().visit(method)
    ns={**vars(torch_math),**vars(math_utils),'torch':torch,'np':np}
    for node in tree.body:
        if isinstance(node,ast.Assign):exec(compile(ast.Module(body=[node],type_ignores=[]),'original/constants','exec'),ns)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'original/force.py','exec'),ns)
    cfg=resolve_config(default_config())
    task=SimpleNamespace(cfg=cfg,sim_params=cfg.sim)
    LFCLifecycle._parse_cfg(task,cfg)
    task.num_envs=4;task.device='cpu';task.gripper_stator_index=22;task.ee_measurement_index=23
    task.force_or_position_control=torch.tensor([0.,1.,1.,0.]);task.freed_envs=torch.zeros(4,dtype=torch.bool)
    task.episode_length_buf=torch.zeros(4,dtype=torch.long);task.push_interval=torch.full((4,1),200)
    task.selected_env_ids=torch.zeros(4,dtype=torch.int32);task.push_end_time=torch.zeros(4);task.push_duration=torch.zeros(4)
    task.force_target=torch.zeros(4,3);task.current_Fxyz_cmd=torch.zeros(4,3)
    task.commands=torch.zeros(4,23);task.commands[:,15]=.6
    task.base_pos=torch.zeros(4,3);task.base_pos[:,2]=.65
    task.base_quat=torch.tensor([[0.,0.,0.,1.]]).repeat(4,1)
    task.rigid_body_state=torch.zeros(4,24,13);task.rigid_body_state[:,23,:3]=torch.tensor([.5,0.,.6])
    task.forces=torch.zeros(4,24,3);task.gripper_force_kps=torch.full((4,3),100.);task.gripper_force_kds=torch.full((4,3),10.)
    task.gripper_velocity=torch.zeros(4,3);task.forces_deactivated=False
    original,port=deepcopy(task),deepcopy(task)
    for step in (0,1,100,175,250):
        original.episode_length_buf[:]=step;port.episode_length_buf[:]=step
        torch.manual_seed(12+step);ns['_push_gripper'](original,torch.arange(4),original.cfg)
        torch.manual_seed(12+step);LFCLifecycle._push_gripper(port,torch.arange(4),port.cfg)
        for key,value in vars(original).items():
            if isinstance(value,torch.Tensor):torch.testing.assert_close(value,getattr(port,key),rtol=0,atol=0)
        assert port.forces[[0,3],22].count_nonzero()==0
        assert port.commands[[0,3],12:15].count_nonzero()==0


def test_xyzw_rotation_and_euler_convention():
    angles=torch.tensor([-.7,.3,1.2])
    q=torch_math.quat_from_euler_xyz(angles[:1],angles[1:2],angles[2:])
    recovered=torch.stack(torch_math.get_euler_xyz(q),-1)
    torch.testing.assert_close(recovered,angles.reshape(1,3)%(2*torch.pi))
    v=torch.tensor([[.2,-.6,.8]])
    torch.testing.assert_close(torch_math.quat_rotate_inverse(q,torch_math.quat_apply(q,v)),v)
