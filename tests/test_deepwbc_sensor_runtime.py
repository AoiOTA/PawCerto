"""CPU interface checks for the real runtime's PRE/POST sensor ordering."""
from types import SimpleNamespace, ModuleType
import sys
import torch
from pawcerto.isaac.deepwbc_runtime import Go1WidowXIsaac


def test_pre_reads_com_before_dynamics_and_public_c_before_velocity(monkeypatch):
    calls=[]
    env=Go1WidowXIsaac.__new__(Go1WidowXIsaac)
    env.num_envs=1;env.device='cpu'
    class Data:
        @property
        def body_com_pose_w(self):
            calls.append('com_pose')
            return SimpleNamespace(torch=torch.zeros(1,27,7))
        @property
        def body_link_quat_w(self):
            calls.append('rotation')
            return SimpleNamespace(torch=torch.zeros(1,27,4))
    env.robot=SimpleNamespace(data=Data())
    def native(name):
        calls.append(name)
        return torch.zeros(1)
    env._native_tensor=native
    def predict(**data):
        calls.append('predict')
        assert data['com_position'].shape==(1,27,3)
        return torch.zeros(1,27,6),None,None
    env._solver_force=SimpleNamespace(predict=predict)
    math=ModuleType('isaaclab.utils.math');math.matrix_from_quat=lambda q:torch.eye(3).repeat(1,27,1,1)
    monkeypatch.setitem(sys.modules,'isaaclab.utils.math',math)
    env._sensor_pre_step()
    assert calls[0]=='com_pose'
    assert calls.index('get_coriolis_and_centrifugal_compensation_forces')<calls.index('get_link_velocities')
    assert calls[-1]=='predict'


def test_real_step_uses_full_post_acceleration_and_pre_rotation():
    calls=[];env=Go1WidowXIsaac.__new__(Go1WidowXIsaac)
    env.force_signal='reconstructed_sensor_wrench';env.num_envs=2;env.device='cpu';env.dt=.005
    env.joint_ids=list(reversed(range(20)));env.feet_indices=[4,8,12,16]
    env.time=torch.zeros(2);env.motor_strength=torch.ones(2,18)
    env.joints=lambda:(torch.zeros(2,20),torch.zeros(2,20))
    torque=torch.arange(40).reshape(2,20).float()
    env.controller=SimpleNamespace(torques=lambda *a:torque)
    def effort(value,*,joint_ids):
        assert value is torque and joint_ids==env.joint_ids
        calls.append('PD')
    env.robot=SimpleNamespace(set_joint_effort_target=effort,
        instantaneous_wrench_composer=SimpleNamespace(active=False),permanent_wrench_composer=SimpleNamespace(active=False))
    env.scene=SimpleNamespace(write_data_to_sim=lambda:calls.append('write'),update=lambda dt:calls.append('update'))
    env.sim=SimpleNamespace(step=lambda **kw:calls.append('physics'))
    free=torch.ones(2,27,6);rotation=torch.eye(3).repeat(2,27,1,1)
    def pre(): calls.append('pre');return free,rotation
    env._sensor_pre_step=pre
    reported=torch.ones(2,27,6)*2
    def native(name):
        assert name=='get_link_accelerations';calls.append('post');return reported
    env._native_tensor=native
    env._sensor_com_b=torch.zeros(2,27,3);env._sensor_gravity=torch.zeros(2,27,3)
    result=torch.arange(2*27*6).reshape(2,27,6).float()
    def approximate(actual,predicted,**kw):
        assert actual is reported and predicted is free and kw['link_rotation'] is rotation
        calls.append('six_axis');return result
    env._solver_force=SimpleNamespace(approximate_sensor_wrench=approximate)
    env._sensor_wrenches=torch.zeros(2,4,6);env.state=lambda:env._sensor_wrenches.clone()
    got=env.step_control()
    assert calls==['PD','write','pre','physics','update','post','six_axis']
    torch.testing.assert_close(got,result[:,env.feet_indices])
    torch.testing.assert_close(env.time,torch.full((2,),.005))


def test_resume_rejects_changing_saved_sensor_mode_before_launch(tmp_path):
    import pytest
    from scripts import train_deepwbc
    from pawcerto.methods.deepwbc.config import default_config
    cfg=default_config();cfg['pawcerto_runtime']={'force_mode':'reconstructed_sensor_wrench'}
    checkpoint=tmp_path/'checkpoint.pt';torch.save({'config':cfg},checkpoint)
    with pytest.raises(ValueError,match='Resume force mode differs'):
        train_deepwbc.main(['--resume',str(checkpoint),'--force-mode','native_joint_wrench',
            '--iterations','1','--output',str(tmp_path/'run')])
    args=train_deepwbc.parser().parse_args(['--force-mode','reconstructed_sensor_wrench',
        '--iterations','21','--output',str(tmp_path/'run')])
    assert args.force_mode=='reconstructed_sensor_wrench'


def test_initialization_reads_current_randomized_mass_inertia_and_com():
    env=Go1WidowXIsaac.__new__(Go1WidowXIsaac)
    env.num_envs=2;env.device='cpu';env.dt=.005
    env.config={'sim':{'gravity':[0.,0.,-9.81]}}
    env.body_names=['base']+[f'link{i}' for i in range(1,27)]
    native={
        'get_masses':torch.ones(2,27),
        'get_inertias':torch.eye(3).repeat(2,27,1,1).flatten(-2),
        'get_dof_max_velocities':torch.full((2,20),1000.),
        'get_dof_armatures':torch.zeros(2,20),
        'get_disable_gravities':torch.zeros(2,27)}
    env._native_tensor=lambda name:native[name].clone()
    com=torch.zeros(2,27,7)
    properties={'maxLinearVelocity':1000.,'maxAngularVelocity':57295.78,
        'linearDamping':0.,'angularDamping':0.,'retainAccelerations':False,'enableGyroscopicForces':True}
    class Prim:
        def GetAttribute(self,name):return SimpleNamespace(Get=lambda:properties[name.split(':')[-1]])
    env.sim=SimpleNamespace(stage=SimpleNamespace(GetPrimAtPath=lambda path:Prim()),
        cfg=SimpleNamespace(physics=SimpleNamespace(enable_external_forces_every_iteration=False)))
    env.robot=SimpleNamespace(root_view=SimpleNamespace(
        shared_metatype=SimpleNamespace(fixed_base=False,link_names=env.body_names,dof_names=list(range(20)),link_parent_indices={name:0 for name in env.body_names[1:]}),
        link_paths=[[f'/env{n}/{name}' for name in env.body_names] for n in range(2)]),
        joint_names=list(range(20)),data=SimpleNamespace(body_com_pose_b=SimpleNamespace(torch=com)))
    env._initialize_sensor_reconstruction()
    first=env._solver_force
    native['get_masses'][1,0]=3.
    native['get_inertias'][1,0]*=3.
    com[1,0,:3]=torch.tensor([.1,.2,.3])
    # Creation-randomization invalidation routes the next PRE initialization here.
    env._solver_force=None
    env._initialize_sensor_reconstruction()
    assert env._solver_force is not first
    assert env._solver_force.masses[1,0]==3.
    torch.testing.assert_close(env._solver_force.body_inertias[1,0],torch.eye(3)*3.)
    torch.testing.assert_close(env._sensor_com_b[1,0],torch.tensor([.1,.2,.3]))
    torch.testing.assert_close(env._sensor_gravity[1,0],torch.tensor([0.,0.,-29.43]))


def test_train_resume_accepts_pointer_alias_of_checkpoint_physical_usd(tmp_path,monkeypatch):
    import hashlib
    import pytest
    from scripts import train_deepwbc
    from pawcerto.methods.deepwbc.config import default_config
    usd=tmp_path/'robot.usda';usd.write_text('#usda 1.0\n')
    pointer=tmp_path/'usd_path.txt';pointer.write_text(str(usd)+'\n')
    cfg=default_config();cfg['pawcerto_runtime']=dict(force_mode='reconstructed_sensor_wrench',
        robot_usd=str(usd),robot_usd_sha256=hashlib.sha256(usd.read_bytes()).hexdigest())
    checkpoint=tmp_path/'checkpoint.pt';torch.save({'config':cfg},checkpoint)
    class AcceptedIdentity(Exception):pass
    class Launcher:
        def __init__(self,**kwargs):raise AcceptedIdentity()
    app=ModuleType('isaaclab.app');app.AppLauncher=Launcher
    monkeypatch.setitem(sys.modules,'isaaclab.app',app)
    # Stop at the mocked launch boundary; this test starts no simulator.
    with pytest.raises(AcceptedIdentity):
        train_deepwbc.main(['--resume',str(checkpoint),'--robot-usd',str(pointer),
            '--iterations','1','--output',str(tmp_path/'run')])
    usd.write_text('#usda 1.0\n# changed physical asset\n')
    with pytest.raises(ValueError,match='Resume robot USD differs'):
        train_deepwbc.main(['--resume',str(checkpoint),'--robot-usd',str(pointer),
            '--iterations','1','--output',str(tmp_path/'run')])
