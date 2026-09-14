"""Candidate control/serialization checks; no physics advance or optimizer step."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from pawcerto.methods.umi_on_legs import UmiController, UmiPolicy, RobotState
from pawcerto.methods.umi_on_legs.actuation import actuation_mode, require_same_actuation
from pawcerto.methods.umi_on_legs.robot_binding import as2_config, joint_order
from pawcerto.methods.umi_on_legs.training import load_config, UmiTrainer
from pawcerto.methods.umi_on_legs.training.semantics import runtime_contract, require_resume_contract
from pawcerto.methods.umi_on_legs.training.isaac_env import UmiIsaacTrainingEnv


def config(native=True):
    cfg = as2_config(load_config())
    if native:
        cfg['actuation_mode'] = 'as2-native-servo'
        cfg['joint_velocity_limit_override_rad_s'] = 1e6
    cfg['pawcerto_runtime'] = runtime_contract('normal-contact', cfg.get('joint_velocity_limit_override_rad_s'), cfg.get('actuation_mode','external-pd'))
    return cfg


def test_old_contract_and_native_rejections():
    assert runtime_contract('normal-contact') == dict(force_signal='normal-contact',physics_revision='umi-physics-mapping-v3',joint_friction_model='physx-legacy-friction-coefficient')
    assert 'actuation_mode' not in runtime_contract('reconstructed-solver',1000)
    require_same_actuation({}, {'actuation_mode':'external-pd'})
    with pytest.raises(ValueError,match='AS2/Piper'):
        actuation_mode(dict(actuation_mode='as2-native-servo'))
    with pytest.raises(ValueError,match='normal-contact'):
        runtime_contract('reconstructed-solver',1e6,'as2-native-servo')
    with pytest.raises(ValueError,match='1000000'):
        runtime_contract('normal-contact',1000,'as2-native-servo')
    with pytest.raises(ValueError):
        require_resume_contract(config(False)['pawcerto_runtime'],config()['pawcerto_runtime'])


@pytest.mark.parametrize('native',[False,True])
def test_actual_delay_source_arm_clip_and_dispatch(native):
    cfg=config(native);c=UmiController(cfg,2)
    raw=torch.tensor([[100.,-100.]*9,[-100.,100.]*9])
    q=c.offset.repeat(2,1)+.03;qd=torch.linspace(-.5,.5,36).reshape(2,18)
    class Runtime:
        def step_servo(self,target):self.target=target.clone()
        def step_torque(self,effort):self.effort=effort.clone()
    runtime=Runtime();c.push(raw)
    for substep in range(4):
        torch.testing.assert_close(c.position_target(substep),c.offset.repeat(2,1))
    c.push(torch.zeros_like(raw))
    unbounded=c.offset+c.scale*raw.clamp(-c.max_action,c.max_action)
    expected=unbounded.clone()
    if native:
        lo,hi=c.piper_limits;expected[:,12:]=expected[:,12:].clamp(lo,hi)
        assert not torch.equal(expected[:,12:],unbounded[:,12:])
    for substep in range(4):
        torch.testing.assert_close(c.position_target(substep),expected,rtol=0,atol=0)
        value=c.step(runtime,q,qd,substep)
        torque=(c.kp*(expected-q)-c.kd*qd).clamp(-c.torque_limit,c.torque_limit)
        torch.testing.assert_close(value,torque,rtol=0,atol=0)
        torch.testing.assert_close(runtime.target if native else runtime.effort,expected if native else torque)
    torch.testing.assert_close(expected[:,:12],unbounded[:,:12],rtol=0,atol=0)
    c.push(torch.zeros_like(raw))
    torch.testing.assert_close(c.position_target(0),c.offset.repeat(2,1))


class Runtime:
    """Tensor-only call boundary; does not import or replace a physics result."""
    device='cpu';num_envs=2;force_signal='normal-contact';actuation_mode='as2-native-servo'
    joint_velocity_limit_override_rad_s=1e6
    def __init__(self,cfg):
        self.cfg=cfg;self.time=torch.zeros(2);self.joint_names=joint_order(cfg)
        self.body_names=[x.get('name') for x in ET.parse(cfg['pawcerto_robot']['urdf_path']).findall('link')]
        self.q=torch.tensor(cfg['env']['controller']['offset']['data']).repeat(2,1)
        self.qd=torch.full((2,18),.3);self.calls=[]
        self.setup={v['key']:torch.zeros(2,v['dim']) for k,v in cfg['env']['privileged_setup_obs'].items() if not k.startswith('_')}
        self.setup['dof_damping']=torch.linspace(.01,.5,36).reshape(2,18)
        self.kp=torch.zeros_like(self.q);self.kd=torch.zeros_like(self.q)
    def reset(self,ids):self.time[ids]=0
    def set_servo_gains(self,kp,kd,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.kp[ids]=kp;self.kd[ids]=kd
    def joints(self):return self.q,self.qd
    def step_torque(self,value):raise AssertionError('Native path dispatched external torque')
    def step_servo(self,target):
        self.calls.append(dict(target=target.clone(),q=self.q.clone(),qd=self.qd.clone(),kp=self.kp.clone(),kd=self.kd.clone()))
        self.qd.add_(.01);self.time.add_(.005)
    def state(self):return RobotState(torch.zeros(2,3),torch.tensor([[0.,0.,-1.]]).repeat(2,1),self.q,self.qd,torch.eye(4).repeat(2,1,1),self.time)
    def training_setup(self):return self.setup
    def training_state(self):
        return dict(root_pos=torch.tensor([[0.,0.,.3]]).repeat(2,1),local_root_lin_vel=torch.zeros(2,3),contact_forces=torch.zeros(2,len(self.body_names),3),feet_force_z=torch.tensor([[1.,1.,1.,1.],[-2.,0.,2.,6.]]),body_pos=torch.zeros(2,len(self.body_names),3))
    def apply_domain_randomization(self,step):pass


@pytest.mark.parametrize('dense',[False,True])
def test_training_reset_effective_gains_reward_sampling_and_normal_emd(tmp_path,dense):
    cfg=config();cfg['env']['dense_rewards']=dense
    path=tmp_path/'poses.pkl'
    with path.open('wb') as f:pickle.dump([dict(ee_pos=np.zeros((8,3)),ee_axis_angle=np.zeros((8,3)))],f)
    if dense:
        from pawcerto.methods.umi_on_legs.training.evaluation import author_eval_config
        cfg['env']['constraints']['energy']=author_eval_config(cfg,path,'cpu')['env']['constraints']['energy']
    runtime=Runtime(cfg);env=UmiIsaacTrainingEnv(runtime,cfg,path)
    ids=torch.tensor([1]);before_kd=env.controller.kd[0].clone()
    g=torch.Generator().set_state(env.generator.get_state());rand=cfg['env']['cfg']['domain_rand'];expected={}
    for key in ('kp','kd'):
        lo,hi=rand[f'{key}_ratio_range'];expected[key]=(torch.rand((1,18),generator=g)*(hi-lo)+lo)*getattr(env,'initial_'+key)
    env._reset_indices(ids)
    torch.testing.assert_close(runtime.kp[ids],expected['kp'])
    torch.testing.assert_close(env.randomized_kd[ids],expected['kd'])
    torch.testing.assert_close(runtime.kd[ids],expected['kd']+runtime.setup['dof_damping'][ids])
    torch.testing.assert_close(env.controller.kd[0],before_kd)
    # Disabling new sampling on a second reset must not add dof_damping twice.
    cfg['env']['cfg']['domain_rand']['randomize_pd_params']=False
    prior=env.controller.kd.clone();env._reset_indices(ids);torch.testing.assert_close(env.controller.kd,prior)
    capture=[];reward=env.reward
    class Capture:
        curriculum=reward.curriculum;reward_dt=reward.reward_dt
        def __call__(self,state):
            capture.append((state.torque.clone(),state.prev_dof_vel.clone(),state.dof_vel.clone()))
            return reward(state)
    env.reward=Capture();env.controller.push(torch.full((2,18),100.))
    _,critic,_,done,info=env.step(torch.zeros(2,18))
    assert len(runtime.calls)==4 and len(capture)==(4 if dense else 1)
    relevant=runtime.calls if dense else runtime.calls[-1:]
    for sample,record in zip(capture,relevant):
        expected_effort=(record['kp']*(record['target']-record['q'])-record['kd']*record['qd']).clamp(-env.controller.torque_limit,env.controller.torque_limit)
        torch.testing.assert_close(sample[0],expected_effort)
        torch.testing.assert_close(sample[1],record['qd'])
        assert not torch.equal(sample[1],sample[2])
    key=('reward/' if dense else '')+'constraint/even_mass_distribution'
    torch.testing.assert_close(info[key],torch.tensor([0.,-.125])*(.02 if dense else 1.))
    if dense:
        assert 'servo_mechanical_power_estimate' in info and 'mechanical_power' not in info
        assert 'servo_electrical_power_estimate' in info and 'electrical_power' not in info
    assert not done.any();assert critic.shape==(2,cfg['runner']['alg']['actor_critic']['num_critic_obs'])
    saved=env.training_state_dict()['native_servo_gain_samples']
    torch.testing.assert_close(saved['effective_kd'],saved['randomized_kd']+saved['sampled_dof_damping'])


def test_resume_rejects_before_mutation_actor_initialization_and_export(tmp_path):
    # Build/save fresh networks only; no PPO update is called.
    env=SimpleNamespace(num_envs=2)
    old=config(False);new=config()
    source=UmiTrainer(env,old,rollout_steps=4);source.save(tmp_path/'old.pt')
    target=UmiTrainer(env,new,rollout_steps=4);before={k:v.clone() for k,v in target.alg.actor_critic.state_dict().items()};rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='actuation mode'):
        target.load(tmp_path/'old.pt')
    for k,v in before.items():torch.testing.assert_close(target.alg.actor_critic.state_dict()[k],v,rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state()) and not target.alg.optimizer.state
    target.load_actor(tmp_path/'old.pt')
    for k,v in target.alg.actor_critic.state_dict().items():
        expected=source.alg.actor_critic.state_dict()[k] if k.startswith('actor.') or k=='std' else before[k]
        torch.testing.assert_close(v,expected,rtol=0,atol=0)
    assert target.iteration==0 and not target.alg.optimizer.state
    target.save(tmp_path/'native.pt');(tmp_path/'config.json').write_text(json.dumps(new))
    resumed=UmiTrainer(env,new,rollout_steps=4);resumed.load(tmp_path/'native.pt')
    changed=deepcopy(new);changed['env']['controller']['kd']['data'][12]*=2
    with pytest.raises(ValueError,match='contract'):
        UmiTrainer(env,changed,rollout_steps=4).load(tmp_path/'native.pt')
    policy=UmiPolicy(tmp_path/'native.pt');policy.export(tmp_path/'export',joint_order(new));restored=UmiPolicy(tmp_path/'export')
    assert restored.config==new and restored.controller(1).native_servo
    (tmp_path/'config.json').write_text(json.dumps(old))
    with pytest.raises(ValueError,match='actuation mode'):UmiPolicy(tmp_path/'native.pt')
    (tmp_path/'export/config.json').write_text(json.dumps(old))
    with pytest.raises(ValueError,match='execution config'):UmiPolicy(tmp_path/'export')


def test_mujoco_native_model_source_limits_without_simulation(monkeypatch):
    from pawcerto.mujoco.runtime import Go2Arx5Mujoco
    import mujoco
    # Constructor model compilation/readback only; no mj_forward or mj_step.
    monkeypatch.setattr(Go2Arx5Mujoco,'reset',lambda self:None)
    monkeypatch.setattr(mujoco,'mj_step',lambda *a:pytest.fail('No physics advance authorized'))
    sim=Go2Arx5Mujoco(config())
    assert sim.native_servo and sim.model.opt.integrator==mujoco.mjtIntegrator.mjINT_DISCRETE
    np.testing.assert_allclose(sim.model.actuator_gainprm[:,0],sim.kp)
    np.testing.assert_allclose(-sim.model.actuator_biasprm[:,2],sim.kd)
    np.testing.assert_allclose(sim.model.actuator_forcerange[:,1],sim.torque_limit)
    legacy=Go2Arx5Mujoco(config(False));assert not legacy.native_servo
    assert legacy.model.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST
