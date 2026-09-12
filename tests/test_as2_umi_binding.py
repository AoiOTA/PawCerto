"""CPU checks for real robot/config consumers; no Isaac application or training."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from pawcerto.methods.umi_on_legs.robot_binding import as2_config, joint_order, robot_binding, require_same_robot
from pawcerto.methods.umi_on_legs import UmiController, UmiObservation
from pawcerto.methods.umi_on_legs.training import build_algorithm
from pawcerto.methods.umi_on_legs.training.semantics import UmiReward
from pawcerto.mujoco.runtime import Go2Arx5Mujoco
from pawcerto.mujoco.evaluate import robot_state

ROOT=Path(__file__).resolve().parents[1]


def test_released_binding_joint_order_does_not_require_checkout(monkeypatch,tmp_path):
    import importlib
    module=importlib.import_module('pawcerto.methods.umi_on_legs.robot_binding')
    expected=json.loads((ROOT/'configs/umi_go2_arx5_joint_names.json').read_text())
    monkeypatch.setattr(module,'ROOT',tmp_path)
    assert module.joint_order({})==expected


@pytest.fixture
def configs():
    source=json.loads((ROOT/'reference/checkpoints/tossing/ours/config.json').read_text())
    return source,as2_config(source)


def test_binding_preserves_method_and_changes_physical_inputs(configs):
    source,config=configs
    assert 'pawcerto_robot' not in source
    assert robot_binding(source)['root_body']=='base'
    assert joint_order(config)[12:] == [f'piper_joint{i}' for i in range(1,7)]
    for key in ('ctrl_delay','ctrl_buf_len','dense_rewards','obs_history_len'):
        assert config['env'][key]==source['env'][key]
    assert config['env']['cfg']['sim']==source['env']['cfg']['sim']
    assert config['env']['tasks']['reaching']['sequence_sampler']==source['env']['tasks']['reaching']['sequence_sampler']
    assert config['runner']['alg']['actor_critic']['num_actor_obs']==132
    assert config['runner']['alg']['actor_critic']['num_critic_obs']==269
    assert config['env']['controller']['decimation_count']==4
    assert config['env']['cfg']['sim']['dt']==.005
    with pytest.raises(ValueError,match='Checkpoint robot'):
        require_same_robot(config,source)
    with pytest.raises(ValueError,match='Joint order'):
        joint_order(config,ROOT/'configs/umi_go2_arx5_joint_names.json')
    assert as2_config(config)==config
    model=build_algorithm(config,'cpu').actor_critic
    assert model.actor[0].in_features==132
    assert model.critic[0].in_features==269


def test_real_mujoco_tcp_observation_and_delayed_control(configs,tmp_path):
    _,config=configs
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    sim=Go2Arx5Mujoco(path)
    state=sim.state()
    np.testing.assert_allclose(state['ee_pos'],sim.data.site('tcp').xpos,atol=1e-12)
    np.testing.assert_allclose(state['ee_rotmat'],sim.data.site('tcp').xmat.reshape(3,3),atol=1e-12)
    assert sim.model.body_mass.sum()==pytest.approx(22.347)
    controller=UmiController(config,1)
    positions=torch.tensor(state['ee_pos'],dtype=torch.float32)[None,None].repeat(1,400,1)
    rotations=torch.tensor(state['ee_rotmat'],dtype=torch.float32)[None,None].repeat(1,400,1,1)
    observer=UmiObservation(config,positions,rotations)
    observer.advance_pose(robot_state(state).ee_pose)
    observation=observer.observe(robot_state(state),controller.last_action)
    assert observation.shape==(1,132)
    UmiReward(config,[sim.model.body(i).name for i in range(1,sim.model.nbody)],sim.joint_names,1)
    # Compare actual PD torques at each existing physics substep against UMI controller.
    for action in (np.linspace(-.2,.2,18),np.linspace(.2,-.2,18),np.zeros(18)):
        controller.push(torch.tensor(action,dtype=torch.float32)[None])
        samples=[]
        sim.step(action,trace=samples.append)
        assert len(samples)==4
        for sample in samples:
            q=torch.tensor(sample['qpos_before'][sim.qadr],dtype=torch.float32)[None]
            qd=torch.tensor(sample['qvel_before'][sim.vadr],dtype=torch.float32)[None]
            expected=controller.torque(q,qd,sample['substep'])[0].numpy()
            np.testing.assert_allclose(sample['torque_applied'],expected,atol=1e-5,rtol=1e-5)
    assert sim.data.time==pytest.approx(.06)


def test_source_fk_tcp_velocity_and_nominal_critic(configs,tmp_path):
    import mujoco
    import xml.etree.ElementTree as ET
    from pawcerto.mujoco.as2_piper_asset import urdf_fk,transform
    from pawcerto.methods.umi_on_legs.training.semantics import critic_observation
    _,config=configs
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    sim=Go2Arx5Mujoco(path)
    source=ET.parse(ROOT/'reference/as2_piper/robot.urdf').getroot()
    for delta in (0.,.03,-.04):
        sim.data.qpos[sim.qadr]=sim.offset+delta
        mujoco.mj_forward(sim.model,sim.data)
        expected=urdf_fk(source,dict(zip(sim.joint_names,sim.data.qpos[sim.qadr])),sim.data.qpos[:3],sim.data.qpos[3:7])
        tcp=expected['piper_gripper_base']@transform((0,0,.138))
        np.testing.assert_allclose(sim.state()['ee_pos'],tcp[:3,3],atol=1e-6)
        np.testing.assert_allclose(sim.state()['ee_rotmat'],tcp[:3,:3],atol=1e-6)
    sim.data.qvel[:]=np.linspace(-.01,.02,sim.model.nv)
    mujoco.mj_forward(sim.model,sim.data)
    before=sim.state()
    mujoco.mj_integratePos(sim.model,sim.data.qpos,sim.data.qvel,1e-6)
    mujoco.mj_forward(sim.model,sim.data)
    np.testing.assert_allclose((sim.state()['ee_pos']-before['ee_pos'])/1e-6,before['ee_lin_vel_world'],atol=1e-7)
    batch=lambda x:torch.tensor(x,dtype=torch.float32)[None]
    st=robot_state(sim.state())
    c=config['env']['controller']
    bodies=config['env']['cfg']['domain_rand']['randomize_rigid_body_masses']
    setup=dict(kp=batch(c['kp']['data']),kd=batch(c['kd']['data']),
               rigidbody_mass=batch([sim.model.body(n).mass[0] for n in bodies]),
               rigidbody_com_offset=batch(np.concatenate([sim.model.body(n).ipos for n in bodies])),
               dof_friction=batch(sim.model.dof_frictionloss[sim.vadr]),
               dof_damping=batch(sim.model.dof_damping[sim.vadr]),
               rigid_shape_friction=batch(sim.model.geom_friction[1:,0]))
    state=dict(local_root_lin_vel=batch(sim.state()['root_rotmat'].T@sim.state()['root_lin_vel_world']),
               local_root_ang_vel=st.local_root_ang_vel,local_root_gravity=st.local_root_gravity,
               dof_pos=st.dof_pos,dof_vel=st.dof_vel)
    critic=critic_observation(config,state,setup,torch.zeros(1,72),torch.zeros(1,18))
    assert critic.shape==(1,269)


def test_as2_export_retains_binding_and_rejects_wrong_order(configs,tmp_path):
    from pawcerto.methods.umi_on_legs import UmiPolicy
    _,config=configs
    (tmp_path/'config.json').write_text(json.dumps(config))
    # Untrained weights are solely a serialization fixture, never policy evidence.
    model=build_algorithm(config,'cpu').actor_critic
    torch.save(dict(config=config,model_state_dict=model.state_dict()),tmp_path/'model.pt')
    policy=UmiPolicy(tmp_path/'model.pt')
    with pytest.raises(ValueError,match='Export joint order'):
        policy.export(tmp_path/'wrong',joint_order(configs[0]))
    policy.export(tmp_path/'export',joint_order(config))
    loaded=UmiPolicy(tmp_path/'export')
    assert loaded.config['pawcerto_robot']==config['pawcerto_robot']
    assert loaded.joint_names==joint_order(config)
    sim=Go2Arx5Mujoco(tmp_path/'export/config.json')
    assert sim.binding['root_body']=='base_link'
