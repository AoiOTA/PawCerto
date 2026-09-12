"""CPU source equivalence and runner protocol tests; no simulated learning claims."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from pawcerto.methods.roboduet import default_config, build_models
from pawcerto.methods.roboduet.training.ppo import PPO
from pawcerto.methods.roboduet.training.rollout_storage import RolloutStorage
from pawcerto.methods.roboduet.training.runner import RoboDuetRunner

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'third_party/roboduet-reference/go1_gym_learn/ppo_cse_automatic'


def compact_config():
    cfg = default_config()
    for side in ['Dog', 'Arm']:
        cfg[side+'AC_Args'].update(actor_hidden_dims=[32, 24, 16], critic_hidden_dims=[32, 24, 16],
                                 adaptation_module_branch_hidden_dims=[16])
    cfg['RunnerArgs']['num_steps_per_env'] = 2
    cfg['PPO_Args'].update(num_learning_epochs=1, num_mini_batches=1)
    cfg['stage'].update(pretrained_to_hybrid_start=0, pretrained_to_hybrid_end=0)
    return cfg


def upstream_ppo(config):
    """Execute the actual source, replacing only unavailable import dependencies."""
    tree = ast.parse((SOURCE/'ppo.py').read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)
                 or node.module not in ('params_proto', 'arm_ac', 'rollout_storage')]
    class PrefixProto:
        pass
    storage_tree = ast.parse((SOURCE/'rollout_storage.py').read_text())
    storage_tree.body = [node for node in storage_tree.body if not isinstance(node, ast.ImportFrom)]
    storage_namespace = {}
    exec(compile(storage_tree,str(SOURCE/'rollout_storage.py'),'exec'),storage_namespace)
    namespace = dict(PrefixProto=PrefixProto, ArmActorCritic=object,
                     RolloutStorage=storage_namespace['RolloutStorage'])
    exec(compile(tree, str(SOURCE/'ppo.py'), 'exec'), namespace)
    for key, value in config.items():
        setattr(namespace['PPO_Args'], key, value)
    return namespace['PPO']


@pytest.mark.parametrize('side', [0, 1])
def test_ppo_matches_fixed_source(side):
    torch.set_num_threads(1)
    torch.manual_seed(981)
    cfg = compact_config()
    model = build_models(cfg)[side]
    reference = upstream_ppo(cfg['PPO_Args'])(deepcopy(model))
    actual = PPO(deepcopy(model), config=cfg['PPO_Args'])
    sc = cfg['Cfg']['dog' if side == 0 else 'arm']
    name = 'dog' if side == 0 else 'arm'
    dims = [sc[name+'_num_observations'], sc[name+'_num_privileged_obs'], sc[name+'_num_obs_history'],
            sc['dog_actions' if side == 0 else 'num_actions_arm_cd']]
    for alg in [reference, actual]:
        alg.init_storage(5, 2, [dims[0]], [dims[1]], [dims[2]], [dims[3]], [dims[3]])
    inputs = [(torch.randn(5, dims[0]), torch.randn(5, dims[1]), torch.randn(5, dims[2]),
               torch.rand(5), torch.tensor([False, False, True, False, False])) for _ in range(2)]
    outputs = []
    for alg in [reference, actual]:
        torch.manual_seed(42)
        with torch.inference_mode():
            for obs, privileged, history, rewards, dones in inputs:
                alg.act(obs, privileged, history)
                alg.process_env_step(rewards, dones, {'time_outs': dones})
            alg.compute_returns(history, privileged)
        outputs.append((alg.storage.returns.clone(), alg.storage.advantages.clone(), alg.update()))
    for i in (0, 1):
        torch.testing.assert_close(outputs[0][i], outputs[1][i], rtol=0, atol=0)
    assert outputs[0][2] == outputs[1][2]
    for key, value in reference.actor_critic.state_dict().items():
        torch.testing.assert_close(value, actual.actor_critic.state_dict()[key], rtol=0, atol=0)
    assert reference.learning_rate == actual.learning_rate


class ProtocolEnv:
    """Explicit deterministic test double: tests ordering/checkpoint, never physics."""
    def __init__(self, cfg):
        self.cfg, self.num_envs, self.calls = cfg['Cfg'], 5, []
        self.histories = {s: torch.zeros(5, self.cfg[s][s+'_num_obs_history']) for s in ('dog','arm')}
        self.counter = 0
        self.episode_length_buf = torch.zeros(5,dtype=torch.long)
        self.max_episode_length = 1000

    def randomize_episode_lengths(self):
        self.episode_length_buf = torch.randint_like(self.episode_length_buf,high=self.max_episode_length)

    def set_stage(self, stage):
        self.stage = stage

    def set_iteration(self, iteration):
        self.iteration = iteration

    def reset(self):
        self.counter = 0
        for value in self.histories.values(): value.zero_()

    def get_observations(self, side):
        self.calls.append(side)
        cfg = self.cfg[side]
        obs = torch.full((5, cfg[side+'_num_observations']), self.counter/10)
        history = torch.cat((self.histories[side][:, obs.shape[1]:], obs), -1)
        self.histories[side] = history
        return dict(obs=obs, privileged_obs=torch.full((5,cfg[side+'_num_privileged_obs']), .2), obs_history=history)

    def get_arm_observations(self): return self.get_observations('arm')
    def get_dog_observations(self): return self.get_observations('dog')
    def plan(self, action): self.calls.append('plan')

    def step(self, dog, arm):
        self.calls.append('step')
        self.counter += 1
        return -dog.square().mean(-1), -arm.square().mean(-1), torch.zeros(5,dtype=torch.bool), {}

    def clear_cached(self, ids):
        for history in self.histories.values(): history[ids] = 0

    def training_state(self):
        return dict(counter=self.counter, histories=self.histories,episode_length_buf=self.episode_length_buf)

    def load_training_state(self, state):
        self.counter, self.histories = state['counter'], state['histories']
        self.episode_length_buf = state['episode_length_buf']

    def restore_arm_observation_cache(self, cache):
        cache['obs_history'] = self.histories['arm']
        return cache


def test_stage_order_four_optimizers_and_exact_cpu_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(101)
    cfg = compact_config()
    env = ProtocolEnv(cfg)
    runner = RoboDuetRunner(env, cfg)
    arm_before = deepcopy(runner.arm_model.state_dict())
    dog_before = deepcopy(runner.dog_model.state_dict())
    stats = runner.run_iteration()
    assert stats['rollout_stage'] == 1 and runner.stage == 2 and runner.next_iteration == 1
    assert runner.arm_update_count == stats['arm_update_count'] == 0
    assert env.calls == ['arm','dog','step','dog','step','dog']
    assert all(torch.equal(arm_before[k], v) for k,v in runner.arm_model.state_dict().items())
    assert any(not torch.equal(dog_before[k],v) for k,v in runner.dog_model.state_dict().items())
    path = tmp_path/'stage_boundary.pt'
    runner.save(path)
    env.calls.clear()
    expected = runner.run_iteration()
    assert env.calls == ['plan','dog','step','arm','plan','dog','step','arm','plan','dog']
    assert expected['arm_losses'] is not None
    assert runner.arm_update_count == expected['arm_update_count'] == 1
    expected_dog, expected_arm = deepcopy(runner.dog_model.state_dict()), deepcopy(runner.arm_model.state_dict())
    fresh = RoboDuetRunner(ProtocolEnv(cfg), cfg)
    fresh.load(path)
    torch.testing.assert_close(fresh.env.episode_length_buf,env.episode_length_buf,rtol=0,atol=0)
    assert fresh.next_iteration == 1 and fresh.stage == 2
    assert fresh.arm_update_count == 0
    assert fresh.run_iteration() == expected
    for model, reference in [(fresh.dog_model, expected_dog), (fresh.arm_model, expected_arm)]:
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, reference[key], rtol=0, atol=0)
    fresh.save(path)
    payload = torch.load(path, weights_only=False)
    for side in ['dog','arm']:
        assert payload[side+'_optimizer']['state']
        assert payload[side+'_adaptation_optimizer']['state']
    assert payload['arm_update_count'] == 1
    assert payload['torch_rng'].dtype == torch.uint8 and payload['torch_rng'].device.type == 'cpu'
    assert any(not torch.equal(arm_before[k],v) for k,v in fresh.arm_model.state_dict().items())


def test_fresh_episode_age_matches_actual_seeded_upstream_equation():
    from types import SimpleNamespace
    from pawcerto.methods.roboduet.training.isaac_env import RoboDuetIsaacTrainingEnv
    tree = ast.parse((SOURCE/'__init__.py').read_text())
    learn = next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name == 'learn')
    assignment = learn.body[0].body[0]
    code = compile(ast.Module(body=[assignment],type_ignores=[]),str(SOURCE/'__init__.py'),'exec')
    reference = SimpleNamespace(episode_length_buf=torch.zeros(4096,dtype=torch.long),max_episode_length=1000)
    actual = SimpleNamespace(task=SimpleNamespace(episode_length_buf=torch.zeros(4096,dtype=torch.long),max_episode_length=1000))
    torch.manual_seed(123)
    exec(code,dict(torch=torch,self=SimpleNamespace(env=reference)))
    reference_rng = torch.get_rng_state()
    torch.manual_seed(123)
    RoboDuetIsaacTrainingEnv.randomize_episode_lengths(actual)
    torch.testing.assert_close(actual.task.episode_length_buf,reference.episode_length_buf,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),reference_rng)


def test_actual_task_observer_controller_wiring_with_static_state_fixture(tmp_path):
    """Exercise assembly and reward shapes using explicit static (not physics) data."""
    from pawcerto.methods.roboduet.training.isaac_env import RoboDuetIsaacTrainingEnv
    cfg = compact_config()
    class StaticState:
        device = 'cpu'
        num_envs = 5
        dt = .005
        joint_names = [f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR')
                       for joint in ('hip','thigh','calf')] + [f'zarx_j{i}' for i in range(1,9)]
        body_names = ['base','FL_foot','FR_foot','RL_foot','RR_foot','zarx_body6']
        torque_limits = torch.full((20,),30.)
        dof_pos_limits = torch.tensor([[-3.,3.]]*20)
        def __init__(self):
            self.q = torch.tensor([[cfg['Cfg']['init_state']['default_joint_angles'][n]
                                    for n in self.joint_names]]).repeat(5,1)
            self.qd = torch.zeros_like(self.q)
            self.pose = torch.tensor([[0.,0.,.34,1.,0.,0.,0.]]).repeat(5,1)
            self.velocity = torch.zeros(5,6)
        def training_state(self): return deepcopy(vars(self))
        def load_training_state(self,state): self.__dict__.update(deepcopy(state))
        def set_material_properties(self,friction,restitution):
            self.friction,self.restitution = friction.clone(),restitution.clone()
        def set_base_mass_delta(self,mass,body_name):
            assert body_name == 'trunk'
            self.mass=mass.clone()
        def set_gravity(self,gravity): self.gravity=gravity.clone()
        def set_stage(self,stage): self.stage=stage
        def reset(self,ids,joint_pos,root_pose,root_velocity):
            self.q[ids],self.pose[ids],self.velocity[ids] = joint_pos,root_pose,root_velocity
            self.qd[ids]=0
        def joints(self): return self.q,self.qd
        def step_control(self,leg_torque,arm_target, *, capture_state=True):
            assert leg_torque.shape == (5,12) and arm_target.shape == (5,8)
        def state(self):
            body_pos = torch.zeros(5,6,3)
            body_pos[:,0] = self.pose[:,:3]
            body_pos[:,-1] = torch.tensor([.3,0.,.6])
            body_quat = torch.tensor([1.,0.,0.,0.]).repeat(5,6,1)
            return dict(root_pos=self.pose[:,:3],root_quat=self.pose[:,3:],
                        root_lin_vel=self.velocity[:,:3],root_ang_vel=self.velocity[:,3:],
                        joint_pos=self.q.clone(),joint_vel=self.qd.clone(),body_pos=body_pos,
                        body_lin_vel=torch.zeros(5,6,3),contact_forces=torch.zeros(5,6,3),
                        ee_pos=body_pos[:,-1],ee_quat=body_quat[:,-1],
                        ee_lin_vel=torch.zeros(5,3),ee_ang_vel=torch.zeros(5,3))
    runtime = StaticState()
    env = RoboDuetIsaacTrainingEnv(runtime,cfg)
    env.set_stage(1)
    env.reset()
    assert env.get_dog_observations()['obs'].shape == (5,56)
    assert env.get_arm_observations()['privileged_obs'].shape == (5,9)
    env.set_stage(2)
    env.set_iteration(1)
    env.plan(torch.zeros(5,2))
    dog,arm,done,infos = env.step(torch.zeros(5,12),torch.zeros(5,6))
    assert dog.shape == arm.shape == done.shape == (5,)
    assert torch.isfinite(dog).all() and torch.isfinite(arm).all()
    torch.testing.assert_close(runtime.friction,env.buffers['friction_coeffs'][:,0])

    # A timeout followed by a non-timeout must not reuse the old timeout mask.
    env.task.episode_length_buf[:] = env.task.max_episode_length
    _,_,_,timeout_infos = env.step(torch.zeros(5,12),torch.zeros(5,6))
    assert timeout_infos['time_outs'].all()
    _,_,_,next_infos = env.step(torch.zeros(5,12),torch.zeros(5,6))
    assert not next_infos['time_outs'].any()
    # This uses the actual task/curriculum/history checkpoint, still no physics.
    runner = RoboDuetRunner(env,cfg)
    runner.run_iteration()
    path = tmp_path/'task_state.pt'
    runner.save(path)
    expected = runner.run_iteration()
    fresh = RoboDuetRunner(RoboDuetIsaacTrainingEnv(StaticState(),cfg),cfg)
    fresh.load(path)
    assert fresh.run_iteration() == expected


def test_early_full_v1_checkpoint_arm_count_migration():
    from pawcerto.methods.roboduet.training.runner import checkpoint_arm_update_count, FORMAT
    for next_iteration, expected in [(0,0),(10000,0),(10001,0),(10002,1),(10003,2)]:
        assert checkpoint_arm_update_count(dict(format=FORMAT,config=default_config(),
                                                next_iteration=next_iteration)) == expected
