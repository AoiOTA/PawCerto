"""CPU semantic checks; these do not evaluate WBC or claim policy learning."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch

from pawcerto.methods.umi_on_legs.training import (
    load_config, build_algorithm, UmiTrainer, PoseCurriculum, UmiReward, RewardState,
    critic_observation, termination,
)
from pawcerto.methods.umi_on_legs.training.semantics import runtime_contract,require_resume_contract


class TensorEnv:
    """Exercises transport/reset/optimization only, never a robot substitute."""
    num_envs = 4
    def __init__(self):
        self.obs = torch.randn(4, 132)
        self.critic = torch.randn(4, 261)
        self.randomized = False
    def reset(self):
        return self.obs, self.critic
    def randomize_episode_progress(self):
        self.randomized = True
    def step(self, action):
        self.obs.add_(.001)
        self.critic.add_(.002)
        return self.obs, self.critic, -action.square().mean(-1), torch.tensor([0,1,1,0]), {
            'time_outs': torch.tensor([0,0,1,0])}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_body_speed_revision_rejects_legacy_reconstruction_optimizer_resume(self):
        current = runtime_contract('reconstructed-solver')
        self.assertEqual(current['reconstruction_revision'], 'stock-physx-body-speed-damping-v3')
        legacy = dict(current)
        del legacy['reconstruction_revision']
        with self.assertRaisesRegex(ValueError, 'Resume requires matching'):
            require_resume_contract(legacy, current)
        with self.assertRaisesRegex(ValueError, 'Resume requires matching'):
            require_resume_contract(dict(current, reconstruction_revision='stock-physx-gyro-clamp-v1'), current)
        with self.assertRaisesRegex(ValueError, 'Resume requires matching'):
            require_resume_contract(dict(current, reconstruction_revision='stock-physx-velocity-clamps-v2'), current)
        require_resume_contract(current, current)
        normal = runtime_contract('normal-contact')
        self.assertNotIn('reconstruction_revision', normal)
        require_resume_contract(normal, normal)

    def test_original_velocity_limits_revision_rejects_old_physics_resume(self):
        for signal in ('reconstructed-solver', 'normal-contact'):
            with self.subTest(signal=signal):
                current = runtime_contract(signal)
                self.assertEqual(current['physics_revision'],
                                 'umi-physics-mapping-v3')
                for previous in ('all-body-rigid-properties-angular-rad-to-deg-v1',
                                 'all-body-properties-original-velocity-limits-v2'):
                    with self.assertRaisesRegex(ValueError, 'Resume requires matching'):
                        require_resume_contract(dict(current, physics_revision=previous), current)
                require_resume_contract(current, current)

    def test_relaxed_velocity_contract_and_weights_only_restart(self):
        original = runtime_contract('reconstructed-solver')
        self.assertEqual(original, dict(force_signal='reconstructed-solver',
            physics_revision='umi-physics-mapping-v3',
            joint_friction_model='physx-legacy-friction-coefficient',
            reconstruction_revision='stock-physx-body-speed-damping-v3'))
        relaxed = runtime_contract('reconstructed-solver', 1000)
        self.assertEqual(relaxed, dict(original, joint_velocity_limit_override_rad_s=1000.))
        for invalid in (0, -1, 999, float('nan'), float('inf'), True, '1000'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runtime_contract('reconstructed-solver', invalid)
        for saved, current in ((original, relaxed), (relaxed, original)):
            with self.assertRaisesRegex(ValueError, '--weights'):
                require_resume_contract(saved, current)
        require_resume_contract(relaxed, relaxed)
        cfg = load_config()
        cfg['pawcerto_runtime'] = original
        source = UmiTrainer(TensorEnv(), cfg, rollout_steps=4)
        source.train_iteration()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'source.pt'
            source.save(path)
            cfg['joint_velocity_limit_override_rad_s'] = 1000.
            cfg['pawcerto_runtime'] = relaxed
            target = UmiTrainer(TensorEnv(), cfg, rollout_steps=4)
            with self.assertRaisesRegex(ValueError, '--weights'):
                target.load(path)
            rng = torch.get_rng_state().clone()
            target.load(path, load_optimizer=False)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(target.iteration, 0)
            self.assertEqual(target.total_transitions, 0)
            self.assertEqual(len(target.alg.optimizer.state), 0)
            self.assertEqual(target.alg.learning_rate, cfg['runner']['alg']['learning_rate'])
            for name, value in source.alg.actor_critic.state_dict().items():
                self.assertTrue(torch.equal(value, target.alg.actor_critic.state_dict()[name]), name)

    def test_adapter_binds_explicit_velocity_variant(self):
        from pawcerto.methods.umi_on_legs.training.isaac_env import UmiIsaacTrainingEnv
        env = self.force_adapter_fixture('reconstructed-solver', False, velocity_override=1000.)
        self.assertEqual(env.runtime_metadata, runtime_contract('reconstructed-solver', 1000.))
        env.runtime.joint_velocity_limit_override_rad_s = None
        with self.assertRaisesRegex(ValueError, 'joint velocity overrides must agree'):
            UmiIsaacTrainingEnv(env.runtime, env.config, Path('not-reached.pkl'))

    def test_original_ppo_settings_and_network(self):
        cfg = load_config()
        alg = build_algorithm(cfg)
        self.assertEqual(alg.num_learning_epochs, 64)
        self.assertEqual(alg.num_mini_batches, 4)
        self.assertEqual(alg.min_lr, .0001)
        self.assertEqual(alg.actor_critic.actor[0].in_features,132)
        self.assertEqual(alg.actor_critic.critic[0].in_features,261)
        self.assertEqual(tuple(alg.actor_critic.act(torch.zeros(2,132)).shape),(2,18))

    def test_released_actor_and_critic_weights_load(self):
        root=Path(__file__).resolve().parents[1]
        for variant in ['ours','ours-real']:
            directory=root/'reference/checkpoints/tossing'/variant
            alg=build_algorithm(load_config(directory/'config.json'))
            checkpoint=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
            alg.actor_critic.load_state_dict(checkpoint['model_state_dict'],strict=True)
            self.assertTrue(torch.isfinite(alg.actor_critic.std).all())
            self.assertTrue((alg.actor_critic.std>0).all())

    def test_vendored_ppo_retains_original_source(self):
        root=Path(__file__).resolve().parents[1]
        source=root/'third_party/umi-on-legs/mani-centric-wbc/legged_gym/rsl_rl'
        dest=root/'pawcerto/methods/umi_on_legs/training'
        for src, dst in [('algorithms/ppo.py','ppo.py'),('storage/rollout_storage.py','storage.py'),
                         ('modules/actor_critic.py','actor_critic.py')]:
            expected=(source/src).read_text().replace('from git import Optional','from typing import Optional').replace(
                'from legged_gym.rsl_rl.modules import ActorCritic','from .actor_critic import ActorCritic').replace(
                'from legged_gym.rsl_rl.storage import RolloutStorage','from .storage import RolloutStorage').replace(
                'from legged_gym.rsl_rl.utils import split_and_pad_trajectories','from .trajectory_utils import split_and_pad_trajectories')
            self.assertEqual((dest/dst).read_text(),expected)

    def test_timeout_bootstrap_uses_original_pre_step_value(self):
        alg=build_algorithm(load_config())
        alg.init_storage(2,1,132,261,18,1)
        with torch.no_grad():
            alg.act(torch.zeros(2,132),torch.zeros(2,261))
            alg.transition.values.fill_(2.)
            alg.process_env_step(torch.ones(2),torch.ones(2),{'time_outs':torch.tensor([0,1])})
            alg.compute_returns(torch.zeros(2,261))
        torch.testing.assert_close(alg.storage.returns[:, :, 0],torch.tensor([[1.,2.98]]))

    def test_actual_cpu_update_and_resume(self):
        torch.manual_seed(9)
        cfg=load_config()
        trainer=UmiTrainer(TensorEnv(),cfg,rollout_steps=4)
        before=trainer.alg.actor_critic.actor[0].weight.detach().clone()
        stats=trainer.train_iteration()
        self.assertTrue(trainer.env.randomized)
        self.assertEqual(stats['total_transitions'],16)
        self.assertFalse(torch.equal(before,trainer.alg.actor_critic.actor[0].weight))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.pt'
            trainer.save(path)
            expected_random=torch.rand(4)
            restored=UmiTrainer(TensorEnv(),cfg,rollout_steps=4)
            restored.load(path)
            torch.testing.assert_close(torch.rand(4),expected_random)
            self.assertEqual(restored.iteration,1)
            self.assertEqual(restored.alg.learning_rate,trainer.alg.learning_rate)
            self.assertEqual(restored.train_iteration()['iteration'],2)

    def test_curriculum_global_ema_and_reversible_reset(self):
        cfg=load_config()
        c=PoseCurriculum(cfg['env']['tasks']['reaching'],2)
        self.assertEqual(c.pos_sigma,1.)
        self.assertEqual(c.orn_sigma,4.)
        c.step(torch.zeros(2),torch.zeros(2),.005)
        torch.testing.assert_close(c.past_pos_err,torch.full((2,),.99875))
        c.past_pos_err.fill_(.05)
        c.on_reset()
        self.assertEqual(c.pos_sigma,.005)
        c.past_pos_err.fill_(.9)
        c.on_reset()
        self.assertEqual(c.pos_sigma,1.)

    def reward_fixture(self):
        cfg=load_config()
        bodies=cfg['env']['constraints']['collision']['link_names']+['FR_foot','FL_foot','RR_foot','RL_foot']
        joints=[f'leg{i}' for i in range(12)]+[f'joint{i}' for i in range(1,7)]
        reward=UmiReward(cfg,bodies,joints,2)
        q=torch.tensor(cfg['env']['controller']['offset']['data']).repeat(2,1)
        z=lambda *shape:torch.zeros(*shape)
        s=RewardState(q,z(2,18),z(2,18),z(2,18),z(2,18),z(2,18),
                      torch.tensor([[0.,0.,-1.]]).repeat(2,1),torch.tensor([[0.,0.,.4]]).repeat(2,1),
                      z(2,len(bodies),3),torch.ones(2,4),z(2,len(bodies),3),z(2),z(2))
        return cfg,reward,s,bodies

    def test_reward_flying_clipping_and_constraint_activation(self):
        cfg,reward,s,_=self.reward_fixture()
        self.assertEqual(len(reward.constraints),12)
        self.assertNotIn('arm_action_rate',reward.constraints)
        r,terms=reward(s)
        torch.testing.assert_close(r,torch.full((2,),.08))
        s.feet_force_z[1]=0
        r,terms=reward(s)
        self.assertEqual(terms['constraint/even_mass_distribution'][1],-100.)
        self.assertEqual(r[1],0.)

    def test_reward_penalty_physics_dt_and_angles(self):
        _,reward,s,_=self.reward_fixture()
        s.dof_vel.fill_(1)
        s.orientation_error.fill_(1.)
        _,terms=reward(s)
        torch.testing.assert_close(terms['constraint/joint_acc'],torch.full((2,),-.18))
        torch.testing.assert_close(terms['task/reaching/pose'],torch.full((2,),4*__import__('math').exp(-1/4)))

    def test_critic_properties_order_and_no_noise(self):
        cfg=load_config()
        state,setup={},{}
        for name, data in [('privileged_state_obs',state),('privileged_setup_obs',setup)]:
            for key,spec in cfg['env'][name].items():
                if not key.startswith('_'):
                    data[spec['key']]=torch.ones(2,spec['dim'])
        obs=critic_observation(cfg,state,setup,torch.full((2,72),3.),torch.full((2,18),4.))
        self.assertEqual(obs.shape,(2,261))
        torch.testing.assert_close(obs[:,:18],torch.full((2,18),1.))
        torch.testing.assert_close(obs[:,18:36],torch.full((2,18),10.))
        torch.testing.assert_close(obs[:,126+42:126+45],torch.full((2,3),2.))
        torch.testing.assert_close(obs[:,-18:],torch.full((2,18),4.))

    def test_isaac_adapter_reset_and_rollout_lifecycle_on_cpu(self):
        # Tensor transport fixture only; PhysX remains independently required.
        import pickle
        import numpy as np
        from pawcerto.methods.umi_on_legs import RobotState
        from pawcerto.methods.umi_on_legs.training.isaac_env import UmiIsaacTrainingEnv
        cfg,reward,rs,bodies=self.reward_fixture()
        cfg['pawcerto_runtime']=runtime_contract('normal-contact')
        class Runtime:
            device='cpu'
            num_envs=2
            body_names=bodies
            joint_names=reward.joint_names
            force_signal='normal-contact'
            def __init__(self):
                self.time=torch.zeros(2)
                self.num_steps=0
                self.last_global_step=0
            def reset(self,ids):
                self.time[ids]=0
            def state(self):
                return RobotState(torch.zeros(2,3),rs.local_root_gravity,rs.dof_pos,rs.dof_vel,
                                  torch.eye(4).repeat(2,1,1),self.time)
            def joints(self):
                return rs.dof_pos,rs.dof_vel
            def step_torque(self,torque):
                self.time.add_(.005)
                self.num_steps+=1
            def training_state(self):
                return dict(root_pos=rs.root_pos,local_root_lin_vel=torch.zeros(2,3),
                            contact_forces=rs.contact_forces,feet_force_z=rs.feet_force_z,
                            body_pos=rs.body_pos)
            def training_setup(self):
                return {item['key']:torch.zeros(2,item['dim'])
                        for key,item in cfg['env']['privileged_setup_obs'].items()
                        if not key.startswith('_')}
            def apply_domain_randomization(self,step):
                self.last_global_step=step
        runtime=Runtime()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'poses.pkl'
            with path.open('wb') as stream:
                pickle.dump([dict(ee_pos=np.zeros((8,3)),ee_axis_angle=np.zeros((8,3)))],stream)
            env=UmiIsaacTrainingEnv(runtime,cfg,path)
            obs,critic=env.reset()
            self.assertEqual(runtime.num_steps,4)
            self.assertEqual(obs.shape,(2,132))
            self.assertEqual(critic.shape,(2,261))
            rs.contact_forces[1,bodies.index('base'),2]=2
            obs,critic,rewards,done,info=env.step(torch.zeros(2,18))
            self.assertEqual(done.tolist(),[False,True])
            self.assertEqual(runtime.time[1],0)
            self.assertEqual(runtime.last_global_step,2)
            self.assertEqual(info['time_outs'].tolist(),[False,False])
            torch.testing.assert_close(obs[1,42:-18],torch.full((72,),-1.))

    def force_adapter_fixture(self, force_signal, dense, fault=None, velocity_override=None):
        """Exercise the actual adapter/reward with solver force and zero contact."""
        import pickle
        import numpy as np
        from pawcerto.methods.umi_on_legs import RobotState
        from pawcerto.methods.umi_on_legs.training.isaac_env import UmiIsaacTrainingEnv
        cfg,reward,rs,bodies=self.reward_fixture()
        cfg['env']['dense_rewards']=dense
        if velocity_override is not None:
            cfg['joint_velocity_limit_override_rad_s']=velocity_override
        cfg['pawcerto_runtime']=runtime_contract(force_signal, velocity_override)
        class Runtime:
            device='cpu'
            num_envs=2
            body_names=bodies
            joint_names=reward.joint_names
            def __init__(self):
                self.force_signal=force_signal
                self.joint_velocity_limit_override_rad_s=velocity_override
                self.time=torch.zeros(2)
                self.valid=torch.zeros(2,dtype=torch.bool)
            def reset(self,ids):
                self.time[ids]=0
                self.valid[ids]=False
            def state(self):
                return RobotState(torch.zeros(2,3),rs.local_root_gravity,rs.dof_pos,rs.dof_vel,
                                  torch.eye(4).repeat(2,1,1),self.time)
            def joints(self):return rs.dof_pos,rs.dof_vel
            def step_torque(self,torque):
                self.time.add_(.005)
                self.valid.fill_(True)
            def training_state(self):
                raw=dict(root_pos=rs.root_pos,local_root_lin_vel=torch.zeros(2,3),
                         contact_forces=torch.zeros_like(rs.contact_forces),
                         feet_force_z=torch.zeros(2,4),body_pos=rs.body_pos,
                         feet_solver_force_z=torch.tensor([[1.,1.,1.,1.],[-2.,0.,2.,6.]]),
                         feet_solver_force_valid=self.valid.clone())
                if fault=='missing':raw.pop('feet_solver_force_z')
                if fault=='missing-validity':raw.pop('feet_solver_force_valid')
                if fault=='invalid':raw['feet_solver_force_valid'][1]=False
                if fault=='nonfinite':raw['feet_solver_force_z'][1,0]=float('nan')
                return raw
            def training_setup(self):
                return {item['key']:torch.zeros(2,item['dim'])
                        for key,item in cfg['env']['privileged_setup_obs'].items()
                        if not key.startswith('_')}
            def apply_domain_randomization(self,step):pass
        td=tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path=Path(td.name)/'poses.pkl'
        with path.open('wb') as stream:
            pickle.dump([dict(ee_pos=np.zeros((8,3)),ee_axis_angle=np.zeros((8,3)))],stream)
        return UmiIsaacTrainingEnv(Runtime(),cfg,path)

    def test_emd_solver_source_dense_and_sparse_with_zero_contact(self):
        for dense in (False,True):
            with self.subTest(dense=dense):
                env=self.force_adapter_fixture('reconstructed-solver',dense)
                obs,critic,_,done,info=env.step(torch.zeros(2,18))
                key=('reward/' if dense else '')+'constraint/even_mass_distribution'
                # clip[-2,0,2,6] -> [0,0,2,6], normalized unbiased variance=1/8.
                expected=torch.tensor([0.,-.125])*(.02 if dense else 1.)
                torch.testing.assert_close(info[key],expected)
                torch.testing.assert_close(info['supported_feet'],torch.zeros(2))
                self.assertEqual(done.tolist(),[False,False])
                self.assertEqual(obs.shape,(2,132))
                self.assertEqual(critic.shape,(2,261))

    def test_emd_explicit_normal_proxy_keeps_original_flying_branch(self):
        for dense in (False,True):
            with self.subTest(dense=dense):
                env=self.force_adapter_fixture('normal-contact',dense,fault='invalid')
                _,_,_,_,info=env.step(torch.zeros(2,18))
                key=('reward/' if dense else '')+'constraint/even_mass_distribution'
                torch.testing.assert_close(info[key],torch.full((2,),-100.*(.02 if dense else 1.)))
                torch.testing.assert_close(info['supported_feet'],torch.zeros(2))

    def test_emd_reconstruction_missing_invalid_and_nonfinite_propagate(self):
        for dense in (False,True):
            for fault in ('missing','missing-validity','invalid','nonfinite'):
                with self.subTest(dense=dense,fault=fault):
                    env=self.force_adapter_fixture('reconstructed-solver',dense,fault)
                    with self.assertRaises(FloatingPointError if fault=='nonfinite' else RuntimeError):
                        env.step(torch.zeros(2,18))

    def test_runtime_contract_is_saved_and_resume_requires_same_semantics(self):
        cfg=load_config()
        cfg['pawcerto_runtime']=runtime_contract('reconstructed-solver')
        cfg['training_initialization']=dict(mode='random',checkpoint=None,source_pawcerto_runtime=None)
        trainer=UmiTrainer(TensorEnv(),cfg,rollout_steps=4)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.pt'
            trainer.save(path)
            saved=torch.load(path,map_location='cpu',weights_only=False)['config']
            changed=copy.deepcopy(cfg)
            changed['pawcerto_runtime']=runtime_contract('normal-contact')
            target=UmiTrainer(TensorEnv(),changed,rollout_steps=4)
            before=target.alg.actor_critic.actor[0].weight.detach().clone()
            with self.assertRaisesRegex(ValueError,'--weights'):
                target.load(path)
            torch.testing.assert_close(target.alg.actor_critic.actor[0].weight,before)
            target.load(path,load_optimizer=False)
            self.assertEqual(target.iteration,0)
            self.assertEqual(len(target.alg.optimizer.state),0)
            torch.testing.assert_close(target.alg.actor_critic.actor[0].weight,
                                       trainer.alg.actor_critic.actor[0].weight)
            legacy=torch.load(path,map_location='cpu',weights_only=False)
            del legacy['config']['pawcerto_runtime']
            torch.save(legacy,path)
            with self.assertRaisesRegex(ValueError,'--weights'):
                trainer.load(path)
        self.assertEqual(saved['pawcerto_runtime'],cfg['pawcerto_runtime'])
        self.assertEqual(saved['training_initialization']['mode'],'random')
        require_resume_contract(saved['pawcerto_runtime'],cfg['pawcerto_runtime'])
        for old in (None,runtime_contract('normal-contact'),
                    dict(cfg['pawcerto_runtime'],joint_friction_model='physx-static-friction-effort')):
            with self.subTest(old=old),self.assertRaisesRegex(ValueError,'--weights'):
                require_resume_contract(old,cfg['pawcerto_runtime'])
        env=self.force_adapter_fixture('reconstructed-solver',False)
        state=env.training_state_dict()
        self.assertEqual(state['pawcerto_runtime'],cfg['pawcerto_runtime'])
        env.load_training_state_dict(state)
        del state['pawcerto_runtime']
        with self.assertRaisesRegex(ValueError,'--weights'):
            env.load_training_state_dict(state)

    def test_default_trainer_binds_actual_adapter_contract_and_rejects_legacy(self):
        env=self.force_adapter_fixture('reconstructed-solver',False)
        trainer=UmiTrainer(env,rollout_steps=4)
        self.assertEqual(trainer.config['pawcerto_runtime'],env.runtime_metadata)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.pt'
            trainer.save(path)
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual(checkpoint['config']['pawcerto_runtime'],env.runtime_metadata)
            checkpoint.pop('env_training_state')
            del checkpoint['config']['pawcerto_runtime']
            torch.save(checkpoint,path)
            before=trainer.alg.actor_critic.actor[0].weight.detach().clone()
            with self.assertRaisesRegex(ValueError,'--weights'):
                trainer.load(path)
            torch.testing.assert_close(trainer.alg.actor_critic.actor[0].weight,before)
        explicit=load_config()
        bound=UmiTrainer(env,explicit,rollout_steps=4)
        self.assertEqual(bound.config['pawcerto_runtime'],env.runtime_metadata)
        self.assertNotIn('pawcerto_runtime',explicit)
        explicit['pawcerto_runtime']=runtime_contract('normal-contact')
        with self.assertRaisesRegex(ValueError,'actual environment'):
            UmiTrainer(env,explicit,rollout_steps=4)

    def test_termination_exact_boundary_and_safe_bounds(self):
        cfg=load_config()
        force=torch.zeros(3,2,3);force[0,0,2]=1.01
        done,timeouts=termination(cfg,force,['base','FR_foot'],torch.tensor([0,850,851]),torch.zeros(3,3))
        self.assertEqual(done.tolist(),[True,False,True])
        self.assertEqual(timeouts.tolist(),[False,False,True])
        root=torch.zeros(3,3);root[1,0]=2
        _,timeouts=termination(cfg,force,['base','FR_foot'],torch.zeros(3),root,torch.tensor([[-1.,1.],[-1.,1.]]))
        self.assertEqual(timeouts.tolist(),[False,True,False])

class EvaluationEpisodeTests(unittest.TestCase):
    def test_completed_episodes_are_equally_weighted_and_prefixes_excluded(self):
        from pawcerto.methods.umi_on_legs.training.evaluation import CompletedEpisodeStats
        stats=CompletedEpisodeStats(2,5)
        stats.add(torch.tensor([1.,10.]),torch.tensor([False,True]),
                  {'position_error':torch.tensor([1.,10.]),'time_outs':torch.tensor([False,False])})
        stats.add(torch.tensor([3.,30.]),torch.tensor([False,True]),
                  {'position_error':torch.tensor([3.,30.]),'time_outs':torch.tensor([False,True])})
        stats.add(torch.tensor([5.,50.]),torch.tensor([True,False]),
                  {'position_error':torch.tensor([5.,50.]),'time_outs':torch.tensor([True,False])})
        summary=stats.summary()
        self.assertEqual(summary['actual_completed_episodes'],3)
        self.assertEqual(summary['unfinished_episode_count'],1)
        self.assertEqual(summary['unfinished_prefix_steps'],[0,1])
        self.assertAlmostEqual(summary['metrics']['position_error/mean'],(10+30+3)/3)
        self.assertAlmostEqual(summary['metrics']['timeout_survival_proxy'],2/3)
        self.assertEqual(summary['early_termination_metrics']['position_error/mean'],10.)
        self.assertEqual(summary['timeout_episode_metrics']['position_error/mean'],16.5)

    def test_author_window_evicts_old_completions_but_stop_counts_them(self):
        from pawcerto.methods.umi_on_legs.training.evaluation import CompletedEpisodeStats
        stats=CompletedEpisodeStats(2,2)
        stats.add(torch.zeros(2),torch.ones(2,dtype=torch.bool),
                  {'position_error':torch.tensor([1.,100.]),'time_outs':torch.zeros(2,dtype=torch.bool)})
        self.assertFalse(stats.reached_author_stop())
        stats.add(torch.zeros(2),torch.tensor([True,False]),
                  {'position_error':torch.tensor([2.,999.]),'time_outs':torch.zeros(2,dtype=torch.bool)})
        self.assertTrue(stats.reached_author_stop())
        summary=stats.summary()
        self.assertEqual(summary['actual_completed_episodes'],3)
        self.assertEqual(summary['included_completed_episodes'],2)
        self.assertEqual(summary['evicted_completed_episodes'],1)
        self.assertEqual(summary['metrics']['position_error/mean'],51.)

    def test_eval_global_stop_is_strict_and_trial_is_excluded(self):
        from types import SimpleNamespace
        from pawcerto.methods.umi_on_legs.training.evaluation import evaluate_completed_episodes
        class Env:
            num_envs=2
            device='cpu'
            controller=SimpleNamespace(dt=.5,decimation=1)
            def __init__(self,complete):
                self.complete=complete;self.local_step=0;self.reset_count=0
            def reset(self):
                self.local_step=0;self.reset_count+=1
                return torch.zeros(2,1),torch.zeros(2,1)
            def step(self,action):
                self.local_step+=1
                done=torch.full((2,),self.complete and self.local_step%2==0,dtype=torch.bool)
                return torch.zeros(2,1),torch.zeros(2,1),torch.ones(2),done,{
                    'position_error':torch.ones(2),'time_outs':done}
        class Model:
            def __init__(self):self.samples=0;self.means=0
            def eval(self):pass
            def act(self,obs):self.samples+=1;return obs
            def act_inference(self,obs):self.means+=1;return obs
        cfg={'runner':{'num_eval_episode_per_env':2,'alg':{'gamma':.99}},
             'env':{'cfg':{'env':{'episode_length_s':1.}}}}
        env,model=Env(True),Model()
        report,episodes=evaluate_completed_episodes(env,model,cfg)
        self.assertEqual(model.samples,1)
        self.assertEqual(model.means,6)
        self.assertEqual(env.reset_count,2)
        self.assertEqual(report['actual_completed_episodes'],6)
        self.assertEqual(report['included_completed_episodes'],4)
        self.assertEqual(report['policy_steps_executed'],6)
        self.assertEqual(len(episodes),6)
        self.assertEqual(report['stop_reason'],'global_completed_episodes_exceeded_nominal_target')
        report,episodes=evaluate_completed_episodes(Env(False),Model(),cfg)
        self.assertEqual(report['stop_reason'],'step_limit')
        self.assertEqual(report['policy_steps_executed'],8)
        self.assertEqual(report['unfinished_episode_count'],2)
        self.assertEqual(report['actual_completed_episodes'],0)
        self.assertIsNone(report['metrics'])

if __name__=='__main__':
    unittest.main()
