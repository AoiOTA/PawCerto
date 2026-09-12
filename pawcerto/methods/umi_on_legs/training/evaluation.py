"""Checkpoint configuration plus the author's completed-episode eval semantics.

Source: config/eval.yaml, scripts/evaluate.py, runners/on_policy_runner.py and
runners/utils.py in the bundled UMI-on-Legs tree. This is not the current Hydra
starter configuration: observation scales and latency follow the saved actor.
"""
from collections import deque
from copy import deepcopy
import math
import torch


DEFAULT_EVAL_SEED = 2026


def author_eval_config(checkpoint_config, trajectory_path, device, seed=DEFAULT_EVAL_SEED):
    """Resolve the preselected 250-env/nominal-500-episode evaluation condition."""
    config = deepcopy(checkpoint_config)
    config['seed'] = seed
    config['use_noise'] = True
    env = config['env']
    env['seed'] = seed
    env['dense_rewards'] = True
    env['controller']['seed'] = seed
    env['controller']['num_envs'] = 250
    env['cfg']['env']['num_envs'] = 250
    env['cfg']['env']['episode_length_s'] = 17.
    env['cfg']['env']['send_timeouts'] = True
    env['cfg']['domain_rand']['push_robots'] = False
    env['cfg']['domain_rand']['transport_robots'] = False
    env['cfg']['init_state'].update(pos=[-.5,0.,.3],pos_noise=[.05,.05,.001],euler_noise=[.01,.01,.1])
    env['cfg']['asset']['terminate_after_contacts_on'] = [
        'base','hip','Head','thigh','base_arm_link','link1','link2','link3','link4','link5']
    sampler = env['tasks']['reaching']['sequence_sampler']
    sampler.update(episode_length_s=17.,add_random_height_range=None,file_path=str(trajectory_path))
    env['constraints']['energy'] = dict(
        _target_='legged_gym.env.isaacgym.constraints.EnergyUsage',_partial_=True,
        violation_weight=0.,penalty_weight=-1e-7,terminate_on_violation=False,power=2,
        skip_stats=False,
        torque_constant=dict(_target_='torch.tensor',device=device,data=[.63895]*12+[1.4]*3+[.424]*3),
        voltage=dict(_target_='torch.tensor',device=device,data=[30.]*18))
    config['runner']['num_eval_episode_per_env'] = 2
    config['runner']['init_at_random_ep_len'] = False
    def set_device(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('device','sim_device'):
                    value[key] = device
                else:
                    set_device(item)
        elif isinstance(value,list):
            for item in value:
                set_device(item)
    set_device(config)
    return config


class CompletedEpisodeStats:
    """Equal episode weighting, retaining the latest N completed episodes.

    The global stopping count includes every done, even episodes subsequently
    evicted from the author's bounded deque. Unfinished prefixes never enter the
    completed-episode mean. Timeouts are episode-terminal indicators, not an
    invented tracking threshold. Extra instability diagnostics remain separate.
    """
    def __init__(self,num_envs,retained_count,device='cpu',policy_dt=.02,gamma=.99):
        self.num_envs = num_envs
        self.retained_count = retained_count
        self.device = device
        self.policy_dt = policy_dt
        self.gamma = gamma
        self.length = torch.zeros(num_envs,device=device)
        self.sums = {}
        self.minimum_up = torch.full((num_envs,),float('inf'),device=device)
        self.completed_per_env = torch.zeros(num_envs,dtype=torch.long,device=device)
        self.completed_count = 0
        self.records = []
        self.window = deque(maxlen=retained_count)

    def add(self,reward,done,info):
        self.length += 1
        values = dict(info,environment_reward=reward,
                      discounted_environment_return=reward*(self.gamma**self.length))
        for key,value in values.items():
            if value.shape != self.length.shape:
                raise ValueError(f'{key}: expected per-environment scalar {tuple(self.length.shape)}, got {tuple(value.shape)}')
            value = value.to(self.device)
            if key == 'root_up_dot_min':
                self.minimum_up = torch.minimum(self.minimum_up,value)
                continue
            if not torch.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite evaluation metric {key}')
            if key not in self.sums:
                self.sums[key] = torch.zeros_like(self.length)
            self.sums[key] += value.float()
        for env_id in done.nonzero().flatten().tolist():
            steps = int(self.length[env_id].item())
            self.completed_per_env[env_id] += 1
            record = dict(env_id=env_id,episode_index=int(self.completed_per_env[env_id]),
                          steps=steps,observed_duration_s=steps*self.policy_dt)
            for key,total in self.sums.items():
                value = float(total[env_id].item())
                record[key+'/sum'] = value
                record[key+'/mean'] = value/steps
                total[env_id] = 0.
            record['time_out'] = bool(info['time_outs'][env_id])
            # This diagnostic is independent of the original contact termination.
            up = float(self.minimum_up[env_id].item())
            record['root_up_dot_min'] = up if math.isfinite(up) else None
            record['inverted'] = up<0
            self.minimum_up[env_id] = float('inf')
            self.length[env_id] = 0.
            self.completed_count += 1
            self.records.append(record)
            self.window.append(record)

    def reached_author_stop(self):
        # Deliberately strict >, matching OnPolicyRunner.eval (not >=).
        return self.completed_count > self.retained_count

    @staticmethod
    def _aggregate(records):
        if not records:
            return None
        numeric_keys = [key for key in records[0] if key.endswith('/sum') or key.endswith('/mean')]
        result = {key:sum(r[key] for r in records)/len(records) for key in numeric_keys}
        result.update(completed_episodes=len(records),
                      episode_length_mean_steps=sum(r['steps'] for r in records)/len(records),
                      episode_duration_mean_s=sum(r['observed_duration_s'] for r in records)/len(records),
                      timeout_survival_proxy=sum(r['time_out'] for r in records)/len(records),
                      inverted_completed_episodes=sum(r['inverted'] for r in records))
        up = [r['root_up_dot_min'] for r in records if r['root_up_dot_min'] is not None]
        result['minimum_root_up_dot'] = min(up) if up else None
        return result

    def summary(self):
        selected = list(self.window)
        return dict(actual_completed_episodes=self.completed_count,
                    included_completed_episodes=len(selected),
                    evicted_completed_episodes=self.completed_count-len(selected),
                    unfinished_episode_count=int((self.length>0).sum().item()),
                    unfinished_prefix_steps=self.length.long().cpu().tolist(),
                    completed_episodes_per_env=self.completed_per_env.cpu().tolist(),
                    metrics=self._aggregate(selected),
                    timeout_episode_metrics=self._aggregate([r for r in selected if r['time_out']]),
                    early_termination_metrics=self._aggregate([r for r in selected if not r['time_out']]))


def evaluate_completed_episodes(env,actor_critic,config,progress=None):
    """No optimizer updates; retain the author's stochastic trial before eval."""
    with torch.inference_mode():
        actor_critic.eval()
        trial_obs,_ = env.reset()
        trial_action = actor_critic.act(trial_obs)
        env.step(trial_action)
        # The trial is excluded from episode aggregates, as in evaluate.py.
        obs,_ = env.reset()
        policy_dt = env.controller.dt*env.controller.decimation
        nominal = env.num_envs*config['runner']['num_eval_episode_per_env']
        max_episode_length = int(config['env']['cfg']['env']['episode_length_s']/policy_dt)
        max_steps = max_episode_length*(config['runner']['num_eval_episode_per_env']+2)
        collector = CompletedEpisodeStats(env.num_envs,nominal,env.device,policy_dt,
                                          config['runner']['alg']['gamma'])
        reason = 'step_limit'
        for step in range(max_steps):
            action = actor_critic.act_inference(obs)
            obs,_,reward,done,info = env.step(action)
            collector.add(reward,done,info)
            if progress is not None and ((step+1)%100==0 or collector.reached_author_stop()):
                progress(step+1,collector.completed_count)
            if collector.reached_author_stop():
                reason = 'global_completed_episodes_exceeded_nominal_target'
                break
        result = collector.summary()
        result.update(nominal_completed_episode_target=nominal,policy_steps_executed=step+1,
                      max_policy_steps=max_steps,stop_reason=reason,
                      aggregate_scope='Latest nominal-target completed episodes in completion/env-index order',
                      trial_rollout_included=False,warmup_included=False,
                      incomplete_prefixes_included=False,
                      reward_reporting='Raw environment reward; no legacy held-trial-value timeout bootstrap')
        return result,collector.records
