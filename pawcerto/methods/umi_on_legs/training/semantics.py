"""Tensor adapters for the released tossing config's active UMI task/constraints.

Ported from mani-centric-wbc/legged_gym/env/isaacgym/{task,constraints,env}.py.
Inputs use official DOF order, world positions and an explicitly selected EMD signal. Compute
at each physics substep for dense evaluation, or at the final substep for sparse
training, before resetting environments.
"""
from dataclasses import dataclass
import torch


FORCE_SIGNALS = ('reconstructed-solver', 'normal-contact')


def runtime_contract(force_signal, joint_velocity_limit_override_rad_s=None):
    """Minimal reward/physics provenance saved in both config and checkpoints."""
    if force_signal not in FORCE_SIGNALS:
        raise ValueError(f'Unsupported EMD force signal: {force_signal}')
    contract = dict(force_signal=force_signal,
                    physics_revision='umi-physics-mapping-v3',
                    joint_friction_model='physx-legacy-friction-coefficient')
    if force_signal == 'reconstructed-solver':
        contract['reconstruction_revision'] = 'stock-physx-body-speed-damping-v3'
    if joint_velocity_limit_override_rad_s is not None:
        # One explicit research condition, not a general physics override API.
        if (type(joint_velocity_limit_override_rad_s) not in (int, float)
                or joint_velocity_limit_override_rad_s != 1000):
            raise ValueError('joint_velocity_limit_override_rad_s must be 1000 or omitted')
        contract['joint_velocity_limit_override_rad_s'] = 1000.
    return contract


def require_resume_contract(saved, current):
    """Optimizer continuation must not silently cross reward or physics semantics."""
    if saved is None or current is None or saved != current:
        raise ValueError('Resume requires matching pawcerto_runtime metadata in the checkpoint '
                         f'and current config; saved={saved!r}, current={current!r}. '
                         'Use --weights explicitly to initialize a new training run instead.')


def emd_force_z(raw, force_signal, num_envs):
    """Select only the EMD input; normal-contact diagnostics remain separate.

    The runtime owns freshness: reset/state writes invalidate the per-env bool
    until a supported, finite reconstruction from the next real physics substep.
    """
    if force_signal == 'reconstructed-solver':
        if 'feet_solver_force_z' not in raw or 'feet_solver_force_valid' not in raw:
            raise RuntimeError('EMD requires the current reconstructed solver force and validity mask')
        valid = raw['feet_solver_force_valid']
        if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
                or valid.shape != (num_envs,) or not bool(valid.all())):
            raise RuntimeError('EMD reconstructed solver force is missing, invalid or stale')
        force = raw['feet_solver_force_z']
    elif force_signal == 'normal-contact':
        force = raw['feet_force_z']
    else:
        raise ValueError(f'Unsupported EMD force signal: {force_signal}')
    if not isinstance(force, torch.Tensor) or force.shape != (num_envs, 4):
        raise ValueError('EMD force must be a tensor with shape [num_envs, 4]')
    if not force.is_floating_point() or not bool(torch.isfinite(force).all()):
        raise FloatingPointError('EMD force must contain finite floating-point values')
    return force


@dataclass
class RewardState:
    dof_pos: torch.Tensor                 # [N,18], official DOF order
    dof_vel: torch.Tensor                 # [N,18]
    prev_dof_vel: torch.Tensor            # previous PHYSICS substep, not control step
    torque: torch.Tensor                  # applied/clamped torque, final substep
    action: torch.Tensor                  # raw action, before target scaling
    prev_action: torch.Tensor
    local_root_gravity: torch.Tensor      # normalized gravity, body frame [N,3]
    root_pos: torch.Tensor                # world [N,3]
    contact_forces: torch.Tensor          # [N,B,3], port normal-contact forces, body_names ordering
    feet_force_z: torch.Tensor            # [N,4], explicitly selected EMD world-z input; other contacts stay separate
    body_pos: torch.Tensor                # [N,B,3], world
    position_error: torch.Tensor          # [N], world EE-target Euclidean distance
    orientation_error: torch.Tensor       # [N], SO(3) geodesic angle, radians


class PoseCurriculum:
    def __init__(self, task, num_envs, device='cpu'):
        self.task = task
        self.alpha = task['smoothing_dt_multiplier']
        for short, config_key in [('pos', 'pos'), ('orn', 'orn')]:
            schedule = task[f'{config_key}_sigma_curriculum']
            items = sorted(((float(k), float(v)) for k, v in schedule.items()), reverse=True)
            level = task[f'init_{config_key}_curriculum_level'] or 0
            setattr(self, f'{short}_schedule', items)
            setattr(self, f'{short}_level', level)
            setattr(self, f'{short}_sigma', items[level][1])
            setattr(self, f'past_{short}_err', torch.full((num_envs,), items[level][0], device=device))

    def step(self, position_error, orientation_error, physics_dt):
        alpha = physics_dt * self.alpha
        self.past_pos_err.mul_(1-alpha).add_(position_error, alpha=alpha)
        self.past_orn_err.mul_(1-alpha).add_(orientation_error, alpha=alpha)

    def on_reset(self):
        # Original reset uses the GLOBAL EMA mean, does not clear EMA on reset,
        # and permits moving back to earlier (easier) levels.
        for short in ('pos', 'orn'):
            avg = getattr(self, f'past_{short}_err').mean().item()
            for level, (threshold, sigma) in enumerate(getattr(self, f'{short}_schedule')):
                if avg < threshold:
                    setattr(self, f'{short}_sigma', sigma)
                    setattr(self, f'{short}_level', level)

    def state_dict(self):
        return {key: getattr(self, key) for key in
                ['past_pos_err', 'past_orn_err', 'pos_sigma', 'orn_sigma', 'pos_level', 'orn_level']}

    def load_state_dict(self, state):
        for key, value in state.items():
            existing = getattr(self, key)
            if isinstance(existing, torch.Tensor):
                existing.copy_(value)
            else:
                setattr(self, key, value)


def critic_observation(config, state, setup, task_obs, action):
    """Exact 261-D ours / 225-D real order, without actor noise or actor history.

    state/setup are mappings using keys in privileged_state_obs/setup_obs;
    setup values are the actual randomized properties, not nominal placeholders.
    task_obs is the same delayed task pose observation used by the actor.
    """
    output = []
    for group_name, values in [('privileged_setup_obs', setup), ('privileged_state_obs', state)]:
        for name, spec in sorted(config['env'][group_name].items()):
            if name.startswith('_'):
                continue
            value = values[spec['key']].reshape(action.shape[0], -1)
            if value.shape[1] != spec['dim']:
                raise ValueError(f'{name}: expected {spec["dim"]} entries, got {value.shape[1]}')
            if spec.get('scale') is not None:
                value = value * spec['scale']
            if spec.get('offset') is not None:
                value = value - value.new_tensor(spec['offset']['data'])
            if spec.get('clip') is not None:
                value = value.clamp(-spec['clip'], spec['clip'])
            output.append(value)
    result = torch.cat([*output, task_obs, action], dim=-1)
    expected = config['runner']['alg']['actor_critic']['num_critic_obs']
    if result.shape[-1] != expected:
        raise ValueError(f'Critic expected {expected} values, got {result.shape[-1]}')
    return result


class UmiReward:
    def __init__(self, config, body_names, joint_names, num_envs, device='cpu'):
        self.config = config
        env = config['env']
        self.physics_dt = env['cfg']['sim']['dt']
        self.policy_dt = self.physics_dt * env['controller']['decimation_count']
        self.reward_dt = self.physics_dt if env['dense_rewards'] else self.policy_dt
        self.curriculum = PoseCurriculum(env['tasks']['reaching'], num_envs, device)
        self.task = env['tasks']['reaching']
        # Only Hydra partial objects become constraints in official Env.__init__.
        self.constraints = {k: v for k, v in env['constraints'].items()
                            if isinstance(v, dict) and v.get('_partial_') and '_target_' in v}
        self.body_names, self.joint_names = list(body_names), list(joint_names)
        self.indices = {}
        for name, spec in self.constraints.items():
            if spec.get('terminate_on_violation') or spec.get('violation_weight', 0) != 0:
                raise ValueError(f'{name}: adapter expects the released soft-only constraints')
            if 'link_names' in spec:
                self.indices[name] = [self.body_names.index(n) for n in spec['link_names']]
            elif 'link_1' in spec:
                self.indices[name] = (self.body_names.index(spec['link_1']), self.body_names.index(spec['link_2']))
            elif spec.get('joint_names'):
                self.indices[name] = [self.joint_names.index(n) for n in spec['joint_names']]
            else:
                self.indices[name] = list(range(len(joint_names)))
        limits = self.constraints['joint_limit']
        lower = torch.tensor(limits['lower']['data'], device=device)
        upper = torch.tensor(limits['upper']['data'], device=device)
        mid, width = (lower + upper) / 2, upper-lower
        self.soft_lower = mid - 0.5 * width * limits['penalty_scale']
        self.soft_upper = mid + 0.5 * width * limits['penalty_scale']

    def __call__(self, s: RewardState):
        self.curriculum.step(s.position_error, s.orientation_error, self.physics_dt)
        pos = torch.exp(-s.position_error.square() / self.curriculum.pos_sigma)
        orn = torch.exp(-s.orientation_error / self.curriculum.orn_sigma)
        terms = {'task/reaching/pos': pos*self.task['pos_reward_scale'],
                 'task/reaching/orn': orn*self.task['orn_reward_scale'],
                 'task/reaching/pose': pos*orn*self.task['pose_reward_scale']}
        scales = self.config['env']['cfg']['rewards']['scales']
        if set(scales) - {'orientation'}:
            raise ValueError(f'Unsupported base reward terms: {set(scales)-{"orientation"}}')
        terms['orientation'] = s.local_root_gravity[:, :2].square().sum(-1) * scales['orientation']
        for name, cfg in self.constraints.items():
            kind = cfg['_target_'].rsplit('.', 1)[-1]
            idx = self.indices[name]
            if kind == 'JointLimit':
                penalty = ((self.soft_lower-s.dof_pos).clamp(min=0) +
                           (s.dof_pos-self.soft_upper).clamp(min=0)).sum(-1)
            elif kind == 'Collision':
                penalty = (s.contact_forces[:, idx].norm(dim=-1) > cfg['penalty_scale']).sum(-1)
            elif kind == 'ActionRateLimit':
                penalty = (s.action[:, idx]-s.prev_action[:, idx]).abs().pow(cfg['power']).sum(-1)
            elif kind == 'JointAccelerationLimit':
                penalty = ((s.prev_dof_vel-s.dof_vel)/self.physics_dt).square().sum(-1)
            elif kind == 'TorqueLimit':
                penalty = s.torque[:, idx].abs().pow(cfg['power']).sum(-1)
            elif kind == 'EvenMassDistribution':
                force = s.feet_force_z.clamp(min=0)
                fraction = force / (force.sum(-1, keepdim=True)+1e-8)
                penalty = torch.where(fraction.sum(-1)>1e-8,
                                      fraction.std(-1).pow(cfg['power']), cfg['flying_penalty'])
            elif kind == 'LinkPosePair':
                if cfg['mode'] != 'exact' or not cfg['planar'] or cfg['orientation_weight'] != 0:
                    raise ValueError(f'Unsupported LinkPosePair config: {name}')
                distance = (s.body_pos[:, idx[0], :2]-s.body_pos[:, idx[1], :2]).norm(dim=-1)
                penalty = (1-torch.exp(-(distance-cfg['target_distance']).abs()/cfg['distance_sigma']))*cfg['position_weight']
            elif kind == 'StayCloseToDefaultConfig':
                default = s.dof_pos.new_tensor(cfg['default_config']['data'])
                penalty = (s.dof_pos[:, idx]-default).abs().pow(cfg['power']).sum(-1)
            elif kind == 'EnergyUsage':
                penalty = (s.torque*s.dof_vel).pow(cfg['power']).sum(-1)
            elif kind == 'RootHeight':
                penalty = (s.root_pos[:, 2]-cfg['target_height']).square()
            else:
                raise ValueError(f'Unsupported active official constraint: {name} ({kind})')
            terms[f'constraint/{name}'] = penalty*cfg['penalty_weight']
        reward = sum(terms.values())
        if self.config['env']['cfg']['rewards']['only_positive_rewards']:
            reward = reward.clamp(min=0)
        return reward*self.reward_dt, terms


def termination(config, contact_forces, body_names, episode_step, root_pos, safe_bounds=None):
    """Return done and original time_outs; exact > boundary, not >=."""
    env = config['env']
    patterns = env['cfg']['asset']['terminate_after_contacts_on']
    indices = [i for i, name in enumerate(body_names) if any(pattern in name for pattern in patterns)]
    contact_done = (contact_forces[:, indices].norm(dim=-1)>1.).any(-1)
    policy_dt = env['cfg']['sim']['dt'] * env['controller']['decimation_count']
    max_steps = int(__import__('math').ceil(env['cfg']['env']['episode_length_s']/policy_dt))
    timeout = episode_step > max_steps
    if safe_bounds is not None:
        timeout = timeout | ((root_pos[:, :2]<safe_bounds[:, 0]) |
                             (root_pos[:, :2]>safe_bounds[:, 1])).any(-1)
    return contact_done | timeout, timeout
