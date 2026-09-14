"""Fixed-weight UMI-on-Legs execution, preserving released checkpoint semantics.

Adapted from UMI-on-Legs (Huy Ha and Yihuai Gao, 2024), MIT license;
see LICENSE. The owning reference is third_party/umi-on-legs/mani-centric-wbc.
This adapter
implements its observation, trajectory and position-control equations; it is
not a training algorithm or evidence of successful robot control.
"""
from dataclasses import dataclass
import json
import io
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from pawcerto.artifacts import file_identity

# Joint arrays must be supplied in the original simulator asset DOF order.
# The live importer owns this mapping; config comments are not authoritative.


def axis_angle_to_matrix(v):
    # Rodrigues with sinc keeps zero-angle trajectories finite.
    theta = torch.linalg.vector_norm(v, dim=-1)
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)
    return (torch.eye(3, dtype=v.dtype, device=v.device)
            + torch.sinc(theta / torch.pi)[..., None, None] * k
            + (0.5 * torch.sinc(theta / (2 * torch.pi)).square())[..., None, None] * (k @ k))


@dataclass
class RobotState:
    local_root_ang_vel: torch.Tensor
    local_root_gravity: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    ee_pose: torch.Tensor
    episode_time: torch.Tensor


class PoseSequence:
    def __init__(self, file_path, config, device="cpu"):
        self.device = device
        with open(file_path, 'rb') as stream:
            episodes = pickle.load(stream)
        selection = config.get('trajectory_selection')
        self.trajectory_ids = list(range(len(episodes)))
        if selection:
            from .data_split import source_hash
            if source_hash(file_path) != selection['source_sha256']:
                raise ValueError('Trajectory source differs from configured split')
            self.trajectory_ids = selection['trajectory_ids']
            if not self.trajectory_ids or len(set(self.trajectory_ids)) != len(self.trajectory_ids) or any(i < 0 or i >= len(episodes) for i in self.trajectory_ids):
                raise ValueError('Invalid trajectory IDs')
            episodes = [episodes[i] for i in self.trajectory_ids]
        self.positions = torch.from_numpy(np.stack([ep['ee_pos'] for ep in episodes])).float()
        self.rotations = axis_angle_to_matrix(torch.from_numpy(np.stack([ep['ee_axis_angle'] for ep in episodes])).float())
        self.config = config

    def sample(self, batch_size, seed):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randint(len(self.positions), (batch_size,), generator=generator)
        return self._select(indices, generator)

    def select_ids(self, trajectory_ids, seed=0):
        """Select source IDs exactly once in caller order, without replacement."""
        indices = torch.tensor([self.trajectory_ids.index(i) for i in trajectory_ids], dtype=torch.long)
        return self._select(indices, torch.Generator().manual_seed(seed))

    def _select(self, indices, generator):
        batch_size = len(indices)
        length = int(self.config['episode_length_s'] / self.config['dt'])
        pos = self.positions[indices, :length].clone()
        rot = self.rotations[indices, :length].clone()
        if pos.shape[1] < length:
            extra = length - pos.shape[1]
            pos = torch.cat((pos, pos[:, -1:].expand(-1, extra, -1)), 1)
            rot = torch.cat((rot, rot[:, -1:].expand(-1, extra, -1, -1)), 1)
        if self.config['planar_center']:
            pos[..., :2] -= pos[:, [1, 2, 3], :2].mean(1, keepdim=True)
        height = self.config.get('add_random_height_range')
        if height is not None:
            pos[..., 2] += torch.rand((batch_size, 1), generator=generator) * (height[1] - height[0]) + height[0]
        return pos.to(self.device), rot.to(self.device)


class UmiObservation:
    """Call advance_pose after task.step cadence, observe at the policy boundary.

    Released dense_rewards=false updates poses once per 20ms policy step,
    despite pose_latency being configured in 5ms simulation units. Preserve
    that actual reference behavior. Reset histories are identity sentinels.
    Observation noise is disabled by default for fixed-weight evaluation.
    """
    def __init__(self, config, positions, rotations):
        self.config = config
        self.env = config['env']
        self.task = self.env['tasks']['reaching']
        self.positions, self.rotations = positions, rotations
        self.device = positions.device
        self.batch_size = positions.shape[0]
        self.frames = int(np.rint(self.task['pose_latency'] / self.task['sequence_sampler']['dt'])) + 1
        self.pose_history = torch.eye(4, device=self.device).repeat(self.batch_size, self.frames, 1, 1)
        self.obs_history = torch.zeros(self.batch_size, self.env['obs_history_len'], config['env']['cfg']['env']['num_observations'], device=self.device)

    def reset(self, indices=None):
        ids = slice(None) if indices is None else indices
        self.pose_history[ids] = torch.eye(4, device=self.device)
        self.obs_history[ids] = 0

    def advance_pose(self, ee_pose):
        self.pose_history = torch.cat((self.pose_history[:, 1:], ee_pose[:, None].clone()), 1)

    def targets(self, times):
        steps = (times / self.task['sequence_sampler']['dt']).long().clamp(0, self.positions.shape[1] - 1)
        rows = torch.arange(self.batch_size, device=self.device)[:, None] if steps.ndim == 2 else torch.arange(self.batch_size, device=self.device)
        return self.positions[rows, steps], self.rotations[rows, steps]

    def observe(self, state, last_action, generator=None):
        attrs = []
        # IsaacGymEnv.__init__ sorts observation attributes before concatenation.
        for name, attr in sorted(self.env['state_obs'].items()):
            if name.startswith('_'):
                continue
            value = getattr(state, attr['key'])
            if attr['scale'] is not None:
                value = value * attr['scale']
            if generator is not None and attr['noise_std']:
                value = value + torch.randn(value.shape, generator=generator, device=self.device) * attr['noise_std']
            if attr['offset'] is not None:
                value = value - torch.tensor(attr['offset']['data'], device=self.device)
            if attr['clip'] is not None:
                value = value.clamp(-attr['clip'], attr['clip'])
            attrs.append(value)
        if self.task['target_relative_to_base']:
            raise ValueError('This EE-state adapter serves released ours/ours-real, not body-space checkpoints')
        times = state.episode_time[:, None] + torch.tensor(self.task['target_obs_times'], device=self.device)
        pos, rot = self.targets(times)
        pose = self.pose_history[:, -max(1, self.frames - 1)]
        inverse = torch.linalg.inv(pose)
        local_pos = (inverse[:, None, :3, :3] @ pos[..., None]).squeeze(-1) + inverse[:, None, :3, 3]
        local_rot = inverse[:, None, :3, :3] @ rot
        p = (local_pos * self.task['pos_obs_scale']).flatten(1)
        if self.task['pos_obs_clip'] is not None:
            p = p.clamp(-self.task['pos_obs_clip'], self.task['pos_obs_clip'])
        r = (local_rot[..., :2, :] * self.task['orn_obs_scale']).flatten(1)
        task_obs = torch.cat((p, r), -1)
        task_obs[(pose == torch.eye(4, device=self.device)).all(-1).all(-1)] = -1
        obs = torch.cat((*attrs, task_obs, last_action), -1)
        self.obs_history = torch.cat((self.obs_history[:, 1:], obs[:, None]), 1)
        return self.obs_history.flatten(1)


class UmiController:
    def __init__(self, config, batch_size, device="cpu"):
        from .actuation import actuation_mode, piper_target_limits, AS2_NATIVE_SERVO
        self.actuation_mode = actuation_mode(config)
        self.native_servo = self.actuation_mode == AS2_NATIVE_SERVO
        bounds = piper_target_limits(config)
        self.piper_limits = None if bounds is None else torch.tensor(bounds, device=device)
        self.device = device
        env = config['env']
        ctrl = env['controller']
        self.decimation = ctrl['decimation_count']
        self.dt = env['cfg']['sim']['dt']
        self.max_action = env['max_action_value']
        self.offset, self.scale, self.kp, self.kd, self.torque_limit = [torch.tensor(ctrl[k]['data'], device=device) for k in ('offset', 'scale', 'kp', 'kd', 'torque_limit')]
        self.delay_steps = torch.round(torch.tensor(env['ctrl_delay']['data'], device=device) / self.dt)
        self.buffer = torch.zeros(batch_size, env['ctrl_buf_len'], ctrl['control_dim'], device=device)

    @property
    def last_action(self):
        return self.buffer[:, 0]

    def reset(self, indices=None):
        self.buffer[slice(None) if indices is None else indices] = 0

    def push(self, action):
        action = action.clamp(-self.max_action, self.max_action)
        self.buffer = torch.cat((action[:, None], self.buffer[:, :-1]), 1)

    def position_target(self, substep):
        indices = torch.ceil((self.delay_steps - substep) / self.decimation).long()
        action = self.buffer.permute(2, 1, 0)[torch.arange(len(indices), device=self.device), indices].T
        target = self.offset + self.scale * action
        if self.piper_limits is not None:
            target[:, 12:] = target[:, 12:].clamp(self.piper_limits[0], self.piper_limits[1])
        return target

    def torque(self, dof_pos, dof_vel, substep):
        """External effort or native servo pre-state estimate, selected explicitly."""
        target = self.position_target(substep)
        return (self.kp * (target - dof_pos) - self.kd * dof_vel).clamp(-self.torque_limit, self.torque_limit)

    def step(self, runtime, dof_pos, dof_vel, substep):
        """Keep the original 5 ms sampling while dispatching the selected actuator."""
        effort = self.torque(dof_pos, dof_vel, substep)
        if self.native_servo:
            runtime.step_servo(self.position_target(substep))
        else:
            runtime.step_torque(effort)
        return effort


class UmiPolicy:
    """Load an original directory, training model_N.pt, or exported actor directory."""
    def __init__(self, checkpoint_dir, device="cpu"):
        self.device = device
        source = Path(checkpoint_dir)
        directory = source.parent if source.is_file() else source
        self.config = json.loads((directory / 'config.json').read_text())
        names_path = directory / 'joint_names.json'
        self.joint_names = json.loads(names_path.read_text()) if names_path.exists() else None
        # Execution config beside a model can be overwritten by another run.
        # Only the weights' embedded metadata qualifies their training partition.
        self.training_selection = None
        self.training_asset = None
        self.training_runtime = None
        self.source_config = file_identity(directory / 'config.json')
        if source.is_dir() and (directory / 'actor.ts').exists():
            from .actuation import actuation_mode, AS2_NATIVE_SERVO
            provenance_path = directory / 'export.json'
            provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
            if (actuation_mode(self.config) == AS2_NATIVE_SERVO
                    or provenance.get('execution_actuation_mode') == AS2_NATIVE_SERVO):
                if provenance.get('execution_config', {}).get('sha256') != self.source_config['sha256']:
                    raise ValueError('Native-servo export execution config differs from its saved identity')
            self.source_weights = file_identity(directory / 'actor.ts')
            self.actor = torch.jit.load(str(directory / 'actor.ts'), map_location=device).eval()
        else:
            ac = self.config['runner']['alg']['actor_critic']
            layers = []
            for spec in ac['actor']['_args_']:
                cls = {'torch.nn.Linear': nn.Linear, 'torch.nn.ELU': nn.ELU}[spec['_target_']]
                layers.append(cls(**{k: v for k, v in spec.items() if not k.startswith('_')}))
            self.actor = nn.Sequential(*layers).eval()
            weights_path = source if source.is_file() else directory / 'model.pt'
            self.source_weights = file_identity(weights_path)
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            from .robot_binding import require_same_robot
            require_same_robot(self.config, checkpoint.get('config', {}))
            from .actuation import require_same_actuation
            require_same_actuation(self.config, checkpoint.get('config', {}))
            from .data_split import checkpoint_training_selection
            self.training_selection = checkpoint_training_selection(checkpoint)
            self.training_asset = checkpoint.get('config', {}).get('pawcerto_asset')
            self.training_runtime = checkpoint.get('config', {}).get('pawcerto_runtime')
            self.actor.load_state_dict({k[6:]: v for k, v in checkpoint['model_state_dict'].items() if k.startswith('actor.')})
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.actor.to(device)

    def export(self, output_dir, joint_names):
        """Write the CPU TorchScript actor and the execution config after reload parity.

        The model emits raw actions. Consumers retain this module's observation,
        action delay, scaling and PD path. A fresh output directory preserves
        existing exports and unrelated files.
        """
        if torch.device(self.device).type != 'cpu':
            raise ValueError('Export uses a CPU UmiPolicy; load the source with device="cpu"')
        dim = self.config['runner']['alg']['actor_critic']['num_actor_obs']
        actions = self.config['runner']['alg']['actor_critic']['num_actions']
        from .robot_binding import joint_order
        if list(joint_names) != joint_order(self.config):
            raise ValueError('Export joint order differs from the saved UMI robot binding')
        if len(joint_names) != actions or len(set(joint_names)) != actions:
            raise ValueError(f'Expected {actions} unique asset joint names in policy order')
        blob = io.BytesIO()
        torch.jit.save(torch.jit.script(self.actor), blob)
        blob.seek(0)
        restored = torch.jit.load(blob, map_location='cpu').eval()
        example = torch.linspace(-1., 1., 4 * dim).reshape(4, dim)
        with torch.inference_mode():
            expected, actual = self.actor(example), restored(example)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / 'actor.ts').write_bytes(blob.getvalue())
        (output_dir / 'config.json').write_text(json.dumps(self.config, indent=2) + '\n')
        (output_dir / 'joint_names.json').write_text(json.dumps(joint_names, indent=2) + '\n')
        # Preserve provenance for inspection without promoting an actor-only
        # package to independently verified training-partition evidence.
        provenance = {
            'format': 'pawcerto.umi.export.v1',
            'source_weights': self.source_weights,
            'source_config': self.source_config,
            'actor': file_identity(output_dir / 'actor.ts'),
            'checkpoint_training_selection': self.training_selection,
            'checkpoint_training_asset': self.training_asset,
            'checkpoint_training_runtime': self.training_runtime,
            'execution_runtime': self.config.get('pawcerto_runtime'),
            'execution_actuation_mode': self.config.get('actuation_mode', 'external-pd'),
            'execution_config': file_identity(output_dir / 'config.json'),
        }
        (output_dir / 'export.json').write_text(json.dumps(provenance, indent=2) + '\n')
        return {'output': str(output_dir), 'max_actor_error': (actual - expected).abs().max().item(),
                'observation_dim': dim, 'action_dim': actions}

    def observer(self, positions, rotations):
        return UmiObservation(self.config, positions.to(self.device), rotations.to(self.device))

    def controller(self, batch_size):
        return UmiController(self.config, batch_size, self.device)

    def trajectories(self, file_path):
        return PoseSequence(file_path, self.config['env']['tasks']['reaching']['sequence_sampler'], self.device)
