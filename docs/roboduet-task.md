# RoboDuet task tensors

`pawcerto.methods.roboduet.training.task.RoboDuetTask` ports the default Go1
`automatic` task from RoboDuet revision
`a7e1528215c048199f90cb69ceb7749a1d745f28`. It has no simulator dependency.
The environment supplies physical measurements and controller histories; the task
owns command sampling, command curriculum, gait/contact targets, reward sums,
episode counters and termination decisions.

## Environment interface

Construct `RoboDuetTask(config, num_envs, device, dt=None)` with `default_config()`
(or its `Cfg` dictionary). Full configuration preserves the `stage` schedule.
Bind real tensors with `update_physics(**state)` before reset or stepping.

Required upstream names and ordering:

- `root_states[N,13]`, `end_effector_state[N,13]`: world xyz, xyzw quaternion,
  world linear/angular velocity. The task computes base-frame velocities and
  projected gravity from root state and `gravity_vec[N,3]` (default downward unit
  vector; randomized gravity direction must be supplied by the environment).
- `foot_positions[N,4,3]`, `foot_velocities[N,4,3]`,
  `contact_forces[N,B,3]`, `feet_indices[4]`, `penalised_contact_indices[K]`.
- `dof_pos`, `dof_vel`, `torques`, `last_dof_vel`, `joint_pos_target`,
  `last_joint_pos_target`, `last_last_joint_pos_target`: `[N,20]` in upstream
  order. `default_dof_pos[1,20]`, `dof_pos_limits[20,2]` use the source's soft
  joint limits (not raw URDF limits).
- `actions`, `last_actions`, `last_last_actions`: `[N,18]`;
  `plan_actions`, `last_plan_actions`: `[N,2]`.
- `measured_heights`: zero for the default plane. Nondefault terrain and vision
  are outside the validated default training path.

Initial reset: bind physical/controller tensors, then `reset(all_ids)`.
Each control step: update physical/controller tensors, `advance()`,
`check_termination()` returning `(done, timeouts)`, then `compute_reward()`
returning `(dog_reward, arm_reward)`. Before resetting physics for done IDs,
call `reset(done_ids)`, which returns per-environment episode reward sums.
Build post-reset observations, then advance controller histories as in upstream.
The controller/environment owns reset of action and velocity history; task reset
intentionally retains the source's last-contact history behavior.

Read `commands_dog`, `commands_arm_obs`, `clock_inputs`, and other named task
buffers directly for observation assembly. The controller updates dog pitch/roll
commands from plan actions in stage 2. `set_iteration(count, switch_open=flag)`
sets reward progress and release independently. Omitting the flag uses
`count > pretrained_to_hybrid_start`, preserving the source runner's one-rollout
lag between the reward schedule and arm release. `set_stage('stage1'/'stage2')`
only changes arm release and terminal-reward settings; it does not overwrite the
reward schedule count.

`state_dict()` / `load_state_dict()` preserve task tensors, command bins,
curriculum weights and its private RNG, schedule/release flags and accumulators.
The checkpoint owner must additionally save physics/controller state and global
NumPy/Torch RNG states: arm trajectory sampling uses global NumPy and uniform
command sampling uses global Torch, as upstream.

## Fidelity and evidence

All 33 default active reward functions and their source ordering are retained,
including zero-weight hybrid terms. Arm rewards omit only upstream's two walking
velocity tracking terms. Both diagnostic EE rewards accumulate in episode sums
and do not enter learning rewards. Reward scaling by control `dt`, batch-sum sign
classification, positive-reward shaping, contact reward offsets for curriculum,
and strict timeout `episode_length > max_episode_length` are preserved.

Position reward evaluates a point 0.1 m forward of the EE link, in yaw-aligned
coordinates with ground-relative z offset 0.38 m. Orientation uses the source's
axis-projection angles, not substituted Euler-angle error. Pitch termination is
gated by greater than 60 percent of the current arm trajectory interval.

The CPU port skips arm resampling while the arm is closed and samples
uninitialized trajectories upon release. Upstream applies integer remainder to
zero trajectory lengths while closed; this operation is not portable to CPU.
This is a declared initialization adaptation, not a validated statement about
that CUDA edge case. NumPy's removed `np.int` is replaced by `np.int64` and boolean
conversion uses `.numpy()` with unchanged values.

Run:

```sh
/home/lyb/miniconda3/envs/pawcerto-lab-sim610/bin/python -m unittest discover -s tests -p test_roboduet_task.py -v
```

Five CPU tests execute pinned upstream class/function bodies on the same inputs
for all active reward terms and aggregation, sampled commands and targets,
standing/moving contact clocks, threshold curriculum, termination boundaries,
stage reward schedule, and checkpoint continuation. The comparisons isolate
simulator imports; shared quaternion helpers are independently owned by the core
observation module. They establish tensor semantics, not Isaac Lab contact-signal
equivalence, complete simulator replay, learning, or policy performance.
