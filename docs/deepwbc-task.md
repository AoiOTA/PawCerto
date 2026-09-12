# DeepWBC task tensor boundary

`pawcerto.methods.deepwbc.task.DeepWBCTask` implements the task mechanics from
DeepWBC commit `8159e4ed8695b2d3f62a40d2ab8d88205ac5021a`. It imports neither
Isaac Gym nor Isaac Lab and does not generate robot motion. This is an executable
CPU equation/lifecycle boundary, not a simulator or a learning-success result.

## Runtime sequence

1. Create `DeepWBCTask(N, config)` with the full `default_config()` dictionary.
   `dt` is simulator timestep times controller decimation (public default 20 ms).
2. Call `sample_creation_randomization()` once and apply the returned base added
   mass, added COM xyz, gripper added mass, friction, motor strengths, and box
   added mass to the physical model. Box lateral offsets sample both signs,
   retaining the original left/right placement distribution. Persist these same values for privileged
   observations. The distribution is reconstructed with Torch; exact mixed
   NumPy/Torch random-stream parity is not claimed. Friction uses 1,000 buckets,
   including the original negative lower bound. Backend interpretation of
   negative friction requires explicit verification; this code does not clamp it.
3. Reset all rows with `reset(ids, default_dof_pos=..., env_origins=..., start=True)`.
   Apply returned root/DOF position/velocity writes in the simulator. Position
   writes for the second actor come from `reset_box_position(root, box_delta_y)`;
   only its XY components are used on episodic reset. Box Z, orientation and
   velocity are untouched in the original reset. Reset the
   separate observer and controller histories for those rows.
4. Run the separate controller through every physics substep. Refresh state and
   call `bind_state(...)` with final substep torque and delayed simulator-order
   actions. Inputs are SI tensors: root `[N,13]` (position, xyzw quaternion,
   world linear/angular velocity), joint position/velocity/torque `[N,20]`,
   actions `[N,18]`, world EE position `[N,3]`, xyzw EE quaternion `[N,4]`, body
   contact forces `[N,B,3]`, four six-axis foot wrenches `[N,4,6]`, and actual
   termination-contact body indices. Simulator DOFs must be 12 legs, six arm,
   two fingers. No placeholder sensor force is synthesized.
5. `after_physics()` returns `(leg_reward, arm_reward, done, extras)` before reset.
   If extras contains `push_velocity_xy`, write those world base XY velocities
   at callback timing; upstream computes body velocities before this push.
   Reset done rows, apply writes, refresh state and call `bind_state` again.
   Construct observations with observer inputs `commands`,
   `curr_ee_goal_sphere`, `ee_goal_delta_orn_euler`, and `episode_length_buf`.
   Then `finish_step()` stores last actions/DOF velocity. The enclosing env returns
   the original six fields `(obs, None, leg_reward, arm_reward, done, extras)`.
   Preserve terminal rewards/dones/timeouts across reset. Forward reset's
   `episode` dictionary in extras for original episodic logging.
6. `update_command_curriculum()` is called by the training runner on its original
   schedule, not automatically on every environment step or reset.

Reward terms beyond the active default may require additional physical inputs,
accepted by `bind_state` as named optional tensors: projected gravity, measured
heights, penalized/foot indices, position/velocity/torque limits and default DOFs.
Missing required physical inputs fail instead of receiving invented values.
The default does not measure terrain heights. Enabling terrain curriculum or
height sampling requires backend implementation beyond this CPU boundary.

## Preserved original details

- Original task reward and goal methods are vendored with their license. Rewards
  are split into leg and arm channels; neither is multiplied by `dt`; both totals
  are divided by 100. Episode sums retain the pre-division values. Optional
  positive clipping precedes termination penalty. Timeouts have no terminal
  penalty. Arm energy uses the original `12:-2` slice of 20 torques.
- LPY interpolation uses sampled fixed per-environment trajectory/hold durations,
  with goal timer increment after interpolation and resampling on strict `>`.
  Durations are not rerolled for every goal. Collision tests use strict box
  interiors plus the underground bound over sampled LPY interpolation. At most
  ten candidate attempts occur; the last rejected goal survives exhaustion,
  matching upstream. No fallback is inserted.
- Reset starts from the previous goal, even for `is_init=True`; the initial-start
  helper is unused by the active resampler. Reset does not immediately recompute
  current goal tensors. Orientation target adds base yaw at sampling time and
  is not refreshed on every step. These potentially surprising behaviors remain.
- Command deadband uses positive forward command OR absolute yaw command,
  not forward absolute value. Falling resets preserve commands; timeouts resample
  commands. Periodic command sampling uses episode length modulo the original
  three-second interval.
- Termination uses contacts over 1 N, goal-sign-dependent roll/pitch excursions
  over 0.2 radians, base height threshold, and episode length strictly greater
  than the maximum. The configured roll/pitch threshold values are unused by
  original code. `evaluation_config` reproduces relevant `play.py` overrides:
  height threshold becomes zero and randomization/pushing remain enabled;
  configured roll/pitch values of 1.0 still do not change the hard-coded formula.
- Signed Euler conversion, quaternion rotation, and LPY helpers are local explicit
  reconstructions. The upstream imported helper extension is missing from its
  source tree. LPY uses `z=-length*sin(pitch)`; world goals use yaw and base XY
  with fixed Z=0.53. Numeric parity of those missing helper implementations is
  unverified. Source AST parity does not establish their parity.

## Validation and remaining evidence

Run with the existing CPU environment (no installation required):

```sh
OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python -m pytest -q tests/test_deepwbc_task.py
```

Six tests pass. They compare vendored function ASTs with the actual pinned
source, execute original default reward functions against the same tensors,
and check dual-channel scaling, goal interpolation/collision/exhaustion,
strict timeout and evaluation threshold behavior, reset semantics,
randomization tensor dimensions, curriculum, and externally supplied-state
transition behavior. The first attempted test environment lacked pytest;
the already installed `pawcerto-mujoco` environment completed validation.

This does not establish physics signal parity, supported-API foot wrench
reconstruction parity, friction handling, end-to-end training, task learning,
sim-to-sim performance, or hardware readiness. A physical runtime must use the
original Go1/WidowX model, terrain and sensors, apply all reset/randomization
writes, and validate those differences before stronger evidence is claimed.
