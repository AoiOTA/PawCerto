# DeepWBC Isaac Lab training entry

`DeepWBCIsaacTrainingEnv` connects the original task, observer and dual-reward
runner to `Go1WidowXIsaac`. `scripts/train_deepwbc.py` requires an explicit
iteration budget and a converted original Go1/WidowX robot USD. It starts
Isaac Lab only inside `main`, after command-line and input validation.
For the installed Lab 6.1 launcher, `--headless` maps to `visualizer='none'`
and `--no-headless` maps to `'kit'`. Runtime exceptions print their traceback
before shutdown; `close(exit_code=1)` preserves failure when Kit exits directly.

A GPU owner can run the following with the configured official Isaac Lab
Python environment after robot conversion and runtime verification:

```sh
python scripts/train_deepwbc.py \
  --robot-usd /absolute/path/to/go1_widowx.usd \
  --force-mode reconstructed_sensor_wrench \
  --schedule public_fresh --num-envs 256 --iterations 100 \
  --output /absolute/path/to/deepwbc-run --headless --device cuda:0
```

The numbers above illustrate an explicit budget, not a claim that 100 iterations
establish learning. A finite budget alone does not establish learning.
`--schedule paper` selects the existing paper-budget/regularization override;
it retains the public position-only task and is not full paper 6D reproduction.
`--config` instead accepts a complete resolved JSON configuration. Runtime
metadata records the chosen force signal and robot USD SHA-256. Output contains
resolved config, per-iteration metrics, runner checkpoints and a completion
artifact whose finite-update status does not assert learning success.
Metrics are flushed after every completed iteration. Checkpoints are written
at the starting iteration, configured `runner.save_interval` boundaries, and
the final completed budget iteration; a final periodic boundary is written
only once. The CLI defers intermediate end-of-call saves through
`runner.learn(1, save_final=False)`. Direct `learn(...)` calls retain final-save
behavior by default. A failed or interrupted iteration propagates its original
exception, retains already flushed metrics/checkpoints, and does not save an
incomplete update as a final checkpoint.

## Actual sequence and preserved semantics

- The runtime owns action mapping, clipping, the two-policy-step delay, and
  explicit PD at each of four 5 ms physics substeps. The env pushes the policy
  action once per policy step and calls `step_control` four times.
- Initial reset performs one complete zero-action policy step, as the original
  `BaseTask.reset` does. Getter calls use the cached observation and do not
  advance observation history.
- After physics the task advances goals, periodic commands, pushes, termination
  and separate leg/arm rewards. The env writes a requested velocity push,
  resets done rows and returns `(obs, None, leg_reward, arm_reward, done, extras)`.
  Original terminal transitions and timeout masks survive automatic reset.
  The original 24 privileged values remain embedded in the 860-column input;
  there is no separate privileged observation tensor.
- Reset writes root and DOF state. Derived local velocities, EE and contact
  caches retain terminal values until the next post-physics refresh, matching
  the source ordering. Action-history slots clear, while the already applied
  delayed action remains available to source `last_actions` bookkeeping.
- Origins sample the source bounds on the shared terrain: default X in
  `[-3.75,-3]`, Y in `[-115,115]`, plus reset perturbations. Runtime clone-grid
  offsets are disabled. Box lateral offsets have both signs. Box episodic reset
  changes XY only, preserving its physical Z, orientation and velocity.
- The runner updates curriculum once per learning iteration. History adaptation
  occurs at zero-based iterations 0, 20, 40, etc.; remaining iterations use PPO.
  The env forwards both reward channels without combining or rescaling them.

## Force signal and physics evidence boundary

Fresh training requires an explicit signal selection. `reconstructed_sensor_wrench`
uses all 27 bodies and 20 DOFs to reconstruct free COM acceleration from PRE
public M/C/G/J, native actuation and body kinematics. After the physics step,
`StockSolverForce.approximate_sensor_wrench` combines the full six-axis solver
residual with gravity and expresses force and moment at the PRE foot origin.
The task receives the fourth 5 ms sample for each 20 ms policy step. Creation
randomization invalidates the model's cached mass, inertia and COM parameters;
the next PRE step reads their actual native values. Unsupported external
wrenches or physics properties fail explicitly.

This is an approximate signal: three short original-Gym/Lab scenes matched
contact bits, while contact-scene Fz² MAE was about 10–13% of the original mean.
The full angular channels are reconstructed, not filled with zero. See
[the measured comparison](experiments/deepwbc-solver-wrench-20260912.md) for
amplitude errors, PRE timing and qualification boundaries. This runtime mode
preserves that approximation explicitly; it does not claim exact sensor parity.

`native_joint_wrench` remains available as a diagnostic mode using native incoming
joint reaction transformed to the foot origin. It differs from the original
Gym force sensor. The runtime constructor default remains native, and training
records the explicit selected mode and evidence statement in its configuration.

The terrain uses the dedicated original Perlin/slopes builder and physical
collision mesh. A flat-plane substitute is not selected. Terrain curriculum and
height-measurement variants fail explicitly because they require additional
physical task logic; both are disabled in the public original recipe.

## Resume

```sh
python scripts/train_deepwbc.py --resume /absolute/run/model_100.pt \
  --iterations 100 --output /absolute/run-continuation --headless --device cuda:0
```

Resume uses the checkpoint's complete config, resolved physical robot-USD identity and force mode.
A `.txt` pointer alias of that same USD is accepted; changing its physical target is rejected.
Configuration/schedule/environment-count overrides are rejected. It restores
runner model, optimizer, history optimizer and iteration/counter state. Task
curriculum resumes at the runner iteration, and new physical episodes start.
This is continuation from fresh physical episodes, not exact physical replay;
the runner checkpoint does not contain simulator, task, or all random-generator
states. Only load trusted checkpoints.

## CPU validation

```sh
OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python -m pytest -q \
  tests/test_deepwbc_env.py tests/test_deepwbc_task.py tests/test_deepwbc_algorithm.py
```

The combined env/task/algorithm validation passes 20 tests (four upstream Torch
JIT deprecation warnings). The env tests use an explicitly nonphysical interface double to verify reset
and call ordering, policy mapping/delay, six-field return, dual rewards,
automatic reset/timeouts, push timing, curriculum and CLI configuration.
Actual CPU runner updates through the CLI loop verify periodic and final
checkpoint files, one JSON metric per iteration, and preservation of injected
runtime errors and keyboard interruption. Their state source remains the
nonphysical interface double.
Launcher interface tests also reproduce a close operation that exits directly,
checking that runtime errors and interruption retain traceback and nonzero code.
Task tests compare executable source equations; algorithm tests independently
cover original PPO/DAgger and checkpoint behavior. These tests cannot establish
physics compatibility, learning success, sim-to-sim transfer or hardware safety.
The designated GPU operator completed a fresh corrected-terrain run with 21
actual updates (16 envs, 40 rollout steps each, seed 1): two history updates and
19 PPO updates, 13,440 transitions, all losses and parameters finite. Its
fixed 500-step model-21 evaluation has physical contact on all samples and
exact exported-policy parity. EE RMSE .47936 m, forward-velocity MAE .49878 m/s
and one signed-roll failure do not demonstrate learned task success. The
455-sample final episode is unfinished. The earlier zero-contact execution was
caused by the terrain pose and is retained as invalid physical-task evidence.
See the [actual run and repair record](experiments/deepwbc-reconstructed-runtime-20260912.md).
