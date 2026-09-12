# DeepWBC fixed-policy evaluation and export

## Status and boundary

The corrected-terrain run completed the actual 21-update checkpoint export and
500-step fixed Isaac Lab evaluation. All 500 observations have physical contact
and lie inside the terrain; fused/split exports and recorded actions match the
checkpoint exactly. EE RMSE is .47936 m, forward-velocity MAE .49878 m/s, and
one episode terminates for signed roll. The final 455-sample episode is
unfinished, so it is not a completed success. These are bounded physical
execution and policy-consumption results, not demonstrated task learning.
The earlier zero-contact evaluation was caused by incorrect terrain placement
and remains preserved as an invalid contact-task run. See
[the reconstructed-runtime report](experiments/deepwbc-reconstructed-runtime-20260912.md).

## Source semantics

The fixed reference is `third_party/deepwbc-reference`, revision
`8159e4ed8695b2d3f62a40d2ab8d88205ac5021a`:

- `legged_gym/legged_gym/scripts/play.py`: curriculum update, reset, deterministic
  `act_inference(..., hist_encoding=True)`, unless stochastic is requested.
  This evaluator intentionally supports the deterministic fixed-policy path.
- `legged_gym/legged_gym/scripts/save_jit.py`: separately consumes history encoder
  and actor(proprioception + latent). Both this split consumer and a fused
  full-observation consumer are exported here.
- `widowGo1.py` task equations and the local task/observer/controller remain the
  owners of observations, history, actions, reward, termination and reset.

Input is float32 `[N,860]`: current proprioception 76, privileged slots 24,
oldest-to-newest history `10×76`. History inference ignores privileged slots.
The observer returns the **old** history before updating it; reset history is
zero, then filled with repeated proprioception for episode length ≤1. Evaluation
never calls the observer a second time. It uses the observation returned by
`reset`, avoiding the original play script's stale pre-reset observation variable.
Reset itself executes one zero-action policy step; this warmup is not in the
recorded step budget. Its history/action-delay effects are retained.

Raw output has 18 policy-order actions: FR, FL, RR, RL leg joints, then the six
arm joints. The existing controller owns reordering, clipping, two-step delay,
action scaling, motor-strength multiplication and PD, at four 5 ms substeps per
20 ms policy step. Two finger efforts stay zero. Only 15 public target scales
are nonzero; this is not a new full-6D task or a 15-DOF replacement robot.

Leg and arm rewards stay separate, including source `/100` scaling. Diagnostics
compute geometry directly, never invoke reward methods with accumulation side
effects. EE target world coordinates use base XY, **fixed Z=.53**, and base yaw,
not full base roll/pitch/height.

Termination is exactly the source task: configured contact-body force norm >1,
goal-sign-dependent roll/pitch ±.2 rad, base Z below threshold, or episode length
**strictly greater** than its maximum. The play script's r/p threshold fields
set to 1.0 are unused by these hardcoded signed predicates. They do not relax
actual roll/pitch termination. Multiple reasons can coexist. Timeout alone is
not counted as a physical failure; partial budget-ending episodes are not
counted as completed or successful.

## Fixed task and evaluation profile

Create a task JSON outside training directories, for example `/tmp/deepwbc-task.json`:

```json
{
  "commands": [0.5, 0.0, 0.0],
  "ee_goal_lpy": [0.6, -0.3, 0.3],
  "traj_time_s": 2.0,
  "hold_time_s": 1.0
}
```

These four fields are required. Commands are `[vx, vy, yaw_rate]` in m/s, m/s,
rad/s; source `vy` must be zero. LPY is length in metres, pitch/yaw in radians;
positive pitch points downward. Initial/final sampler ranges collapse to the
specified values, but source interpolation, resampling, reset and rejection
logic remain. An impossible fixed initial path or endpoint fails before launch.
The source command deadband remains: if `vx <= .3` and `abs(yaw_rate) <= .6`,
**all commands become zero**. Metadata stores the requested task; trajectories
store the actual commands. Goals initially interpolate from zero LPY, not an
instantaneous target; subsequent identical endpoints hold the same target.

`--profile original-play` (default) applies the existing play overrides, retains
randomization/pushes and advances curriculum once to the final scales. It is
**not an exact replay of the play script**: checkpoint terrain geometry and
placement are retained rather than applying play's 600×600 resize. No flat plane
is substituted. `--profile checkpoint` preserves checkpoint randomization,
thresholds and schedules; it still fixes task ranges and advances curriculum
once. Both profiles record the complete resolved configuration. A fixed task
and seed do not eliminate randomized physics or guarantee cross-device bitwise
reproducibility. No observer, reward or control modification is made to suppress
that variation.

## CPU preflight and export

Only trusted checkpoints should be loaded (runner files use Python pickle).
PawCerto full checkpoints carry their config; upstream checkpoints without it
require `--config /absolute/resolved-config.json`. Explicit config conflicts and
missing model keys fail rather than guessing or partially loading. No optimizer
is constructed, resumed or updated; no latest-checkpoint search is performed.
All output directories must be new, so existing results are never overwritten.

```bash
CPU_PY=/home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python
"$CPU_PY" scripts/export_deepwbc.py \
  --checkpoint /absolute/model_40000.pt --output /absolute/eval-export

"$CPU_PY" scripts/eval_deepwbc.py \
  --checkpoint /absolute/model_40000.pt --task /tmp/deepwbc-task.json \
  --robot-usd /absolute/prepared/merged/merged.usda \
  --force-mode reconstructed_sensor_wrench --num-envs 1 --steps 500 \
  --exported-policy /absolute/eval-export/policy.pt \
  --output /absolute/eval-preflight --preflight
```

The signal defaults to the checkpoint's recorded `force_mode`. For a checkpoint
without signal metadata, select it explicitly. `reconstructed_sensor_wrench`
uses the approximate PRE six-axis reconstruction tested in the
[signal experiment](experiments/deepwbc-solver-wrench-20260912.md); contact-scene
Fz² input MAE was about 10–13% of the original Gym mean. Native joint reaction
remains available for explicit diagnostic comparison. An evaluation override
is recorded in its resolved config and must not be interpreted as evaluating
the checkpoint under its original signal semantics.

Preflight checks checkpoint/config/task, USD existence and identity, the adjacent
`source-manifest.json` and `merged.urdf` consumed by the real runtime, limits and
link connectivity. A runtime `.txt` USD pointer is also supported. Checkpoint identity hashes the
resolved physical USD, so pointer aliases of the same asset are accepted while
a changed physical USD remains rejected. It does not
compose/validate all USD payloads, instantiate sensors, or prove robot conversion
or physics readiness. It writes `preflight.json` with `physics_run=false` and
`evaluation_success=false`. It never imports the simulator launcher. Optional
exported-policy preflight comparison uses zero observations only; subsequent
real evaluation compares every actual policy input.

Export writes `policy.pt`, `actor.pt`, `history_encoder.pt`, `contract.json`,
`comparison.json`. Contract contains checkpoint hash/config, dimensions,
observation layout, history timing and joint/action order. Default parity inputs
are explicitly synthetic, not physical observations. Export can instead consume
a saved trajectory chunk:

```bash
"$CPU_PY" scripts/export_deepwbc.py \
  --checkpoint /absolute/model_40000.pt \
  --observations /absolute/evaluation/trajectory_00000.npz \
  --output /absolute/eval-export-on-recorded-inputs
```

### Independent consumption (only Torch required)

```python
import torch
policy = torch.jit.load('/absolute/eval-export/policy.pt', map_location='cpu').eval()
with torch.inference_mode():
    actions = policy(observations.float())  # [N,860] -> raw [N,18]

# Equivalent original-author-style split path:
history = torch.jit.load('/absolute/eval-export/history_encoder.pt', map_location='cpu').eval()
actor = torch.jit.load('/absolute/eval-export/actor.pt', map_location='cpu').eval()
with torch.inference_mode():
    actions_split = actor(torch.cat((observations[:, :76], history(observations[:, 100:])), 1))
```

Here `observations` must come from the unchanged DeepWBC observer, including
scaling, clipping and caller-managed history; neither artifact constructs raw
sensor observations or advances history. `compare_export` reloads the modules
and checks fused/split actions against the checkpoint's history-conditioned mean
at `atol=1e-6, rtol=1e-5`. It fails on mismatch, not just logging a claim.

## Real Isaac Lab evaluation command

The device owner runs the converted robot in the installed official Lab
environment by omitting `--preflight` and selecting a new output:

```bash
/home/lyb/miniconda3/envs/pawcerto-lab-sim610/bin/python scripts/eval_deepwbc.py \
  --checkpoint /absolute/model_40000.pt --task /tmp/deepwbc-task.json \
  --robot-usd /absolute/prepared/merged/merged.usda \
  --force-mode reconstructed_sensor_wrench --num-envs 1 --steps 500 --seed 0 \
  --exported-policy /absolute/eval-export/policy.pt \
  --device cuda:0 --headless --output /absolute/deepwbc-evaluation
```

The installed launcher uses `visualizer='none'` for `--headless`, `'kit'` for
`--no-headless`. Startup/loop exceptions are recorded and printed before closing
with a nonzero exit status, including interruption. No training runner, PPO or
optimizer is created. Asset/terrain/runtime failures are propagated, not replaced
with a mock, alternate robot, plane, fallback sensor or physics implementation.

## Recorded evidence

| Artifact/fields | Meaning |
|---|---|
| `metadata.json` | Checkpoint SHA/iteration, task/seed/config, USD/manifest/URDF identities, evidence source and parity limits |
| `trajectory_00000.npz`, … | Up to 256 steps per chunk; leading axes `[step,env,...]`; lossless arrays, including NaN/Inf on failure |
| `observations`, `policy_actions` | Exact **pre-action** input and raw policy output |
| `root_states`, `dof_pos/vel`, `torques` | **Post-physics, pre-push/pre-reset** state; root XYZ/xyzw/world linear/angular velocity; 20 source-ordered joints |
| `actions`, `latest_sim_actions` | Delayed applied and latest undelayed simulator-order actions |
| `ee_pos/orn`, optional `ee_lin_vel/ang_vel` | Measured EE world-frame state, xyzw quaternion; velocities present on the real runtime |
| `curr_ee_goal_sphere`, `ee_goal_sphere`, `ee_target_world` | Current interpolated and sampled endpoint goals, world target |
| `ee_error_l2_m/l1_m`, `ee_error_sphere_weighted` | Cartesian errors and source weighted absolute LPY error (no extra wrapping) |
| `base_lin_vel/ang_vel`, `velocity_error`, `yaw_rate_error` | Body-frame motion, signed actual-minus-command errors |
| `reward_leg/arm`, `done`, `reason_mask`, `finite` | Unmodified reward/done and reason columns: contact, signed roll, signed pitch, height, timeout; finite is diagnostic only |
| `contact_forces`, `foot_wrenches` | Actual supplied contact and six-axis FL/FR/RL/RR wrench signals, not reconstructed success labels |
| `episode_id`, `episode_length_buf`, `step`, optional `time` | Per-env episode IDs, source length, global recorded-step index, runtime physical time (includes warmup) |
| `push_applied`, `push_velocity_xy` | Requested source push, applied after this snapshot |
| Optional `exported_actions`, `export_action_max_abs_error` | Independent exported-policy output and discrepancy on each identical input; original policy still controls physics |
| `episodes.jsonl` | Completed-episode or nonfinite-event causes, sampled leg/arm returns and EE RMSE; multiple reasons retained |
| `summary.json` | Completed/failed execution status, traceback cause text, counts, aggregate EE RMSE, velocity/yaw MAE, unfinished episode sample counts |

Episode returns/metrics include recorded steps only (not initial warmup). A finite
horizon cut is explicitly partial. A nonfinite action stops before its physics
step; a nonfinite transition is saved, then execution stops without changing
source termination rules. Summary flushes buffered chunks on ordinary exceptions
and KeyboardInterrupt. Abrupt process kill can lose the current unflushed chunk;
this is not a transactional recorder. `status=completed` means the requested
step budget finished, **not** that tracking, learning or the task succeeded.

## CPU validation and observed issues

Final run: **46 passed, 4 skipped, 31 warnings in 5.75 s** in the specified CPU
Python. The skip reasons are listed below; skipped USD checks are not passes.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python -m pytest -q \
  tests/test_deepwbc_algorithm.py tests/test_deepwbc_env.py \
  tests/test_deepwbc_policy.py tests/test_deepwbc_runtime.py \
  tests/test_deepwbc_task.py tests/test_deepwbc_terrain.py \
  tests/test_deepwbc_evaluation.py tests/test_deepwbc_export.py
```

Coverage includes source-network weight/action parity, fixed commands/interpolation,
source deadband, terminal-before-reset capture, distinct timeout/height/signed-roll
causes, unchanged reward accumulators/history, numerical failures, chunk reload,
export comparisons on runtime-double observations, isolated Torch-only subprocess
consumption, CPU CLI import guards and runtime-signature/launcher/exit-code checks.
CPU fixtures never integrate robot motion and their random weights are not a
trained checkpoint. Runtime-double trajectory values are not physical data.

Observed and addressed: terminal state loss through auto-reset (added a read-only
pre-reset callback); asset preflight omitted required manifest/URDF (now rejected
before launch); installed launcher/close API differs from legacy headless usage
(now uses visualizer and preserves failure status). TorchScript deprecation
warnings remain visible; the currently installed Torch still executes the chosen
standalone, dependency-free consumer. Four standalone-USD tests skip in the
specified CPU Python because its `pxr` bindings are absent; no packages were added.

Real evaluation still needs a specified trained checkpoint, a fully validated
original-robot USD including payloads/body/joint/contact/mimic behavior, a free
and authorized device run with the official runtime, and saved real transitions.
The native incoming joint wrench (even after correct frame transform) is not yet
proven equivalent to original Gym's net-body six-axis force sensor. Reconstructed
LPY helpers and actual randomized material/inertia behavior also retain their
existing parity limitations. These must be measured, not inferred from tensor
shape, CPU tests or successful export. No unmodified-engine dependency constraint
is relaxed, and no physical-success threshold has been invented.
