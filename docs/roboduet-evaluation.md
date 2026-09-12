# RoboDuet fixed physical evaluation

`scripts/eval_roboduet.py` consumes all five exported policy modules through the
existing RoboDuet observer, planner, mixed controller, task, and official Isaac
Lab/PhysX runtime. No second controller or physics implementation is introduced.
The CPU tests verify this integration with prescribed states; actual physical
results require running the command below on the simulation device.

The protocol was frozen before learned results in
`outputs/roboduet-fixed-evaluation-20260912/protocol.json`; the identical portable
copy is [`configs/roboduet_evaluation.json`](../configs/roboduet_evaluation.json).
It is a new engineering protocol, not an official RoboDuet benchmark. It sets
no numerical pass threshold and must not be retuned against endpoint results.

## Fixed cases and timing

Use seeds 2027, 2028, 2029, each with three parallel cases for 20 seconds at
50 Hz. The source reset's seeded joint-position scale, random yaw, and root
velocity remain; dynamics are nominal plane, with mass/motor/gravity/material
randomization disabled. Friction is 1, restitution 0, gravity is the existing
environment's nominal -9.8 m/s². Observation noise is disabled in configuration
(the source observation methods already do not add noise).

Each case cycles through four targets with original `T_traj = 2.5 s` and integer
arm-clock resampling. The targets remain inside the original supported ranges:

| Case | Leg command | Arm command |
| --- | --- | --- |
| standing_ee_motion | vx, vy, yaw rate all zero | Four LPY/RPY targets |
| locomotion_fixed_ee | Forward, sideways, turning, backward turning | Fixed LPY (0.5, 0, 0), zero RPY |
| simultaneous_commands | Same varying leg sequence | Same varying arm sequence |

The complete numerical sequence is in the JSON. A fixed EE goal means fixed
in RoboDuet's yaw-aligned, terrain-referenced command coordinates; it is not a
stationary world-space target during locomotion. LPY uses the source 0.38 m
height reference and the 0.1 m grasper offset. RPY constructs Rz Ry Rx, then
uses the original projected-axis alpha/beta/gamma encoding for observations.
There is no invented interpolation between the command endpoints.

All checkpoints are played with the arm physically released (Stage 2) and
`official_play`: raw arm body guidance, then `plan`, then dog observation and
action, then four physical substeps. The extra arm observation getter before
the playback loop matches upstream `play_by_key.py`. The original zero-action
reset step is recorded separately. An initialization termination ends the case
as an invalid prefix. The subsequent requested rollout is 1000 control steps;
its last step can also carry the original episode timeout because reset already
advanced the episode clock once.

Compare full checkpoints at next_iteration 0, 10001, and 50000. At 10001 the
arm has received zero PPO updates, even though Stage 2 is next. An optional
`training_mean` package is a separate graph comparison, not accepted by this
main protocol.

## Run and artifacts

Use the [prepared official Lab environment](isaaclab-install.md) and converted
Go1/ARX5 asset. From the repository root, set `PAWCERTO_PYTHON` to that
environment's Python executable and select the complete checkpoint to evaluate.
The recorded workstation runs used an independent user service with an 8 GiB
memory budget and 1 GiB swap limit; those are local execution settings, not
method parameters. A portable direct invocation is:

```bash
export PAWCERTO_PYTHON=/path/to/lab-environment/bin/python
OMNI_KIT_ACCEPT_EULA=yes PYTHONNOUSERSITE=1 \
  "$PAWCERTO_PYTHON" scripts/eval_roboduet.py \
  --checkpoint outputs/RUN/model_50000.pt \
  --output outputs/roboduet-fixed-evaluation-20260912/model_50000 \
  --visualizer none --device cuda:0
```

Repeat for checkpoints 0 and 10001 using distinct output
directories without changing the protocol. `--export`
can consume an existing complete `official_play` directory. `--usd` can select
the prepared Go1/ARX5 asset; the default first uses the checkpoint's
`pawcerto_asset.usd_path` and verifies its saved SHA256, then uses the runtime's
asset-path record for older checkpoints. Explicit overrides are recorded. An existing output directory is refused to preserve previous evidence.

Each result contains the exact protocol and evaluation config, export identity
and checkpoint SHA256, per-seed `summary.json`, `states.pt`, `substeps.pt`, and
`policy_inputs_outputs.pt`. The 5 ms substep artifact retains pre/post q/qd,
controller joint targets, clipped actions, the exact `combined20` effort command
sent to the runtime, root pose/velocity/height/up-dot, body positions, contact
forces, and issued targets. Each case stops recording after its first native
control-step termination, including all four substeps of that terminal interval.
The reset initialization is retained too. The sent effort is not a measured
solver reaction and excludes the runtime implicit spring/damper contribution. States retain the initial sample and each case's
first physical prefix through termination, with root pose/velocity, joints,
EE pose/velocity, contacts, issued commands and errors; quaternions are xyzw.
These states can be consumed by a later CPU video renderer. Policy inputs and
outputs retain the first and last executed batched inference for real-data
export verification; they are not synthetic inputs generated by the exporter.

## Measurements and evidence limits

The optional `pre_reset_callback` in the existing environment reads physical
state after reward and termination computation but before automatic reset.
Callback exceptions propagate. Errors use the targets issued before that step,
so a command resampled during `advance()` cannot retroactively change the
preceding control interval's target.

Each case reports planar base-frame linear-velocity norm error (m/s), absolute
base-frame yaw-rate error (rad/s), grasper Cartesian position error (m), and
quaternion geodesic orientation error (rad). Orientation uses the commanded
orientation relative to base yaw; stored ABG values remain available for
source-specific diagnostics. Error means and maxima cover only the retained
prefix and are shown alongside non-timeout termination, timeout, survival,
finite state, invalid prefix, and full-duration completion. A simultaneous
physical termination and timeout still counts as physical failure.

The model 1600 comparison is an additional early diagnostic with the same
released-arm Stage 2 protocol. Its differences cannot be attributed solely to
Stage 1 learning; the 0/10001/50000 main endpoints remain required.

Additional measurements are minimum world-up dot product, policy action clip
fraction, and commanded leg-effort limit fraction. Arm mixed-controller entries
are position targets and are not mislabeled as torques. Available Head contact
forces are recorded at the last physical substep; these are not peak impact
forces across all four substeps. Absence of that sensor produces null summary
values rather than fabricated zeros.

Reward increase, a finite optimization step, or fewer terminations alone does
not establish learned coordination. Interpret tracking and survival jointly,
and retain early failures rather than comparing them as complete trajectories.

CPU validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_roboduet_evaluation.py tests/test_roboduet_training.py \
  tests/test_roboduet_export.py -q
```

The CPU command needs PyTorch and pytest, but does not launch Isaac Lab. The
optional saved-state renderer additionally uses the `analysis` extra and
`ffmpeg`; it reads recorded physics and does not simulate the robot again.
