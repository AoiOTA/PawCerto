# Bounded external ReLIC PLAY runtime

This optional consumer executes the externally supplied original Spot task and
TorchScript actor. It is separate from the default package dependencies and
contains only authored interoperability and recording code. The RAI Research
License source, meshes, URDF, and weights remain in the ignored external checkout.
See [external export provenance](relic-external-export.md) for license and model
identity. No engine build, installed Lab modification, optimizer, or replacement
robot/controller is involved.

## Original source and explicit migrations

Original revision: `27f8033c5064d32f049a17accb71cd1091422878`.
The immutable [upstream archive](https://codeload.github.com/rai-opensource/relic/tar.gz/27f8033c5064d32f049a17accb71cd1091422878)
has SHA-256 `a5da491db69602781f454b06dd2bf2eaefbb07b0ce685906eaedfd44e2a196f1`.
It is retained as `artifacts/relic-external-runtime/upstream.tar.gz`, with its
original license; the extracted checkout is `third_party/relic-reference`.
Original code and weights are not added to the wheel.

Actual CPU import exposed two namespace changes in current Lab. The external
`source/relic/relic/tasks/loco_manipulation/interlimb_env_cfg.py` has two marked
import migrations:

1. `AdditiveUniformNoiseCfg` → `UniformNoiseCfg`. The current default operation is
   additive; source PLAY already disables observation corruption.
2. `isaaclab_tasks.manager_based.locomotion.velocity.mdp` →
   `isaaclab_tasks.core.velocity.mdp`.

Actual single-environment simulation exposed a third source compatibility bug:
`mdp/terminations.py` used unrestricted `.squeeze()` on the filtered contact
matrix, deleting the batch axis at one environment. Its marked `.squeeze(-2)`
replacement preserves the batch and selected-body axes. For historical vectors
it removes the ground-filter axis; the current scalar-normal layout is
interpreted as described below. The norm, body maximum and threshold remain.
CPU checks exercise the actual imported function at batch sizes one and two.

`external-modifications.json` retains each exact failure and before/after file
hash. Every run records that provenance and a complete derived-source manifest.
These are external import migrations; they are not claims that current reward,
physics, or actuator implementations exactly match historical Lab 2.1.

The authored `compatible_spot_knee` class has MRO:
compatible class → original Spot knee → constructor bridge → current Lab
Remotized PD. Original initialization still builds its torque-speed tables and
interpolators; the compute method is the original function object. The bridge
removes the obsolete zero-valued armature/friction positional constructor slots
and forwards current actuator effort/velocity limits by name. Physical
armature/friction belong to the current articulation collection. It does not
alter the original position-dependent or speed-dependent torque clipping.

The public `cfg.sim.use_newton_actuators=False` selects the existing Python
actuator path because the custom knee is not a native Newton actuator. Official
Lab/PhysX remains unmodified. Current native PD behavior has not been compared
against historical Lab 2.1 physics.

## Actual run boundary

The entry point imports `relic.tasks`, instantiates original
`Isaac-Spot-Interlimb-Play-v0`, then sets one environment after PLAY's constructor
sets 128. It retains the source terrain, commands, command interpolation, arm
targets, selected manipulation-leg overwrites, rewards, and terminations. The
pretrained actor consumes actual source `(1,84)` observations and produces 12 raw
leg actions; the environment controls the complete 19-joint Spot with arm. PLAY
inherits Phase1 configuration, but the export's training stage/seed/update count
is unknown. A run cannot establish Phase1 learning or training reproduction.

The recorded experiment used 500 policy steps with zero optimizer updates. The
entrypoint accepts a positive step count; that experiment budget is not a
permanent API limit. The source
task resets normally on termination. Public `RecorderTerm` callbacks capture raw
actor values, processed leg/arm targets, manipulation selection/commands before
physics, and actual root/body/joint states, target buffers, applied torque,
reward, and termination terms after physics/reward computation but before
automatic reset. The callback returns `(None, None)` and disables dataset export;
it preserves the original manager behavior. Input observations are retained too.

From the repository root, the designated simulator/GPU operator runs:

```bash
PYTHONPATH=. .venvs/isaaclab-sim610/bin/python scripts/eval_relic_external.py \
  --external-root third_party/relic-reference \
  --policy third_party/relic-reference/source/relic/relic/assets/spot/pretrained/policy.pt \
  --external-provenance artifacts/relic-external-runtime \
  --output artifacts/relic-external-runtime/run-001 \
  --steps 500 --device cuda:0 --seed 0
```

The output directory must be new. It contains source/model/runtime identities,
resolved configuration, actual joint/body/action ordering, compressed trajectory,
and summary. Exceptions retain the original traceback and completed snapshots.
Termination count, total reward, and minimum pre-reset root height are bounded
diagnostic outputs, not success criteria for whole-body control or generalization.

## CPU evidence

The current official Lab checkout is
`412fb31b30ee605b4ffec4327436fc0fe53281d8`. On 2026-09-13, two CPU tests passed
against that installed Lab and the actual external source. They assert the
original constructor's incompatible keyword failure, successful bridge
construction, original compute identity, original tables/gains, and speed-limit
clipping against current Remotized PD at velocities -15, 0, and +15. They also
instantiate the actual PLAY config and recorder. This proves constructor and
sampled computation interoperability, not simulated performance.

The environment lacked pytest; the first failure is retained in `cpu-tests.log`.
Existing conda pytest was then used without installation:

```bash
PAWCERTO_RELIC_EXTERNAL_ROOT=third_party/relic-reference PYTHONPATH=. \
  .venvs/isaaclab-sim610/bin/python - <<'PY'
import sys
sys.path.append('/home/lyb/miniconda3/envs/pawcerto-mujoco/lib/python3.12/site-packages')
import pytest
raise SystemExit(pytest.main(['-q', 'tests/test_relic_external_runtime.py']))
PY
```

Result: 2 passed, 56 upstream deprecation warnings, retained in
`cpu-tests-existing-pytest.log`. Without an explicit external source environment
variable, these optional tests skip rather than fetch licensed source.

The first simulator attempt (`run-001`) exited 1 after environment setup but did
not preserve a traceback/result; its consumed step count is unknown. The failure
path now prints and flushes the traceback before Kit closes, serializes diagnostic
metadata safely, and retains observations with completed/partial records.

`run-002` exposed the singleton contact bug during the first `env.step`, after
physics advanced but before a complete recorder row. Its failure JSON, log, and
pending actor input are retained. It is not a zero-physics attempt. Four CPU
checks then passed after the shape-only fix, logged in
`cpu-tests-singleton-fix.log`. Each failure now records its phase, attempted
steps, and completed recorder rows. Failed attempts are separate from the
requested 500-step completion window; total physical steps across all attempts
cannot be asserted to be at most 500.

Current official PhysX warns that `force_matrix_w` returns the normal-force
matrix, not total contact force. The actual full sensor shape is `(1,25,1)`;
the source selects four upper-leg bodies, giving scalar forces `(1,4,1)`.
Here `.squeeze(-2)` leaves the four-body axis intact; the original last-axis norm
becomes absolute normal force, followed by the original body maximum/threshold.
For historical vectors `(N,4,1,3)`, the same change removes the filter axis.
Additional CPU tests cover both shapes at one and two environments. This is
valid current normal-force evaluation, with historical contact-signal equivalence
explicitly unverified.

## Completed actual runtime evidence

`run-003` completed 500 policy steps (10 seconds of aggregate simulated time,
including normal automatic resets) and exited 0. All trajectory fields were
finite. There were eight undesired-ground-contact terminations and no timeouts;
the minimum recorded pre-reset root height was approximately 0.299342 m. Total
source reward was -48.91573. These outcomes do not show a stable no-termination
trajectory or successful reproduction of reported ReLIC performance.

`run-003/trajectory.npz` contains all actual 84-value observations, raw 12-value
actor outputs, 19-joint target/state buffers, all 25 body states, rewards, and
termination flags. `summary.json` gives actual articulation, policy-action and
arm joint ordering, rather than assuming observation ordering from source regex.
The external model hash matched the pinned original export.

CPU reload auditing replayed all 500 actual observations through the original
TorchScript export: maximum CPU/GPU actor difference `5.7220458984375e-06`, within
`rtol=1e-5, atol=1e-5`. Reconstructing the source `raw * 0.2 + default` leg targets
and selected-leg overwrite matched the recorded processed targets exactly.
All 12 leg and seven arm targets matched actual target buffers exactly.
Manipulation was active for 404 steps; all four leg selections occurred.
`run-003/cpu-audit.json` and the external `audit_run003.py` retain these checks.

Recorded pre-reset episode lengths at termination were
115, 39, 88, 40, 61, 53, 49, and 47 steps. Thus the terminal states are retained
before resets, rather than averaged from fresh reset states. No further simulator
run or optimization followed this completed window. Final optional CPU tests:
six passed with upstream deprecation warnings, in `cpu-tests-observed-scalar.log`.

## Saved-data physical tracking

`run-003/tracking-audit.json` adds physical tracking statistics from the saved
trajectory only. The original run results and matched execution source snapshots
are unchanged. The independently authored external `tracking_run003.py` computes
post-physics joint/velocity errors against the processed targets and commands
applied before that same step. All eight pre-reset terminal rows are included.

Current official PhysX stores the recorded root quaternion in **XYZW** order.
`root_state_w` concatenates root-link world pose with root COM world linear and
angular velocities. The audit inverse-rotates those COM velocities by the
root-link quaternion to match current `root_lin_vel_b` / `root_ang_vel_b` semantics,
then compares body XY velocity and body angular Z against the source base command.
It records these semantics and the corresponding installed-source hashes; it does
not assume historical WXYZ ordering or treat COM velocity as root-link velocity.

| Physical tracking measure | Samples | Error |
| --- | ---: | ---: |
| Seven arm joints versus processed arm targets | 500 | 0.08834 rad component RMSE |
| Three selected-leg joints while manipulation is active | 404 | 0.30100 rad component RMSE |
| Body-frame base COM XY velocity versus base command | 500 | 0.90853 m/s vector RMS error |
| Body-frame angular Z versus commanded yaw rate | 500 | 0.89630 rad/s RMSE |

Joint component RMSE is `sqrt(mean(error²))` over all sampled joint components.
XY vector RMS is `sqrt(mean(error_x² + error_y²))`; the corresponding component
RMSE is 0.64243 m/s. These use actual measured states, rather than target-buffer
equality. Active selected-leg tracking by source leg selection is:

| Leg | Active samples | Three-joint component RMSE |
| --- | ---: | ---: |
| Front left | 154 | 0.36196 rad |
| Front right | 61 | 0.31461 rad |
| Hind left | 149 | 0.25244 rad |
| Hind right | 40 | 0.14944 rad |

Correct target writes therefore coexist with physical tracking errors and eight
terminations. This bounded run demonstrates executable source control and
measured tracking, without establishing stable whole-body performance, successful
learning, or historical physics equivalence. No new simulation or optimizer work
was used for this extension.
