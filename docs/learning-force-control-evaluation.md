# Learning Force Control exported-policy evaluation

This entrypoint evaluates the actual 19-output deterministic student from a full
PawCerto LFC training checkpoint on the original B1/Z1 task in official Isaac
Lab. It is a bounded integration evaluation, not the original paper's full
learning experiment. The [physics report](learning-force-control-physics.md)
records the source body lookup, native position-drive semantics and known
signal differences.

## Inputs and execution

Export a trusted full checkpoint with
`python scripts/export_learning_force_control.py --checkpoint MODEL --output EXPORT`.
The output retains the original `body_latest.jit` and
`adaptation_module_latest.jit` pair. `contract.json` records the checkpoint hash,
full configuration and counters; `comparison.json` checks both actions and
adaptation outputs after loading the exported modules. Its default seven
synthetic histories prove inference parity only. The separate raw upstream
`ac_weights_latest.pt` consumer still requires its matching supplied config.

Use the official Lab Python environment for:

```bash
python scripts/eval_learning_force_control.py \
  --checkpoint MODEL \
  --exported-policy EXPORT \
  --task configs/learning_force_control_evaluation.json \
  --robot-usd reference/isaac/b1_z1/usd_path.txt \
  --output NEW_OUTPUT_DIRECTORY --steps 500 --num-envs 16 --seed 0
```

Add `--preflight` for a CPU-only checkpoint/config/asset-identity check. It
does not initialize physics or qualify the evaluation. Output paths must be
new so previous failures and trajectories remain available.

The default fixed sampler requests forward velocity 0.3 m/s, zero lateral
velocity/yaw rate and EE spherical target `[0.6, 0, 0]` with a two-second
trajectory time. Binary force/position mode resampling, force commands,
spring disturbances, rough terrain, domain randomization, observation noise
and automatic source resets remain active. The source command deadbands and
interpolation still apply. This is a sampled execution window, not a fixed
complete-episode benchmark or a matched no-disturbance comparison.

The environment starts fresh with the source zero-action reset warmup and
cleared history. Checkpoint counters are recorded and restored, but saved
physical/task state is not used as the initial trajectory. LFC force schedules
depend on episode time and resampling, unlike UniFP's late global force stage.
Infeasible fixed EE goals below the source 0.05 m ground-clearance threshold
are rejected: resampling an identical infeasible fixed range cannot recover.

## Recorded evidence

Every step compares the CPU-exported JIT pair, moved to the runtime device,
against the checkpoint's student actions and adaptation prediction at
`atol=1e-6, rtol=1e-5`. Those JIT actions actually enter `env.step`. The source
environment then clips/delays/scales them, overwrites action 18 to `-0.1` and
sends position targets through native drives. Raw 19-output actions are not
joint torques.

`trajectory.npz` saves the actual 870-element policy histories, 16 privileged
training labels, raw actions, predicted latents, commands, rigid-body/joint
states, position targets, contact and externally applied forces, mode, freed
spring status, tracking errors, reward and termination reasons. Physical rows
are copied after reward/termination and before automatic reset. Failed and
nonfinite rows are retained; valid-environment statistics do not discard
other finite environments from the same step. Reset or inference failure
before physics cannot claim exported-policy consumption or zero parity error.

The EE position metric uses the original reward's negative-pitch spherical
transform, yaw-only rotation and fixed nominal base height. Actual source
Gym lookup `gripperStator=-1` reads `gripperMover`; the port names that body
explicitly. `link06` remains the force injection body. Position error is
reported for position-mode transitions. Force error compares the known
applied spring/external force, transformed into current base-yaw coordinates,
against the source force command. All force-mode and constrained force-mode
RMS values have separate counts; neither is independent contact-force
measurement. `source_pd_torques` is the source's computed PD tensor, not a
measurement of the native position-drive effort.

The summary includes actual observed JIT-driven steps, finite environment
transitions, mode sample counts, termination counts, locomotion error and
minimum base-up alignment. A finite 500-step process containing resets is
not a 500-step no-fall trajectory. Learning, contact quality and compliance
acceptance require their own experiment and cannot be inferred from export
parity or optimizer activity.

## Current verification

Four CPU tests exercise full-checkpoint export/reload, the source target
frame and fixed-goal feasibility, actual source curriculum initialization and
sampling, and finite-mode/failure accounting. They passed; the exported actions
and latents exactly matched the CPU checkpoint on the synthetic inputs.
Independent read-only review found the infeasible fixed-goal resampling issue
above; the input validation and regression case address it.

The real model 21 is frozen at SHA-256
`5f1e09891536758039cdb3c6839324380f104b8a4793b2a9c3fcd613314876e2`,
with 16,128 training samples and source policy counter 1,009. Its actual export
and CPU preflight passed. The first physical evaluation then failed during
reset with `ValueError: probabilities contain NaN`, before consuming any JIT
action. The evaluator had written nonexistent `limit_lin_vel_*` keys while
the source curriculum reads `limit_vel_x/y/yaw`; its remaining zero-velocity
grid center lay outside the requested fixed 0.3 m/s range, leaving zero
sampling weights. Correcting that explicit field mapping and exercising the
actual source initializer/sampler on CPU resolved the cause. The source
curriculum and trained checkpoint were unchanged. The failed output remains
at `outputs/learning-force-control-execution-20260913/model21-eval500`;
the missing 500-step evaluation uses a new attempt directory.

The corrected attempt exited 0 at
`outputs/learning-force-control-execution-20260913/model21-eval500-attempt2`.
All **500 policy steps × 16 environments = 8,000 transitions** were finite.
Every action actually entering physics came from the exported JIT pair;
device-side checkpoint/JIT action and latent maximum errors were both zero.
There were **four body-orientation terminations with automatic resets**.
This is not a no-fall rollout or trained force-control result.

| Recorded quantity | Result |
|---|---:|
| Position-mode transitions | 4,555 |
| Force-mode transitions | 3,445 |
| Constrained force-mode transitions | 2,445 |
| Position-mode mean EE position error | 0.821928 m |
| Force-mode componentwise XYZ force-error RMS | 40.852856 N |
| Constrained force-mode componentwise XYZ force-error RMS | 48.492836 N |
| Base XY velocity-error component RMS | 0.247447 m/s |
| Minimum base-up dot product | 0.282486 |

The force numbers use known applied force, including source ramp/free-spring
behavior, and must not be presented as physical wrist/contact-sensor accuracy.
The poor tracking and four resets remain part of the result.

An independent CPU process consumed all 8,000 actual saved histories using
the same JIT files. CPU checkpoint/JIT actions and latents matched exactly;
maximum differences from saved GPU actions and latents were respectively
`3.278255e-7` and `1.335144e-5`, passing the stated combined absolute/relative
tolerance. No physics was rerun. The checker and its result are
`outputs/learning-force-control-execution-20260913/compare_actual_inputs_cpu.py`
and `model21-actual-input-cpu-comparison.json` in the same directory.

Frozen trajectory SHA-256:
`824178f260e19177125a5e78931d797a5b72ab89765afb82d472ddbc287a270f`.
The initial seven synthetic-input export comparison remains separate from
this actual-input consumption record.
