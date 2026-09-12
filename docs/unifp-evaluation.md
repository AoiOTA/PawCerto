# UniFP export and fixed-task execution

This route consumes the original B2/Z1 UniFP student policy. It retains the
method's 2336-value actor history, concurrent 12-value estimator, 17 actions,
19-joint controller and original task equations. The initial execution budget
is a bounded integration check, not a trained force-control result.

After the [official Isaac Lab environment](isaaclab-install.md) is installed,
install the method's original terrain converter with
`python -m pip install -e '.[unifp]'` from the source checkout. The tested extra uses pydelatin 0.3.0;
it does not replace or rebuild Isaac Lab or PhysX.

## Checkpoint and export

The Lab runner records its full resolved `config` (`env` and `train`), actual
`global_steps`, completed iteration count, model and both optimizers. A trusted
older checkpoint without configuration requires explicit `--config`; physical
evaluation also requires its saved policy-step count to preserve force timing.
Iteration count is not substituted for the actual policy-step counter.

```bash
python scripts/export_unifp.py \
  --checkpoint /path/to/model_21.pt --output /new/path/to/unifp-export
```

`policy.pt` consumes a nonempty float32 `[batch,2336]` history and returns the
raw `[batch,17]` deterministic student mean. `estimator.pt` consumes the same
history and returns the scaled `[batch,12]` supervised estimates. The actor
uses its 64-dimensional encoded latent; the decoded estimates do not directly
enter it. Neither module owns history initialization, observation scaling or
PD control. Use the UniFP environment's observation dictionary with
`ExportedUniFP.act_inference`, and zero the original history buffers on reset.

Export independently reloads both modules and compares their outputs with the
checkpoint model. By default its inputs are seeded synthetic histories,
including zeros. To compare actual evaluation inputs, pass
`--observations /path/to/evaluation/trajectory.npz`. The comparison file names
the input source. Numerical parity alone does not establish physical behavior.

## Fixed task

After the official Lab asset conversion and training entrypoints have run:

```bash
python scripts/eval_unifp.py \
  --checkpoint /path/to/model_21.pt \
  --robot-usd reference/isaac/b2_z1/usd_path.txt \
  --task configs/unifp_evaluation.json --steps 500 --num-envs 1 --seed 0 \
  --exported-policy /path/to/unifp-export --output /new/path/to/evaluation
```

Run this command in the supported Isaac Lab environment. `--preflight` checks
CPU inputs and configuration without opening a simulator, using a separate
new output directory. It is not physical validation.

The fixed task sets the velocity sampler to `[0.3,0,0]` m/s, m/s, rad/s and the
spherical EE goal sampler to `[0.66,0,0]` m, rad, rad. Goal interpolation uses
2 s travel and 1 s hold. It disables the random zero-velocity-command choice;
the original deadbands remain. Initial goal interpolation, terrain, noise,
physical randomization, pushes, force schedules, rewards and automatic resets
remain those of the saved configuration. These explicit evaluation overrides
are recorded alongside the complete resulting configuration.

The saved `global_steps` is set before the original reset warmup, which performs
one zero-action policy step and increments the counter. A 21-update checkpoint
does not reach the strict force-stage boundary `global_steps > 8000 * 24`.
Any separate check that places the counter near this boundary is a stage
execution probe, not evidence of having trained through 8000 updates.

When an export is supplied, the actual simulation uses its actions. Both action
and estimator outputs are also compared with the checkpoint model on those
same real observations. The export's checkpoint hash must match the selected
checkpoint; no latest-checkpoint selection is implicit.

## Records and interpretation

`metadata.json` records the command, checkpoint and asset identities, seed,
configuration, fixed task and overrides. `trajectory.npz` preserves each policy
input, estimate and training supervision, raw actions, joint/root states,
applied torques, commands, goals, contact forces, known injected forces and
pre-reset termination reasons. The original gripper force buffer is the
training label. Contact force is a separate signal; neither is mislabeled as
an external hardware force measurement.

The state snapshot occurs after physics, reward and termination calculation,
before the original reset. It records both geometric EE position error and
the source reward's compliance-offset target error. Roll, pitch, contact and
timeout reason masks must reproduce the original termination decision.

`summary.json` distinguishes completed execution from exceptions and retains
any saved invalid prefix. Its means cover finite recorded environment transitions,
including terminal endpoints; they are not complete-episode means. It also
reports resets, minimum base-up dot product and base-velocity tracking. A
completed 500-step execution with automatic resets is not a 500-step no-fall
trajectory, learned force control, sim-to-sim transfer or hardware acceptance.
One invalid environment does not exclude other finite environments at the same
step. Export consumption requires an observed post-physics transition; before
any completed comparison, export-error fields remain null.

Current evidence: the 12 UniFP CPU tests passed on 2026-09-13, including the
full-size original network, optimizer/equation comparisons and standalone
TorchScript action/estimator reload for several batch sizes. Two additional
failure-accounting regression tests passed for mixed valid/invalid environment
transitions and reset failure before export consumption.

The frozen model 21 then completed the prescribed 500-step Lab execution with
all 500 transitions finite and driven by the exported policy. Action and
estimator parity on those same GPU inputs had maximum error 0. One roll
termination triggered the original reset. Mean EE error was 0.458129 m,
minimum base-up dot product 0.593997, and base XY velocity-error RMS
0.259503 m/s. These are finite-window metrics, including terminal endpoints.

Independent CPU consumption of all 500 saved real histories matched the CPU
checkpoint outputs exactly. Compared with saved Lab outputs, maximum absolute
differences were 4.172325e-7 for actions and 1.490116e-7 for estimates, within
atol 1e-6 / rtol 1e-5. The report is
`outputs/unifp-execution-20260913/model21-actual-input-cpu-comparison.json`;
the original seven-synthetic-input comparison is retained separately.

[The runtime report](unifp-lab-runtime.md) documents the 20+1 updates, recovered
probe error, force-stage execution and the `gripperMover` property gap during
training. The evaluation used the repaired runtime, so its physics is not
claimed identical to that of the frozen checkpoint's short training run.
No full UniFP learning, learned force stage, MuJoCo transfer or hardware
validation was performed.
