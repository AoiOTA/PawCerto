# UniFP official Isaac Lab integration

The B2/Z1 runtime ports the pinned UniFP configuration and tensor equations to official, unmodified Isaac Lab/PhysX. Source revision is `68847a070f88d731058c3d8476929bc3b205f5bd`. This document distinguishes execution of the port from completed UniFP learning; the authorized integration budget is 16 environments × 24 rollout steps × at most 21 optimizer updates, a separate-process short resume, and an explicitly artificial force-stage boundary probe. Full 8000+ update training is not authorized or demonstrated.

## Entry points and configuration

`pawcerto.methods.unifp.training.config.default_config()` returns a complete JSON-compatible `{'env': original_environment_configuration, 'train': original_PPO_configuration}`. `resolve_config(config)` returns the attribute-style environment configuration used by the equations, without changing source mapping fields. The training script writes the resolved configuration to the run directory and embeds it in every checkpoint.

After the official `AppLauncher` starts:

```python
cfg = resolve_config(config)
terrain = UniFPTerrain(cfg.terrain)
runtime = B2Z1Isaac(cfg.env.num_envs, device, robot_usd, cfg, terrain)
env = UniFPIsaacTrainingEnv(runtime, config)
```

The runtime initializes physics but executes no physics step in its constructor. The environment constructor allocates source tensors but does not reset/step. `env.reset()` preserves source `BaseTask.reset`: reset all environments, execute one zero-action policy step, return `(observation_dict, rewards)`. This real warm-up step increments `global_steps`. `get_observations()` and `step(actions)` retain the original runner protocol and full 2336 actor / 447 critic / 12 supervised / 73 single-frame dimensions. There are 19 joints and 17 policy actions, no locked-arm or reduced-state substitute. The runtime's public tensors use xyzw quaternions, matching the pinned UniFP equations and the current official Lab backend.

`env.transition_observer`, when non-None, is called after physics, command/goal updates, termination and reward computation, immediately before automatic per-environment reset. Evaluation can collect terminal states there without changing reset behavior or copying training rollout tensors. `reset_buf` and `time_out_buf` are the original masks. `env.force_stage_info` is a read-only summary of the actual global policy-step count, strict threshold, active flag, and maximum known injected EE/base forces.

## Preserved method and simulator bindings

- `UniFPEquations` retains original network observations, supervision, force/goal schedules, PD equation and all 27 active reward terms. The source lifecycle initialization, parse/reset and force-gain randomization bodies live in `training/lifecycle.py`; only simulator tensor acquisition is replaced by measured Lab tensors. No PPO/estimator equation changes were made.
- The source 0.005 s physics step and four-step decimation are preserved. All 19 PD efforts are saturated by source asset effort limits; only the first 17 targets depend on policy actions. The last two joints retain their original nominal target and 64/1.5 gains.
- Training force activation remains `global_steps > force_start_step * 24`, tested before each physics substep. The literal 24 is independent of runner rollout length. `_push_gripper` is called on that path; `_push_robot_base` remains uncalled, as in source.
- Known world-frame external force buffers are applied with the public permanent wrench composer at body COM. Labels and force-offset rewards read that same known buffer. They are not contact force, joint effort, or an inferred force sensor. Contact force is separately reconstructed as the official ContactSensor net normal vector plus its filtered tangential vectors, with ground and self-contact partners included.
- Full source reset ranges are retained: first twelve joint angles scale nominal by [0.5,1.5], first five arm joints add [-0.5,0.5] rad, root xy adds [-1,1] m within the sampled terrain origin, root yaw is ±pi/2, and all six root velocities are sampled in [-0.5,0.5]. Force buffers/history reset and source reward/time-out ordering remain intact. Source velocity pushes every 8 s are retained.
- Friction uses 256 source buckets in [0.3,2.0]; motor offset strength uses [0.85,1.15]. Actual base/gripper mass and source-index COM changes use public setters, with the 22-value privileged randomization vector retained. Default disabled leg-mass randomization contributes seventeen zeros.

## Terrain and assets

The original active terrain is rough-flat, not a flat-plane replacement: 10 × 20 tiles, 8 m per side, horizontal scale 0.05 m, vertical scale 0.005 m, downsample scale 0.075 m, and 25 m border. Each source roughness draw uses the original Python/NumPy RNG producer order. SciPy removed `interp2d`; its regular-grid linear replacement is `RectBivariateSpline(kx=1, ky=1)` with source integer quantization and `rint` preserved. Delatin uses the original flip/transpose, vertical scaling and `max_error=0.1` configuration.

The seed-0 CPU terrain build produced a 2600 × 4200 heightfield, min/max -0.045/+0.045 m, 3,995,996 vertices and 7,991,986 faces. Each original face is retained once. Faces are partitioned into at most 250,000-face collision shapes to fit PhysX BV4 cooking; only vertex indices are rebased. This changes collision-shape boundaries, which can affect contact solving, and is not a claim of bitwise Gym contact parity. Static geometry is shared across all environments.

`scripts/convert_unifp_usd.py` imports the original URDF using official Lab tooling. Fixed joints marked `dont_collapse` remain distinct, including feet, arm mount and EE marker. Collision offsets are authored in shared geometry layers; runtime clones keep geometry instances shared. The actual converted path is `reference/isaac/b2_z1/merged/merged.usda`, recorded by `usd_path.txt`.

## Checkpoint and resume

The training runner subclasses the original implementation only for checkpoint/context and metric output. Both original Adam optimizer states, adaptive learning rate, model state and completed iteration are preserved. Checkpoints additionally contain complete `config` and actual `global_steps`. Resume restores these values and continues the strict force stage; physical episodes restart. It is not deterministic mid-rollout state replay and does not claim RNG or contact-cache continuation.

The explicit `--force-boundary-probe` executes only after saving the training checkpoint. It sets artificial `global_steps` to the boundary and episode time near the sampled external-force schedule, then executes three zero-action policy steps. Pre-reset force/contact/body snapshots label each actual stage used. Those steps do not constitute trained force-stage behavior or a learned policy evaluation, and the artificial stage state is never saved over the training checkpoint.

## Physics migration limits

The original property producer adds base mass to `props[0]` but adds the value named base COM to `props[1]`. The port must preserve the actual named body corresponding to those indices; the original-Gym asset/property probe confirmed these names as `base_link` and `FL_hip`, respectively. All 26 body names and all 19 joint names match the converted source asset. The COM change is applied to `FL_hip`, preserving the actual producer.

Gym uses `set_actor_rigid_body_properties(..., recomputeInertia=True)`. The public Lab port scales each original inertia by its mass ratio and writes COM separately. A seed-0 original Gym property probe found a maximum absolute difference of 2.58e-7 between this scaling rule and Gym recomputed inertia across all 26 source bodies. This supports that one property-update rule; imported initial inertias and cross-engine dynamics are not established equivalent. Collision/contact solving, asset importer output and fresh-process reset also prevent claiming bitwise trajectory equivalence. No private solver buffers, engine patches or custom PhysX rebuilds are default dependencies.

## Execution evidence

CPU source/equation/optimizer/history/export checks: 12 passed before simulator execution. The seed-0 complete terrain built successfully in the official Lab Python environment. The missing `pydelatin` dependency was installed as 0.3.0 by the resource operator. Official asset conversion completed with exit code 0. The first conversion invocation failed because this Lab AppLauncher does not accept `--headless`; the same conversion used supported `--visualizer none` successfully, with both logs retained in `outputs/unifp-execution-20260913/`.

The real 16-environment run completed 20 updates (480 rollout policy steps plus one source reset warm-up), exited 0, and saved `train/model_20.pt` at global step 481. Separate-process resume completed update 21 and saved `resume/model_21.pt` at global step 505. Its subsequent boundary reset failed with PyTorch's “inference tensor outside InferenceMode” write error; that process exited 1 and its log remains `resume.log`. The error occurred after the checkpoint was saved. The probe reset was moved into the proper inference context and rerun separately with **zero optimizer updates**, using the frozen model 21.

The corrected separate boundary process exited 0. Its three real transitions used global steps 192000, 192001 and 192002, with active flags false, true and true. The first two injected-force maxima were zero (inactive stage, then start of the original ramp); the third was 0.96923691 N, nonzero in 12 of 16 environments. Contact vectors were saved separately. This demonstrates execution of the strict schedule and actual runtime application path, not learned force control or cross-engine contact/trajectory equivalence.

`resume-check.json` compares the original model 20 with the checkpoint saved immediately after loading in the separate process: model parameters, both optimizer states, adaptive learning rate, completed iteration and global steps are exactly equal. Model 21 has finite parameters, with both Adam state step counters at 420 (21 updates × 5 epochs × 4 minibatches). Its SHA-256 is `c88f38ea0d76a88903de39def1885f56bc3695e503bba362eb1fa55911b1e80a`. All observed training losses are finite; adaptive learning rate reached its original 1e-5 floor. These are integration and optimizer-continuation results, not learning acceptance.

`terrain-parity.json` records a full seed-0 CPU comparison against the original pinned terrain builder executing original Gym `random_uniform_terrain` with SciPy 1.10.1 `interp2d`: all 10,920,000 heightfield cells match exactly. The comparison skips mesh conversion only on the original isolated reference side; the actual Lab training uses the full Delatin mesh. The original-Gym property probe also completed with zero physics rollout steps; its initial missing `libpython3.8.so.1.0` error was resolved using the existing Conda `lib` directory in that subprocess's `LD_LIBRARY_PATH`, without rebuilding or modifying Gym.

A **real training configuration gap** was found in `train.log`: the Lab nested rigid-property modifier skipped `gripperMover`, which is itself an instance root. The 21-update checkpoint was produced with that body's source damping/velocity/gravity overrides not successfully applied. The current runtime repairs this by authoring the same public `PhysxRigidBodyAPI` attributes directly on the instance root while retaining shared geometry. No additional optimizer updates were run and the frozen checkpoint was not rewritten. The single subsequent 500-step evaluation uses this repaired runtime; training/evaluation physics are therefore not claimed fully identical. The evaluation records its current runtime source SHA separately.

Reproduction commands (use the official Lab Python environment):

```bash
python scripts/convert_unifp_usd.py --visualizer none --output reference/isaac/b2_z1
python scripts/train_unifp.py --num-envs 16 --iterations 20 --seed 0 --output outputs/unifp-execution-20260913/train
python scripts/train_unifp.py --resume outputs/unifp-execution-20260913/train/model_20.pt --iterations 1 --output outputs/unifp-execution-20260913/resume
python scripts/train_unifp.py --resume outputs/unifp-execution-20260913/resume/model_21.pt --iterations 0 --force-boundary-probe --output outputs/unifp-execution-20260913/force-boundary
```

The commands describe the corrected entry points. Archived command/source snapshots and original logs retain the exact runs, including the failed combined resume/probe. Running them again would consume additional training budget and is not part of this completed bounded check.

### Final single-checkpoint physical evaluation and consumer check

The only 500-step physical evaluation used model 21 on one environment and exited 0. All 500 observations and environment transitions were finite; the exported TorchScript student actually drove all 500 steps. Raw network versus exported action and estimate errors on the same GPU observations were both exactly zero. The original automatic termination/reset path recorded one roll termination. Mean EE position/compliance-target error across the finite execution window was 0.458129376 m, minimum base-up dot product 0.593997359, and xy-velocity RMS error 0.259503037 m/s. These are bounded execution-window metrics, not successful locomotion/manipulation learning.

The repaired `gripperMover` instance-root authoring path executed during this evaluation: the earlier nested-modifier failure warning is absent from `eval500.log`, and the process completed normally. The evaluated runtime SHA-256 was `e7046ebb3154969fc4cbfe87a46a9a1f4308afbacbcf5d4d56077608cfd3c315`, retained in `eval500-runtime-source.json`. This does not retroactively fix the training-time coverage gap for the frozen model 21.

A fresh CPU consumer reloaded the export and checkpoint and processed all 500 actually observed history tensors, exiting 0. CPU export/checkpoint action and estimate errors were zero; CPU versus saved Lab outputs differed by at most 4.172325e-7 for actions and 1.490116e-7 for estimates, within the stated atol=1e-6, rtol=1e-5. This is real observation/export-consumer parity, not policy-quality evidence.

Final artifacts under `outputs/unifp-execution-20260913/`:

- `train/`, `resume/model_21.pt`, `resume-check.json`: 20+1 update execution and exact optimizer-load continuation.
- `source-asset-properties.json`, `terrain-parity.json`: original-Gym static properties and full original heightfield comparison.
- `resume.log`: original post-checkpoint boundary reset failure, preserved.
- `force-boundary/evidence.json`: independent zero-update three-step boundary execution.
- `model21-export/`, `model21-eval500/summary.json`, `model21-eval500/trajectory.npz`: exported model and single physical evaluation.
- `model21-actual-input-cpu-comparison.json`: independent CPU reload of all real observation histories.

The authorized simulation/update budget is complete. No full UniFP training, force-stage learning, cross-engine contact parity or real-robot readiness is claimed; no additional simulator runs or optimizer updates were launched to pursue those unassigned outcomes.
