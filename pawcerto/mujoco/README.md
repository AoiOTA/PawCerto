# Independent Go2 + ARX5 MuJoCo path

New evaluation defaults to Conda `pawcerto-mujoco` with MuJoCo3.13.0, Python3.12.14, Torch2.11.0+cpu, NumPy2.5.3 and SciPy1.18.1. Activate it with `conda activate pawcerto-mujoco` and run `python` from the repository root. The [actual Conda migration](../../outputs/mujoco/conda-runtime-migration/README.md) completed the existing final seed0 package's17-second historical condition with all seven arrays and all JSON fields exactly equal to the prior3.13 interpreter. This is one environment-consumption check; the [three-seed fixed16 evaluation](../../outputs/mujoco/umi_relaxed_body_speed_three_seed_summary/README.md) remains the performance evidence.

The formal fixed16 results were produced by `third_party/mujoco-3.13-probe`, following the [48-case version regression](../../outputs/mujoco/version-3.13-regression/README.md). Their original interpreter records and the older3.8 environment are retained. The commands below use the current Conda environment; historical experiment results retain their original interpreter provenance. No model, controller or integration parameter changed for migration.

`asset.py` imports the released `go2_arx5_finray_x85_z94.urdf`. It keeps the full floating robot, all 41 named links, explicit masses/inertias and fixed transforms. Collision meshes remain; DAE visual meshes are omitted. Cylinders become capsules as in the source asset option. No upstream files are modified. Rebuild after moving the workspace because mesh references are absolute.

```sh
conda activate pawcerto-mujoco
python scripts/build_mujoco.py
python -m unittest discover -s tests -p test_mujoco_asset.py -v
python -m pawcerto.mujoco.evaluate --seconds 17 --output outputs/mujoco/official_example.json
```

`Go2Arx5Mujoco.state()` reads physical joint q/qd, root and EE world poses and velocities. `step(action)` executes four 5 ms physics steps with the released PD, scaling, torque limits and control delay. State quaternions use xyzw; rotations map local to world. Joint arrays currently follow the actual URDF/MuJoCo traversal: FL, FR, RL, RR (hip/thigh/calf each), joint1..joint6. The original IsaacGym runtime confirmed the same order in `reference/isaac/original_gym_joint_names.json`.

The source-URDF comparison checks 32 random configurations against independent rigid-transform FK for all 41 links, plus individual inertial tensors and total mass. Measured maximum FK component discrepancy was 8.88e-16 and total mass was 18.879 kg. The actual 2 s default-PD diagnostic verifies floating-base stepping, torque bounds and the causal one-policy-step action delay. A finite-difference test checks root and EE link-origin linear velocities (MuJoCo body velocities must be shifted from the inertial center).

`evaluate.py` consumes the shared UMI actor/observation/trajectory implementation and records both EE errors and root support/orientation throughout a full 17 s trajectory. Seed-0 results after matching the original state-observation key sorting and reset zero-action step:

| Released model | EE mean / RMS position error (m) | Mean orientation error (rad) | Minimum root height (m) | Minimum root up dot |
|---|---:|---:|---:|---:|
| ours | 0.00892 / 0.01464 | 0.02908 | 0.20037 | 0.98178 |
| ours-real | 0.00568 / 0.01247 | 0.02210 | 0.18125 | 0.96759 |

Both executions completed with finite state, upright root orientation and low EE tracking error on this full trajectory. This is an actual fixed-weight sim2sim execution result, not learned Isaac Lab WBC acceptance or a general success-rate estimate. They use nominal masses, deterministic initial state, no observation noise and MuJoCo contacts/solver; original PhysX contact behavior and randomization have not been shown equivalent. The two checkpoints differ in observation dimensions, control delays and torque limits, so this comparison does not isolate a cause.

Earlier runs with incorrect state-observation ordering tumbled; those diagnostics are retained separately under `outputs/mujoco/diagnostic_before_observation_order_fix_*` and cannot be used to assess the released method. Current results are `outputs/mujoco/evaluation_ours_seed0.json` and `outputs/mujoco/evaluation_ours_real_seed0.json`.

## Historical Pilot paired training-checkpoint evaluation

For the real 100-iteration Isaac Lab adaptation Pilot, each model evaluates exactly one `sample(16, 0)` draw, sliced into 16 cases. It does not replace this draw with separate single-case seeds. Each case includes 20 ms reset warmup and 849 actor steps, totaling 17 seconds. To rerun this historical condition using the current Conda
interpreter (requires the historical export bundle), use a fresh output directory:

```sh
python -m pawcerto.mujoco.evaluate --checkpoint outputs/export/umi_lab_transfer_100 --cases 16 --seed 0 --seconds 17 --output outputs/mujoco/umi_lab_transfer_rerun/model_100
```

Each case produces JSON metrics and an NPZ with sampled targets and actual EE/root poses, q/qd, raw actions, last-substep torques and foot world-z forces. `supported_feet` counts named feet whose summed contact world-z force exceeds 1 N; inversion means an observed root up-dot below zero. These are reported measurements, not a new task success criterion.

All 80 runs across iterations 0/25/50/75/100 completed with finite physical state; saved target position and rotation arrays were checked equal across all checkpoints. The aggregate and per-case deltas are in `outputs/mujoco/umi_lab_transfer/paired_summary.json`. Model 100 versus model 0 had 3.86% higher mean EE position error and 33.60% higher mean orientation error, with zero inverted cases and no zero-foot-support samples in either model. This paired MuJoCo evaluation therefore does not establish improved joint tracking and support after the Pilot.

## Non-foot / non-finger contacts (added before scratch iteration 2000)

The existing contact-force read now also retains `nonfoot_contact_records` in each new case JSON: time, both body names and geom IDs, external/self classification, contact position/separation, world force vector, normal force and force norm. Geometric contacts below threshold remain in the raw records. Summary occupancy counts require a **per-contact force norm strictly greater than 1 N**; this threshold is a reporting convention, not a task-success rule.

Only the four named foot bodies and the two official `collider` finger-mesh geoms are excluded. `link6` wrist/structural geoms are still recorded; the whole link is not silently exempted as a gripper. A contact is external if one geom belongs outside the robot base tree; otherwise it is self-contact. Multiple contact points in one policy sample count once per external/self category.

Sampling occurs at the existing policy endpoint after `runtime.step()`/`mj_forward`, every 20 ms. `nonfoot_contact_sampled_occupancy_s` is flagged-sample count times 20 ms, **not the full continuous collision duration**: the three intervening 5 ms substeps and brief contacts between samples are not observed. Earlier evaluation files lack these records; absence of a field is not evidence of no non-foot contact. No older candidate was rerun to backfill this measurement.

`test_mujoco_contacts.py` verifies unchanged integration state, solver forces and subsequent physical stepping after the reader runs; actual lowered and folded model fixtures verify external/self classification, finger exclusion and force thresholds.

## Direct foot-ground audit

The legacy `supported_feet` fields still mean foot **net contact** world-Fz >1 N, including self contacts. New `ground_supported_feet` fields use only contacts between each foot geom and the actual named ground geom. NPZ files add `foot_ground_force_z_N` and `ground_supported_feet`; their rows align with the existing metrics. Each case JSON splits all sampled contacts into `nonfoot_contact_records` and `foot_or_finger_only_contact_records`, with no contact point omitted from their union.

Fresh 16-case, 17-second audits of official ours, scratch500 and scratch2000 are under `outputs/mujoco/contact-audit/`. All original physical and metric arrays matched the previous runs bit-for-bit. Net and ground foot counts matched for official ours and scratch500. For scratch2000 they differed at 960 samples; zero-ground-foot fraction was 6.05%, while the legacy zero-net-force-foot fraction was 2.39%.

Scratch500 still had sustained RR-calf ground contact in several cases, but sampled non-foot upward ground-force share was only 0.166% across the full group. This establishes contact, not that the policy primarily relies on calf support. Official ours also had brief calflower1 ground contacts. None of these descriptive 1 N counts creates a new task success/failure rule. The author Gym sensor may include solver joint-constraint forces; neither our legacy net-contact sum nor ground-only force has been claimed exactly equal to that sensor.

## Numerical-failure qualification

MuJoCo may automatically reset to finite positions after a BADQACC warning. `runtime.step` now propagates a numerical warning or clock rollback as `SimulationInstability`, checked after each physics step. This was an observed scratch4000 case3 failure at16.88s, not a new physics setting. A real engine auto-reset regression test covers the old finite-qpos blind spot.

An interrupted case stores only its pre-failure records, `completion_status=invalid`, `metrics_scope=pre_failure_prefix`, and the engine warning/time details. Batch output retains the requested denominator and all cases. If any case is invalid, full-set mean fields are absent; `complete_case_metrics` explicitly describes only the complete subset. The CLI exits2 after saving an incomplete result. Prefix metrics must never be pooled with full17s cases.

The scratch4000 case3 retry reproduced the same16.88s failure once, with its843pre-reset rows identical to the original prefix. Old erroneous post-reset records/aggregate are preserved under `model_4000/pre_failure_propagation/`. Other fifteen4000cases and previous0/500/1000/2000 clocks were verified from existing arrays without unnecessary reruns.

The independent4000export is `outputs/export/umi_lab_scratch_4000`. Exported actor reload error was0; direct17s package consumption is saved as its `deployment_rollout.json`. That additional `sample(1,0)` check is distinct from the paired `sample(16,0)` set and does not replace it.
