# Reconstructed-force seed 0: completed training, unstable transfer

The preselected 4000-iteration run completed, but its final policy did not achieve stable MuJoCo transfer. Isaac Lab evaluation shows useful learned behavior; the independent engine exposes severe failures. The conditional seed-1/2 repetition is therefore not started. A completed paired nominal Lab replay confirms strong tracking on the same target draw in the training engine, while preserving the failed MuJoCo endpoint.

## Training evidence

`runs/umi_reconstructed_solver_velocity_seed0_4096` starts from random seed 0, without pretrained weights or optimizer recovery. It retains the published `ours` recipe: 4096 environments, 24 transitions per rollout, 64 PPO epochs, four minibatches, 17-second tasks and 132/261 actor/critic inputs. The selected engine is fixed Lab `412fb31b30ee605b4ffec4327436fc0fe53281d8` with Sim 6.1.0.0. Runtime metadata identifies physics v3 and reconstruction v2.

The [actual run](../runs/umi_reconstructed_solver_velocity_seed0_4096/execution.json) exited 0 after 393216000 transitions. All 4000 metric rows and 41 checkpoints were independently checked; final Adam step is 1024000, tensors are finite and the 32 recorded training sources did not change. Total process time was 7663.493 seconds (127.72 minutes), mean iteration time 1.884 seconds, mean collection time 1.264 seconds, and sampled GPU-process memory peak 6254 MiB. These checks establish complete execution, not behavior acceptance. See [completion validation](../runs/umi_reconstructed_solver_velocity_seed0_4096/training-completion-validation.json).

## MuJoCo fixed trajectories

The fixed draw is `sample(16,2027)`, requested for 17 seconds in MuJoCo 3.13. Initialization was reused only after exact tensor, inference-configuration, source and target checks. Iterations 500 and 4000 were actually run on every requested case. The official reference is the previously executed 3.13 regression group.

| Node | Requested | Complete | Numerically invalid | Inverted complete cases | Inverted invalid prefixes |
|---|---:|---:|---:|---:|---:|
| Random 0 | 16 | 16 | 0 | 0 | 0 |
| 500 | 16 | 16 | 0 | 3 | 0 |
| 4000 | 16 | 15 | 1 | 5 | 1 |
| Official ours | 16 | 16 | 0 | 0 | 0 |

The following comparison uses the same 15 complete cases in every column, excluding case 12 from these means while retaining it in the request/failure counts above.

| Metric, common 15 cases | Random 0 | 500 | 4000 | Official |
|---|---:|---:|---:|---:|
| Mean position error, mm | 249.170 | 71.312 | 359.468 | 8.730 |
| Position RMS, mm | 296.667 | 198.663 | 920.398 | 16.291 |
| Mean orientation error, rad | 0.540759 | 0.154385 | 0.559554 | 0.025409 |
| Inverted cases | 0 | 2 | 5 | 0 |
| Mean ground-supported feet | 4.000 | 3.644 | 2.815 | 3.946 |
| Samples without ground-supported feet | 0% | 3.691% | 23.997% | 0% |

Nine of the 15 cases have better mean position error at 4000 than at 500, but severe failures dominate the final aggregate. This is neither uniform deterioration nor stable improvement. Final case 12 inverted at a saved endpoint of 1.92 seconds and later triggered BADQACC at approximately 14.66 seconds. Its 732-point prefix is retained; the child exited 2 and the batch exited 1. No reset state was included as a successful continuation. There is no valid 16-case full-duration mean for the final policy.

See the [full MuJoCo report](../outputs/mujoco/umi_reconstructed_solver_velocity_seed0_seed2027/README.md), [case-level readout](../outputs/mujoco/umi_reconstructed_solver_velocity_seed0_seed2027/model_4000/readout.json) and [inversion timelines](../outputs/mujoco/umi_reconstructed_solver_velocity_seed0_seed2027/model_4000/inversion-readout.md).

## Isaac Lab author-style evaluation

Both policies use seed 2027, 250 environments, the original evaluation configuration and stopping rule, and the latest 500 completed episodes. Both evaluation processes exited 0. This protocol includes physical randomization, observation noise and early terminations; it differs from the nominal full-duration MuJoCo protocol.

| Metric, latest 500 completed episodes | Official ours | New 4000 |
|---|---:|---:|
| Mean position error, mm | 26.162 | 22.292 |
| Mean orientation error, rad | 0.066330 | 0.058089 |
| Timeout fraction | 79.8% | 91.8% |
| Inverted episodes | 0 | 2 |
| Timeout-only mean position error, mm | 19.558 | 20.960 |
| Timeout-only mean orientation error, rad | 0.05240 | 0.05488 |
| Mean ground-supported feet | 3.8255 | 3.5919 |
| Samples without ground-supported feet | 0.3124% | 0.1118% |

Overall means improve while the timeout-only means are slightly worse; the episode-duration mix matters. Timeout fraction is a survival proxy, not task success, and two inverted episodes remain. Raw episodes and summaries were independently recomputed. See [official evaluation](../outputs/isaac/umi-author-eval-velocity-seed2027-official/summary.json) and [new-policy evaluation](../outputs/isaac/umi-author-eval-velocity-seed2027-model4000/summary.json).

## Interpretation and next diagnostic

Finite training and positive native-engine results do not establish stable WBC transfer. The paired nominal Lab replay completed both policies on all 16 targets for the full requested 17 seconds, without inversion. Official/new-4000 mean position errors are 10.080/12.476 mm and orientation errors are 0.026456/0.027990 rad. This reduces the protocol difference and supports a transfer vulnerability; it does not isolate its physical cause. Several physical mappings changed alongside the reconstructed force input. The [paired raw rollout](../outputs/isaac/umi-paired16-velocity-seed2027/rollout.json) retains both groups. Its saved rows do not contain actual engine time; source-derived float32 Lab accumulation can shift discrete target indexing by one 5 ms frame relative to MuJoCo. The analysis retains this timing limitation and does not identify it as the failure cause. It does not replace the failed primary endpoint or select a different checkpoint.

The [case-by-case comparison](../outputs/diagnostics/umi-lab-mujoco-velocity-seed2027/paired-comparison.png) shows all 16 cases and separately marks the invalid prefix. Saved MuJoCo joint velocities exceed the original URDF limits before first inversion in all six failing cases, reaching 4.077–12.225 times the limits. The official policy also briefly exceeds those limits (1.108–1.390 times over the matching intervals), so overspeed alone is not a sufficient explanation. See the [pre-inversion velocity readout](../outputs/diagnostics/umi-lab-mujoco-velocity-seed2027/velocity-analysis.md). The isolated Lab intervention completed with exit 0 after setting and reading back all 18 native joint limits at 1000 rad/s. Both policies completed all 16 requested cases with finite records; sources, weights and sampled targets were unchanged. Official mean position error stayed near 10 mm (10.080 to 9.940), with no inversion. New-4000 mean position/orientation errors rose from 12.476 mm / 0.027990 rad to 224.026 mm / 0.472518 rad, with 6/16 inversions and zero-ground-support samples rising from 0.1104% to 15.1428%. This controlled intervention supports dependence on the joint-speed constraint. Its inverted cases (0,1,2,6,7,8) differ from MuJoCo, so it does not explain every engine difference. See the [intervention and comparison](../outputs/isaac/umi-paired16-relaxed-velocity-seed2027/comparison-validation.json). Main code and training configuration were not changed.

An independent source check confirmed that the saved training configuration uses `dense_rewards=False`: both its training path and the nominal Lab/MuJoCo observers advance pose history every 20 ms. The author-style evaluation explicitly switches to dense updates every 5 ms. The nominal observer is therefore consistent with this checkpoint; no history-frequency repair was made.

The original EMD formula remains unchanged, but positive force residuals near zero can change its flying branch. The earlier high-gyro diagnostic also retains three angular components outside the original tolerance. These limits and the lack of pointwise original Gym sensor equivalence remain in the [reconstruction report](umi-stock-solver-force.md). All trajectories come from the published training pool, not an independent holdout. No additional seeds of the failed configuration are started. The measured constraint dependence motivates a separate [fixed-budget adaptation experiment](umi-relaxed-velocity-adaptation.md), explicitly labeled as a physical variant. No public release or hardware experiment is performed.

The [final original-limit export package](../outputs/export/umi_reconstructed_solver_velocity_seed0_4000/README.md) preserves the final weights and execution configuration. CPU actor export error is zero; original-checkpoint and exported-package `sample(1,0)` self-tests both exited 0 with exactly equal saved arrays. That separate deployment case does not replace the failed fixed-16 endpoint.
