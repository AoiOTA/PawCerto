# Fresh UMI training with relaxed joint-speed limits

The seed-0 policy learned useful fixed-weight MuJoCo tracking from random initialization. Its preselected final checkpoint completed all 16 requested 17-second cases without inversion or numerical failure. Position error is close to the official reference, with a remaining nonfoot-contact difference. Final native Lab evaluation and export verification also completed. The [fixed seed-1 and seed-2 repetitions](umi-relaxed-seed-repetitions.md) are the next experiment.

This is the documented **body-speed-v3 reconstruction with joint limits of 1000 rad/s**, a physical variant of the original URDF-limit recipe. It retains the original robot, trajectory pool, observation/control path, EMD formula and PPO. It does not establish exact original Gym force-sensor equivalence: the previously measured full-6D numerical failures and near-zero EMD sensitivity remain. The [preselected plan](umi-relaxed-velocity-adaptation.md#next-bounded-experiment-fresh-initialization-under-the-corrected-variant) and [reconstruction evidence](../outputs/diagnostics/umi-body-speed-damping/candidate-readout.md) preserve these limits.

## Training and initialization

The run used no pretrained weights or resume: seed 0, 4096 environments × 24 steps, 64 PPO epochs × four minibatches, 17-second task, eight previews and 132/261 actor/critic inputs. It completed all 4000 iterations and 393,216,000 transitions with actual exit 0 in 7175.405 seconds (119.590 minutes). Sampled GPU memory peaked at 6254 MiB.

Independent review checked the random initializer, fresh optimizer, all 41 checkpoints and all 4000 metric rows. The final Adam step is 1,024,000 for each of 17 parameter states; tensors and metrics are finite, native/cache limits agree, and all 32 recorded source files are unchanged. Final checkpoint SHA-256: `7015d72ad770ed5c3358480505bd9174b7c473f84229bc029ac9856244385a78`. See [execution](../runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4096/execution.json) and [validation](../runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4096/run-validation.json).

## Independent MuJoCo result

All groups use MuJoCo 3.13, the same `sample(16,2027)`, original 5 ms integration and full 17-second requests. Every trajectory in this comparison completed, so the common denominator is all 16 cases. The 500 checkpoint is a preselected diagnostic node, not the chosen final policy.

| Metric | Random initialization | Iteration 500 | Final 4000 | Official ours |
|---|---:|---:|---:|---:|
| Mean EE position / mm | 246.695 | 48.099 | 10.169 | 8.657 |
| EE position RMS / mm | 294.396 | 137.357 | 17.646 | 16.114 |
| Mean orientation / rad | 0.540687 | 0.189841 | 0.024536 | 0.025493 |
| Inverted cases | 0 | 2 | 0 | 0 |
| Invalid cases | 0 | 0 | 0 | 0 |
| Mean ground-supported feet | 4.000 | 3.674 | 3.927 | 3.947 |
| Zero-ground-support samples / % | 0 | 5.249 | 0.0515 | 0 |
| Nonfoot external-contact samples above 1 N | 0 | 698 | 395 | 38 |

Random standing is not successful tracking. The final policy improves position and orientation means substantially, while the intermediate policy inverts in cases 6 and 14. Final nonfoot contact is concentrated in case 7, principally the left rear calf. Its nonfoot fraction of summed upward ground force is 2.036%, with a maximum single saved endpoint fraction of 15.99%; across all 16 cases the summed fraction is 0.154%, versus official 0.0188%. These sampled measurements establish residual contact, without showing primary calf support or harmless contact. See the [contact-quality readout](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_seed2027/contact-quality-readout.md).

The [saved-state coordination readout](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_seed2027/coordination_readout.md) records simultaneous base, leg and arm movement. Across the 16 cases, median base XY net/path displacement is 0.158/0.634 m, versus official 0.071/0.394 m; median maximum foot clearance after the first second is 73.7 versus 56.2 mm. These describe motion, not its necessity or gait quality. Neither zero inversion nor body movement alone proves a correct gait or contact-free operation.

The [complete comparison and figures](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_seed2027/README.md) retain every case, endpoint and actual process result. [Final raw readout](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_seed2027/model_4000/readout.json) includes independent metric, clock, target and ground-support checks.

## Native Lab and exported policy

The current-variant nominal16 comparison uses the same 16 targets and 17-second requests. Both new and official processes exited 0 with all records finite, unchanged recorded sources and matching native/cache limits. Author-style evaluation uses 250 environments and the latest 500 completed episodes. Its official reference was reused only after consumed-input identity checks.

| Metric | Final 4000 | Official ours |
|---|---:|---:|
| Native nominal16 EE / mm | 12.910 | 9.940 |
| Native nominal16 orientation / rad | 0.025513 | 0.027401 |
| Author latest500 EE / mm | 21.455 | 20.076 |
| Author latest500 orientation / rad | 0.052952 | 0.053725 |
| Author timeout completion fraction | 95.0% | 97.8% |
| Author timeout-only EE / mm | 20.161 | 19.331 |
| Observed inversions, either protocol | 0 | 0 |

The new author run completed 503 episodes; the latest 500 contain 475 timeouts and 25 early terminations. Its early-termination mean EE error is 46.027 mm. Mean position tracking and timeout completion remain below the reference despite slightly better orientation error. See [nominal validation](../outputs/isaac/umi-relaxed-body-speed-scratch-nominal16-seed2027/comparison-validation.json) and [author validation](../outputs/isaac/umi-relaxed-body-speed-scratch-author-model4000-seed2027/validation.json).

The [final export package](../outputs/export/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4000/README.md) passed zero-error actor parity. Original checkpoint and exported-directory consumption each completed an independent 17-second MuJoCo process with exit 0, all seven saved arrays exactly equal and matching JSON except the source path. This separate `sample(1,0)` check has 7.532 mm mean EE error and no inversion; it proves package behavior preservation and does not replace the 16-case comparison.

## Interpretation boundary

The user clarified on 2026-09-12 that exact numerical equivalence between Isaac Lab and Isaac Gym is not an acceptance requirement. Acceptance instead concerns preserved UMI method semantics, actual learning in Lab, fixed-policy MuJoCo behavior, and transparent tracking/failure/contact/repetition results. Existing native-6D residual measurements remain diagnostic evidence; numerical inequality alone is not a reason to leave the project incomplete. The relaxed joint-speed limit is an explicit configuration choice and remains documented separately from inherent engine differences.

This meets the preselected nominal MuJoCo condition for this seed and configuration. It is one training seed evaluated on the supplied trajectory pool, not an independent trajectory holdout, a multi-seed result or full paper reproduction. Native Lab results and residual support behavior are reported separately above; repeatability is being tested under the fixed repetition plan. Earlier failed URDF-limit and weights-only adaptation results are retained in the [experiment history](umi-relaxed-velocity-adaptation.md).
