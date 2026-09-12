# Fixed-budget joint-velocity adaptation experiment

This is a physical-configuration variant of the reconstructed-force UMI port. It is not the original URDF velocity-limit recipe or an uninterrupted continuation of its optimizer. The default runtime retains the original limits.

## Evidence and decision

The completed random-initialized 4000-iteration chain learned useful native Lab behavior but failed fixed-weight MuJoCo transfer: five inversions among 15 complete trajectories and one additional inverted BADQACC prefix. A controlled native Lab intervention changed only the 18 joint velocity limits to 1000 rad/s. Official-policy tracking stayed near 10 mm without inversion, while the new policy deteriorated from 12.476 to 224.026 mm and from zero to six inverted cases. This supports dependence on that constraint without attributing every engine difference to it. The [result report](umi-reconstructed-learning-result.md) preserves the primary failures and diagnostic limits.

The saved training configuration uses sparse pose-history updates every 20 ms. Both nominal deployment paths preserve that cadence; an independent source check found no missing original substep history update. The observer is not changed for this experiment.

## Preselected execution

- New output: `runs/umi_reconstructed_solver_relaxed_adapt_seed0_4096`.
- Initialization: weights only from `runs/umi_reconstructed_solver_velocity_seed0_4096/model_4000.pt` (SHA-256 `d8bccb0c77f81a47a007958b413ccbc004799a1ed1fee8d635e55f5d97eb9005`). Actor, critic and policy standard deviation are retained; Adam, iteration, transition counters, random state and curriculum start anew. The original optimizer is not resumed across the physical contract.
- Fixed endpoint: 1000 iterations, seed 0, 4096 environments × 24 steps, 64 PPO epochs and four minibatches, 17-second task, 132/261 observations, eight previews. This adds 98,304,000 transitions. Save every 100 iterations; no checkpoint selection.
- Sole intended physical change: all 18 native joint velocity limits are 1000 rad/s. The same readback must initialize the reconstructed-force cache. The original EMD formula, PPO, task, gains, other physical properties and solver guards remain unchanged.
- Before the full run: existing relevant tests, independent review and one real 4096-environment iteration establish configuration propagation, weights-only initialization and functioning updates. Its weights do not enter the full chain.

This budget is a bounded adaptation experiment informed by the intervention, not an assumed guarantee of stable transfer. A genuine runtime failure ends the attempt with its records intact; no automatic change to guards, reward or endpoint is allowed.

## Fixed behavioral readout

Use the same MuJoCo 3.13 `sample(16,2027)` for iteration 0 and 1000, each requested for 17 seconds. Reuse iteration-0 results only after exact policy and consumed-inference-input identity is verified. Otherwise execute all 16 again. Keep numerical failures, inverted prefixes and full-duration means separate. Compare final tracking and orientation jointly with inversion, ground support and nonfoot contacts; retain the official reference and the original failed endpoint.

The immediate behavioral objective is no observed inversion or numerical failure in the requested set, together with lower position and orientation error than the initial adapted-from policy on the common complete cases. This is a bounded engineering result on the supplied training pool, not independent-holdout validation or formal multi-seed reproduction. A completed update budget alone does not meet it.

The final native evaluation must retain the variant contract. Run the same nominal fixed-16 target draw and the author-style completed-episode protocol, with an official-policy reference under the same physical variant where needed. Changes in episode-duration mix remain explicit. If the variant improves transfer, its physical deviation remains part of the result and cannot be relabeled as original-method equivalence.

## Current state

Implementation, the real 4096-environment iteration and independent review passed. The [preflight](../outputs/isaac/relaxed-velocity-adaptation-one-iteration/validation.json) confirmed all 17 model tensors match the source, an empty initial Adam, zero counters, matching native/reconstruction limits before and after the update, 98,304 transitions and finite updated tensors. It exited 0 in 124.282 seconds with a 5904 MiB sampled GPU peak. The separate full chain launched at 2026-09-11 21:53:03 UTC and exited 1 after 43 complete iterations (4,227,072 transitions), during collection for iteration 44. It did not reach the fixed endpoint; no model 1000 or final evaluation exists. The partially collected rollout is not counted as a complete update. Only model 0 was saved before the planned save-100 interval. See [execution](../runs/umi_reconstructed_solver_relaxed_adapt_seed0_4096/execution.json) and [failure validation](../runs/umi_reconstructed_solver_relaxed_adapt_seed0_4096/failure-validation.json).

The actual CUDA path rejected an unsupported body-angular-velocity clamp branch. CPU recomputation of the captured finite PRE state found environment 2405 FR_thigh at 1024.492 rad/s and FR_calf/FR_foot at 1316.682 rad/s versus the original 1000 rad/s body limit. Native joint limits and reconstruction caches still matched, and source hashes were unchanged. This is an observed unsupported physics branch, not a completed adaptation or evidence that the requested budget failed to learn. The run was not restarted and the guard was not relaxed. The subsequent source inspection and implementation are documented below.

The independent model-service capacity errors around the same time are recorded separately; the training failure has its own Python traceback and actual process exit. Total process wall time was 206.271 seconds and sampled GPU-process memory peak 6254 MiB.

## Restart after adding the source-supported body-speed term

The source investigation identified the omitted body-speed damping wrench and its absence from public C. Implementation and independent review found no formula error. Native controls establish that omitting the term produces much larger errors, but complete 6D numerical validation still fails: the production multiplication order had 21/150 angular failures in the linear-speed sample and 1/150 linear failure in the qualified angular-speed sample. Matching the source multiplication order reduces the first count to 10/150; the second remains unchanged. The tolerance remains `1e-3 + 1e-4*abs(prediction)`. The angular free-dynamics fixture uses diagnostic DOF limits of 1e9 to exclude solver interference; this is not the training configuration.

A separate same-data Jacobian-geometry check did not remove these failures and is not adopted. The source multiplication order is adopted as the smaller source-aligned change. The new term's qualified free-foot residuals still change the original EMD flying branch, so numerical sensor/reward equivalence remains unproven. The old high-gyro failures are also retained. These are limits of this reconstructed-signal research experiment, not results relabeled as PASS.

The next weight chain is `runs/umi_reconstructed_solver_relaxed_body_speed_adapt_seed0_4096`, with `reconstruction_revision=stock-physx-body-speed-damping-v3`. It starts again from the original completed model4000 using weights only and the same fixed 1000-iteration adaptation budget and final evaluation scope. No weights from the 43-update failed attempt or diagnostic iteration are used. The previous 4,227,072 complete transitions are recorded as separate attempted work, not credited to this chain.

The decision to proceed rests on source-supported dynamics, analytical tests, native negative controls and the existing end-to-end integration, while treating the remaining numerical and EMD limitations explicitly. It does not assert a complete native numerical validation or original sensor equivalence. The fixed-endpoint behavioral results follow below; no reward, tolerance, case selection or stopping threshold is changed to obtain a pass.

The new chain completed all 1000 iterations with actual exit 0. Its initialization checkpoint has SHA-256 `621c66558f77c492938ed9f78d6c45aa0830f0d2476981275c41e4668fba5306`; actual verification confirmed 17 identical model tensors, fresh Adam/counters, matching native/reconstruction limits and unchanged recorded sources. CPU identity verification also established unchanged consumed MuJoCo inputs and all 16 targets, so the complete original-model4000 baseline, including its invalid case, is reused. The final behavior improves several transfer metrics but fails the preselected no-inversion objective.


## Completed endpoint and interpretation

The new chain added exactly 98,304,000 transitions in 1895.341 seconds, with sampled GPU memory peak 6254 MiB. All 11 checkpoints and 1000 metric rows passed independent integrity review. Final checkpoint SHA-256 is `991f3c6a95f7e0498db2f1425b0ada7b2374e5495b0b17dd3a6339386552acfe`. Native and reconstruction joint limits remained 1000 rad/s. See [execution](../runs/umi_reconstructed_solver_relaxed_body_speed_adapt_seed0_4096/execution.json) and [final validation](../runs/umi_reconstructed_solver_relaxed_body_speed_adapt_seed0_4096/model1000-validation.json).

| Fixed evaluation | Official reference | Adaptation endpoint |
|---|---:|---:|
| Lab nominal 16, mean EE position / orientation | 9.940 mm / 0.027401 rad | 13.907 mm / 0.024286 rad |
| Lab nominal 16, inverted cases | 0 | 0 |
| Lab author protocol latest 500, mean EE position / orientation | 20.076 mm / 0.053725 rad | 30.682 mm / 0.071190 rad |
| Lab author protocol latest 500, timeout fraction | 97.8% | 89.2% |
| Lab author protocol latest 500, inverted episodes | 0 | 0 |
| MuJoCo common complete 15, mean EE position / orientation | 8.730 mm / 0.025409 rad | 256.539 mm / 0.309867 rad |
| MuJoCo all 16, complete / invalid / inverted | 16 / 0 / 0 | 16 / 0 / 2 |

The official Lab reference uses the same physical variant. Author-style means include early terminations; timeout fraction is not task success. Timeout-only position/orientation errors are 19.332 mm / 0.051965 rad for official and 21.272 mm / 0.053956 rad for adaptation. The nominal protocol runs full duration and preserves the deployed 20 ms observation-history cadence; the author protocol uses its original dense history.

Compared on the same 15 complete MuJoCo cases, initialization to adaptation improves mean position from 359.468 to 256.539 mm and orientation from 0.559554 to 0.309867 rad. Inversions decrease from five to two; the original additional invalid case12 becomes complete. Nevertheless cases3 and5 both invert at the first observed 0.98-second endpoint, and case5 position error worsens. The immediate behavioral objective is not met. The all-16 adaptation mean is 241.215 mm; it is not compared directly with the original 15-case aggregate.

The [MuJoCo report and figure](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_adapt_seed0_seed2027/README.md) retain all cases and failed initialization prefixes. [Lab nominal results](../outputs/isaac/umi-relaxed-body-speed-adapt-nominal16-seed2027/comparison-validation.json), [official author results](../outputs/isaac/umi-relaxed-body-speed-adapt-author-official-seed2027/summary.json) and [adaptation author results](../outputs/isaac/umi-relaxed-body-speed-adapt-author-model1000-seed2027/summary.json) preserve their separate protocols.

The [final export](../outputs/export/umi_reconstructed_solver_relaxed_body_speed_adapt_seed0_1000/README.md) passed exact actor reload parity and direct MuJoCo package consumption. Its separate sample(1,0) runs complete 17 seconds with 11.212 mm / 0.026745 rad and no inversion; this deployment check does not replace the failed fixed-16 evaluation.

The [integration-step diagnostic](../outputs/mujoco/integration-step-convergence-seed2027/README.md) completed eight new CPU rollouts at 2.5 and 1.25 ms on the two failed cases, preserving 5 ms PD refresh and 20 ms policy/history updates. All runs completed. Official remains stable; adaptation case3 inversion is present/absent/present at 5/2.5/1.25 ms, while case5 inverts at every step size. Target and preview indices match the baseline exactly. These nonmonotonic outcomes do not establish convergence or remove the primary failure. No further integration variants are planned.


## Next bounded experiment: fresh initialization under the corrected variant

Independent readout of the existing adaptation log found an early gain followed by a plateau. From iterations 601–800 to 801–1000, training EE worsens from 51.18 to 52.88 mm, orientation from 0.13381 to 0.13500 rad and mean EMD reward from −0.25271 to −0.25897. The optimizer remains active with finite updates and no variance collapse. These observations do not justify assuming another 3000 adaptation iterations will repair transfer.

The next experiment instead trains from random initialization under the corrected body-speed-v3 and joint 1000 rad/s physical variant. At the time this experiment was selected, no completed fresh chain had tested this configuration. It tests whether useful transferable behavior can be learned without inheriting the original policy's constraint-dependent behavior; it does not claim that initialization is the unique cause of the previous failures.

- Output: `runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4096`.
- Original official saved recipe, with the already-tested top-level joint limit override 1000; seed 0, 4096×24, 64 PPO epochs×4 minibatches, 17-second task, eight previews, 132/261 inputs. Original EMD and PPO remain unchanged. Current reconstruction numerical limits remain explicit.
- No `--weights` or `--resume`. Fresh actor, critic, std, Adam, random state and curriculum. Fixed 4000 iterations = 393,216,000 transitions, saving every 100.
- Actual independent MuJoCo 3.13 at 0/500/4000: same fixed sample(16,2027), all 16 cases for 17 seconds, original 5 ms integration, with all invalid prefixes retained. No intermediate-checkpoint selection or case replacement.
- Final nominal Lab fixed 16 and author-style latest 500 completed episodes under the same physical variant. Existing official references may be reused only with unchanged consumed configuration and source inputs.
- Preselected behavioral objective: final no inversion or numerical failure in the fixed 16 MuJoCo cases and lower position/orientation error than random initialization, with support/contact diagnostics and the official comparison retained. Native behavior must also be reported; timeout is not task success. This is an engineering comparison on the supplied trajectory pool, not a holdout or original sensor-equivalence claim.
- A runtime failure ends this attempt with the real exception and available state retained. No automatic reward, guard, endpoint or evaluation changes. Further seeds remain conditional on the actual result.


The fresh chain actually launched at 2026-09-11 23:09:08 UTC. Its [execution plan](../runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4096/execution-plan.json) records the complete command. Initial checkpoint SHA-256 is `9aed896c8d108ff15836da6ae48c43eba1861a919834080e6e6c81e4ccfdfc8b`. An independent CPU reconstruction of the saved, non-consumed initializer RNG matches all 17 initial model tensors exactly; Adam and counters are fresh, and native/cache joint limits agree at 1000 rad/s. The full training execution has now completed; its behavioral results are recorded separately.


The preselected iteration 500 [MuJoCo evaluation](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_seed2027/README.md) completed all 16 cases for 17 seconds without numerical failure. Mean position/orientation errors decrease from 246.695 mm / 0.540687 rad at random initialization to 48.099 mm / 0.189841 rad, but cases 6 and 14 invert. Mean ground-supported feet is 3.6745 and zero-ground-support fraction 5.249%. This is real intermediate learning with remaining failures; checkpoint 500 is not selected as the endpoint and the fixed 4000 chain subsequently completed.


The fresh chain completed all 4000 updates with actual exit 0 and 393,216,000 transitions in 7175.405 seconds (119.590 minutes), with sampled GPU memory peak 6254 MiB. Final checkpoint SHA-256 is `7015d72ad770ed5c3358480505bd9174b7c473f84229bc029ac9856244385a78`. All 41 checkpoints and 4000 metric rows are finite and complete; final Adam has 1,024,000 steps per parameter. Native and reconstruction caches remain 1000 rad/s, and all 32 recorded source files are unchanged. See [run validation](../runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4096/run-validation.json), [final checkpoint](../runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4096/model4000-validation.json) and [actual execution](../runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4096/execution.json). Final CPU and GPU evaluations and export self-tests also completed. All 16 final MuJoCo cases finish without inversion or numerical failure, with mean EE position/orientation errors 10.169 mm / 0.024536 rad. Native nominal16 gives 12.910 mm / 0.025513 rad without inversion; author latest500 gives 21.455 mm / 0.052952 rad and 95.0% timeout completion. The [complete seed-0 result](umi-relaxed-scratch-result.md) retains the residual nonfoot contact and reference gaps. This activates the previously conditional [fixed seed-1 and seed-2 repetitions](umi-relaxed-seed-repetitions.md), without establishing formal acceptance or sensor equivalence.
