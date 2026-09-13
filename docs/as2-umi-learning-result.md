# AS2/Piper-H: completed 4000-update candidate

The authorized fresh seed-0 run completed **4096 environments × 24 transitions × 4000 updates = 393,216,000 transitions**, and the training service exited 0. This completes the prescribed experiment with a negative stable-control/transfer outcome: Lab tracking improves but 7/16 cases invert; MuJoCo transfer fails numerically. The final MuJoCo fixed16 evaluation has **0/16 complete cases: all 16 fail numerically**. No additional seed, reward change, physics change or training extension was introduced.

## MuJoCo fixed16 outcome

All checkpoints use the same original tossing-pool draw, seed 2027, and requested 17 seconds per case. Saved targets were checked element-for-element against the frozen array; config, MuJoCo model and runtime hashes remained unchanged.

| Checkpoint | Full 17-second cases | Numerical failures | Complete-case inversions | Full-episode tracking interpretation |
|---|---:|---:|---:|---|
| 0 | 16/16 | 0 | 0/16 | Mean 0.5105255 m / 1.6324972 rad |
| 500 | 3/16 | 13 | 3/3 | Complete subset only: 5.391835 m / 2.069168 rad; not an overall fixed16 mean |
| 4000 | 0/16 | 16 | No complete cases | No complete-episode mean exists |

Model0 has mean ground-supported feet 3.941696 and no sampled nonfoot contact above 1 N. Model500's three complete cases have mean ground support 0.298783 and all have sampled nonfoot contacts; [the intermediate report](as2-umi-model500-evaluation.md) preserves its failures and matched-subset comparison. Model4000's BADQACC failures leave only 0.18–1.82 seconds of saved policy endpoints. These finite prefixes can already contain numerical explosion and are not successful short episodes.

The following model0/model4000 values compare the exact same saved time prefix within each case. Durations differ between cases; these values must not be pooled into a full-episode result.

| Case | Status | Last saved time (s) | Model0 position / orientation | Candidate position / orientation | Candidate inverted in prefix |
|---|---|---:|---:|---:|---|
| 0 | invalid | 0.66 | 0.39203 m / 1.60523 rad | 4.30575 m / 2.17709 rad | True |
| 1 | invalid | 0.18 | 0.52218 m / 1.53016 rad | 0.52091 m / 1.24978 rad | False |
| 2 | invalid | 0.30 | 0.46899 m / 1.54797 rad | 0.62850 m / 1.86461 rad | False |
| 3 | invalid | 0.20 | 0.47576 m / 1.54148 rad | 67.18844 m / 1.67393 rad | True |
| 4 | invalid | 0.34 | 0.42446 m / 1.52943 rad | 0.57895 m / 1.90455 rad | False |
| 5 | invalid | 0.22 | 0.43670 m / 1.53793 rad | 0.58872 m / 1.78045 rad | False |
| 6 | invalid | 1.04 | 0.48192 m / 1.63751 rad | 1.58380 m / 1.85085 rad | True |
| 7 | invalid | 1.82 | 0.58155 m / 1.63853 rad | 9.84297 m / 2.12329 rad | True |
| 8 | invalid | 0.22 | 0.58224 m / 1.58242 rad | 0.74267 m / 1.52134 rad | False |
| 9 | invalid | 1.28 | 0.50871 m / 1.66748 rad | 1.88764 m / 2.35443 rad | True |
| 10 | invalid | 0.54 | 0.38466 m / 1.55771 rad | 0.87959 m / 1.98293 rad | True |
| 11 | invalid | 0.18 | 0.51312 m / 1.55561 rad | 0.55041 m / 1.32982 rad | True |
| 12 | invalid | 0.60 | 0.35949 m / 1.60792 rad | 1.79719 m / 2.20007 rad | True |
| 13 | invalid | 0.18 | 0.49447 m / 1.55432 rad | 0.58984 m / 1.55144 rad | True |
| 14 | invalid | 0.20 | 0.49695 m / 1.54357 rad | 0.52421 m / 1.36342 rad | True |
| 15 | invalid | 0.28 | 0.42359 m / 1.55251 rad | 0.84713 m / 1.65853 rad | True |


Artifacts under `runs/as2_umi_seed0_4096_4000_fixed16_{0,500,4000}` retain each case JSON/NPZ, summaries and `fixed16-audit.json`. The final directory also has `model0-paired-comparison.json` and `three-checkpoint-common-prefix.json`, containing per-case 0/500/4000 comparisons on their common elapsed prefix. Saved root paths and arm ranges describe motion, not coordinated control.

## Final Lab fixed-policy comparison

The prescribed paired Lab run exited 0. Both policies retain 849 finite policy endpoint rows per case through 17.00015 s, plus 3400 contact samples at 5 ms including the zero-action warmup. The frozen-target hash and asset identities match the training run. No learning or domain randomization was enabled in this evaluation.

| Measure | Model0 | Model4000 |
|---|---:|---:|
| Mean EE position error | 0.510769 m | 0.119318 m |
| Mean EE orientation error | 1.625607 rad | 0.363301 rad |
| Cases inverted in 5 ms samples | 0/16 | 7/16 |
| Minimum root up-dot at 5 ms | 0.963484 | -0.999796 |
| Minimum saved root height | 0.146879 m | 0.071268 m |
| Mean ground-supported feet at 5 ms | 3.951121 | 2.949283 |
| Zero-ground-support samples | 1.2059% | 9.0331% |

Final inverted cases are **1, 2, 7, 9, 11, 13 and 15**. Thus average Lab tracking improved while physical support and inversion outcomes worsened; running all periods is not successful control. Per-case tracking/support results are preserved in `runs/as2_umi_seed0_4096_4000_lab_fixed16-audit.json`, alongside the original `runs/as2_umi_seed0_4096_4000_lab_fixed16.json`.

Nonfoot/nonfinger body net-normal contact exceeds 1 N in 16/16 model0 cases and 15/16 model4000 cases, with peak body force 310.16 N and 7519.16 N respectively. These channels include self-contact and cannot identify collider pairs or external collisions. They also differ from MuJoCo's pair-classified 20 ms endpoint contact report; these counts are not directly comparable contact occupancies. At 20 ms endpoints, net-foot and ground-foot support differ in 270 final samples, versus zero in model0. Joint motion and velocity-limit response were not captured by this Lab summary, so it cannot demonstrate coordinated leg/arm behavior or measured speed clamping.

## Export and independent consumption

Export creation exited 0, producing the AS2 actor, config and joint order with measured actor error 0. The original verification command exited **1** after its original-checkpoint worker exited **2** with numerical failure. The same verifier's exported worker was then executed independently and also exited **2**. Both processes recorded the same BADQACC at 0.740 s and identical saved arrays. The original verifier failure is preserved; **package consumption did not pass**. Matching invalid prefixes establishes identity for that observed prefix, not a valid 17-second rollout.

Artifacts: `runs/as2_umi_seed0_4000_export/` and `runs/as2_umi_seed0_4000_export_verification/{verification.json,original.json,original.npz,exported.json,exported.npz,independent-consumer-failure-comparison.json}`. This single-case consumer draw is separate from the fixed16 batch.

## Interpretation and limits

Iteration4000's existing stochastic training aggregate reports EE error 0.0448586 m / 0.1674291 rad and 121 `done` events across 24 policy steps. These include configured contact/time-bound exits and have no inversion-specific breakdown; they are not the final fixed-policy evaluation. Favorable training aggregates do not negate the numerical failures above.

A known declared constraint difference remains: Lab configures source URDF joint velocity limits, including Piper's 5 rad/s limits, while this MuJoCo implementation does not enforce those velocity constraints. The model500 saved failure review observed early Piper speeds beyond those limits but did not isolate a cause; model0 also modestly exceeded a source limit. The final Lab command uses contact summary, not contact trace, and does not measure joint speed-limit response. This result therefore must not claim observed Lab velocity clamping or attribute the MuJoCo failure solely to this difference. No public binding, orientation, delay or torque-clipping bug was demonstrated by that bounded review.

The final checkpoint SHA-256 is `1b0f267d6dd7ef59f025a484d1be663ad149c032d4dfc43708a1b9ed1914d298`; the frozen target SHA-256 is `d118cdc555bcfa661a0afa566b2a838f1513416e580109efa93fcd48eb1337d5`. Training log/config/checkpoints are in `runs/as2_umi_seed0_4096_4000`; attempt logs, source/asset snapshots and the original OOM record are in `outputs/as2-umi-learning-execution-20260913`. The source-geometry repair was separately checked against the prior 3-second control trace with identical recorded arrays. This remains a supplied-training-pool research result, with nominal uncalibrated AS2/Piper model assumptions and no hardware or held-out acceptance.

The declared one-candidate stop boundary is reached. No further training or simulation was launched to repair these final outcomes. Paths under `runs/` and `outputs/` refer to local evidence and are not asserted to be bundled in a public source checkout.

后续有限诊断、CPU参数/伸展审查及面向AS2的预训练方案见 [下一步迁移方案](as2-umi-migration-next.md)；它不改变本次4000-update负面结果或授权新的训练。
