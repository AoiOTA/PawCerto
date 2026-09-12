# AS2 model 500: fixed16 MuJoCo checkpoint diagnostic

The actual 4096-environment seed-0 checkpoint contains iteration 500 and 49,152,000 transitions. Its SHA-256 is `7d9ec6c6800a011bc2ad42dabb59c2fe4d6b865cd3217d427094056687d1b70f`. The original MuJoCo model, config, controller and target archive were retained.

Evaluator exited 2; 3/16 full 17-second cases. All sampled targets match the frozen seed-2027 input. Invalid prefixes are preserved, not counted as completed episodes.

The only complete cases are **1, 6 and 9, all inverted**. On this same three-case subset, model0 to model500 mean EE error changes from **0.563450 m / 1.642654 rad** to **5.391835 m / 2.069168 rad**. Mean ground-supported feet falls from 3.941107 to 0.298783; zero-ground-support endpoints rise from 1.0601% to 78.7986%. All three have sampled nonfoot contacts above 1 N. These are **3/16 subset values**, never an overall fixed16 tracking score.

Thirteen other cases terminate with `mjWARN_BADQACC`; their original JSON records retain engine warning details and the time rollback detected by the runtime. Finite saved prefixes do not make these episodes valid.

| Case | Status | Last saved time (s) | Model0 position / orientation | Candidate position / orientation | Candidate inverted in prefix |
|---|---|---:|---:|---:|---|
| 0 | invalid | 1.04 | 0.38708 m / 1.59436 rad | 0.45943 m / 2.16444 rad | True |
| 1 | complete | 17.00 | 0.55332 m / 1.63559 rad | 7.99009 m / 2.06013 rad | True |
| 2 | invalid | 0.76 | 0.46935 m / 1.62956 rad | 0.44396 m / 2.03660 rad | False |
| 3 | invalid | 0.56 | 0.32961 m / 1.54337 rad | 0.47395 m / 1.72440 rad | True |
| 4 | invalid | 1.66 | 0.50391 m / 1.59989 rad | 0.67215 m / 2.22842 rad | True |
| 5 | invalid | 1.52 | 0.43781 m / 1.61995 rad | 8.38040 m / 2.07446 rad | True |
| 6 | complete | 17.00 | 0.61019 m / 1.64775 rad | 1.58739 m / 2.20441 rad | True |
| 7 | invalid | 0.46 | 0.38665 m / 1.54316 rad | 37.01755 m / 1.70261 rad | False |
| 8 | invalid | 0.70 | 0.48163 m / 1.64597 rad | 16.24172 m / 2.02153 rad | True |
| 9 | complete | 17.00 | 0.52684 m / 1.64462 rad | 6.59802 m / 1.94296 rad | True |
| 10 | invalid | 0.54 | 0.38466 m / 1.55771 rad | 0.37832 m / 1.74602 rad | True |
| 11 | invalid | 1.38 | 0.43546 m / 1.62239 rad | 0.56931 m / 2.30928 rad | True |
| 12 | invalid | 2.48 | 0.49383 m / 1.66884 rad | 1.75331 m / 2.01249 rad | True |
| 13 | invalid | 0.66 | 0.39604 m / 1.61175 rad | 0.31802 m / 2.06322 rad | False |
| 14 | invalid | 0.80 | 0.43103 m / 1.61222 rad | 0.48788 m / 1.99597 rad | True |
| 15 | invalid | 1.00 | 0.40616 m / 1.61345 rad | 5.87337 m / 2.20365 rad | True |

Each row compares the same elapsed prefix; row durations differ. BADQACC details and root/arm motion remain in model0-paired-comparison.json and case JSON/NPZ. Contact metrics are sampled every 20 ms. No reward, physics, policy or training budget changed for this evaluation.

## Existing training-side evidence

The saved iteration-500 training log reports EE error 0.0611876 m / 0.4501084 rad, mean root height 0.250109 m and mean net-supported feet 2.977325. Its `terminated` value is 5.666667: the mean count of `done` environments per policy step across 24 steps, or 136 reset events over that rollout. It includes configured contact termination and timeouts/safe-bound exits; it is not a fall count or episode success rate. Neighboring iterations 495/505 report 6.958333/6.166667 such events per step. The training log does not separate inversions or termination causes, so it cannot establish their absence. These are stochastic training-rollout aggregates collected before the checkpoint's PPO update, not the fixed-policy nominal MuJoCo protocol.

The discrepancy is observed; this evaluation alone does not locate a training bug, identify an engine-transfer cause or establish final learning failure. The authorized single candidate continues to update 4000 with unchanged task, rewards and physics. Model4000 and final paired Lab evaluation remain pending.

Artifacts: `runs/as2_umi_seed0_4096_4000_fixed16_500/summary.json`, `fixed16-audit.json`, `model0-paired-comparison.json`, and all `case_*.json`/`case_*.npz`. The paired JSON retains root motion, arm ranges, support and exact failure details for each case. Original evaluator exit status was 2; audit success only verifies preserved identities and arrays.
