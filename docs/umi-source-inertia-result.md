# UMI source-inertia correction and split-trained seed 0

The new seed-0 policy completed its fixed 4000-update training budget with
exit 0. **The UMI research reproduction stage is complete**, with tracking
comparable to the old baseline, measured cross-engine behavior and a verified
CPU export. The small mean-tracking difference is not a blocker for the
research preview. This is a practical research assessment, not a statistical
equivalence result from multiple new training seeds.

The frozen-model test attempted all 15 trajectories and completed 14. Test ID
1 inverted, struck its head on the ground and triggered a numerical error.
This is a known trajectory-level limitation, retained alongside the successful
cases. Research publication is proceeding with these boundaries disclosed;
universal stability is not claimed. No additional training or test-driven
tuning was performed to change the recorded result.

## What changed and why

The nonzero-state ground/free diagnosis identified an actual error in the
previously imported USD: principal inertia axes did not represent the source
URDF tensors. The converter now derives principal moments/axes from source
COM-frame tensors and writes them through official USD APIs. The new asset
passed a 25-body tensor check and actual native mass-matrix/free-response
comparison. The largest joint-velocity discrepancy across the three excited
free-state points and repeats fell below 4.11e-5 rad/s. Contact-response
residuals remain; these checks do not by themselves establish policy behavior.
See [the complete diagnosis](umi-contact-next-step.md).

One fresh seed uses that corrected asset and the fixed 71/15/15 trajectory
split. Body-speed-v3 solver-force reconstruction, the explicit 1000 rad/s
joint-speed limit, observation/control/reward semantics and PPO recipe remain
as documented. Isaac Lab and PhysX are unmodified. The old full-pool policies,
old USD and their records retain their original identities. Because both the
asset correction and training partition changed, performance is not a
single-variable inertia ablation.

## Training and evaluation identity

- Random seed 0, no weights initialization or resume; 4096 environments by
  24 rollout steps by 4000 updates: **393,216,000 transitions**.
- All 4000 metric rows are ordered and finite, and all final model parameters
  are finite. Training exited 0 without restart or recipe changes.
- [Execution record](../runs/umi_source_inertia_train71_seed0_4096/execution.json),
  [training verification](../runs/umi_source_inertia_train71_seed0_4096/training-verification.json)
  and the original launcher-source snapshot preserve actual execution identity.
- [Final checkpoint](../runs/umi_source_inertia_train71_seed0_4096/model_4000.pt):
  SHA256 `b0d55b6f4e509b7299e8daf8082637c7f08b914b545f64885657202cf9c018bf`.
- Corrected USD SHA256:
  `23b790d54ab47390a4ca002250707440a8d96c0c59b73f1fbf860e764b856210`.
- The [fixed manifest](../configs/umi_tossing_split_seed2027.json) contains
  71 train, 15 validation and 15 test IDs. Source recording IDs are unavailable;
  content grouping does not establish recording-level independence.

The original 16-case seed-2027 benchmark includes test IDs 8 and 36. Therefore
validation and author evaluation use only the 15 validation IDs first; the
original iteration-4000 candidate is then frozen before fixed16 and the full
15-case test partition. Validation failure does not cancel that final test,
change the checkpoint, or authorize another training run. Test observations
are not used for tuning. Benchmark/test overlap is retained, not counted as
additional independent unseen trajectories.

## MuJoCo validation: all 15 trajectories, full 17 seconds each

| Policy | Position mean (mm) | Orientation mean (rad) | Inverted cases |
|---|---:|---:|---:|
| Random initializer | 232.751 | 0.548997 | 0 |
| Iteration 500 | 18.725 | 0.052840 | 0 |
| Fixed iteration 4000 | 10.430 | 0.027874 | 0 |
| Old seed 0 (full-pool training) | 8.259 | 0.025055 | 0 |

All four rows completed every case. The new candidate demonstrates learning
relative to random initialization but its validation means are worse than the
old seed-0 reference. The old policy trained on all 101 trajectories, including
these validation IDs, so it is a performance reference with a different data
condition, not a fair held-out generalization comparator. The seed-matched
fixed16 benchmark remains a separate acceptance comparison.

At iteration 4000, mean ground-supported feet is 3.8691 and zero-ground-support
occupancy is 0.06282% of the 20 ms endpoints. Calf/nonfoot support is not classified
as failure solely from contact count or load share.

The 5 ms records found 10 head-contact samples in validation trajectory 40
(50 ms occupied samples, spanning 10.280–10.380 s; peak 15.476 N). The 5 ms
body-ground summary contains no head contact above 1 N; the saved 20 ms actual
collider pairs identify `Head_upper` against the `link6` gripper finger. In the
10.20–10.48 s saved-state window, root-up dot is 0.9825–0.9955 and the local EE
error reaches 94.18 mm. This is retained self-contact and local tracking error,
not evidence of the previous 552 N falling head-ground event. Do not describe
these samples as either zero head contact or a continuously occupied 100 ms
head-ground impact.

[Validation summary](../outputs/umi-source-inertia-train71-20260912/validation-model4000/summary.json),
[trajectory 40](../outputs/umi-source-inertia-train71-20260912/validation-model4000/trajectory_040.json)
and [its 5 ms summary](../outputs/umi-source-inertia-train71-20260912/validation-model4000/trajectory_040-substeps.json)
retain the detailed evidence. Every final validation case has 3400 existing
physics-substep samples. The callback adds no physics solve; normal 20 ms
endpoint reporting remains explicitly separate.

## Fixed16 old/new comparison

Both policies use the original seed-2027 target draw and complete all 16 cases
for 17 seconds in each engine. Native old/new evaluations both explicitly use
the corrected USD; MuJoCo uses its unchanged source-derived model.

| Engine / policy | Position mean (mm) | Orientation mean (rad) | Inverted cases |
|---|---:|---:|---:|
| MuJoCo / old seed 0 | 10.169 | 0.024536 | 0 |
| MuJoCo / new seed 0, train71 | 11.818 | 0.025356 | 0 |
| Native Lab / old seed 0, corrected USD | 12.012 | 0.026563 | 0 |
| Native Lab / new seed 0, corrected USD | 10.987 | 0.023062 | 0 |

MuJoCo position error increases by 1.648 mm and orientation error by
0.000820 rad (about 0.047 degrees), while native tracking improves. The current
research milestone accepts these small mean differences. The old policy also
trained on all 101 trajectories while the new one used only 71; this is not a
controlled comparison of data efficiency or inertia correction. Neither
MuJoCo policy has head-ground contact in these 16 cases; the new one retains a
self-contact case. Native 5 ms ContactSensor readout has no head-body net normal
force sample above 1 N in either policy. This native signal is not a
collider-pair-resolved total-force measurement. Ground-supported feet average
3.9074 old versus 3.8839 new; this difference is not itself failure.

For the new native policy, summary off/on reproduces all nine recorded fields
exactly across 849 endpoints and 16 cases. Each case has 3400 existing physics
substeps. Fixed target NPZ identities and corrected-asset identities match.
See [native verification](../outputs/umi-source-inertia-train71-20260912/native-benchmark-paired-verification.json)
and the [old](../outputs/umi-source-inertia-train71-20260912/benchmark16-old/summary.json)
and [new MuJoCo summaries](../outputs/umi-source-inertia-train71-20260912/benchmark16-new/summary.json).

The nominal16 draw contains only 15 unique trajectory IDs because ID 62 occurs
twice; IDs 8 and 36 also occur in the test partition. Do not add these case
counts to claim 31 independent unseen trajectories.

## Native author protocol, validation pool only

Both old/new policies use the same corrected USD, 15 validation IDs and seed
2027. Each finishes 502 episodes, with the prescribed latest 500 retained.

| Policy | Position mean (mm) | Orientation mean (rad) | Timeouts / 500 | Inverted episodes |
|---|---:|---:|---:|---:|
| Old seed 0 | 21.551 | 0.052842 | 486 (97.2%) | 0 |
| New seed 0, train71 | 21.690 | 0.054221 | 485 (97.0%) | 0 |

All early-terminated prefixes remain in the means. Timeout fraction measures
survival, not task success. The same pool and random seed do not establish
one-to-one pairing of individual episodes. See [author readout](../outputs/umi-source-inertia-train71-20260912/native-author-paired-readout.json).

## Frozen-model test: 14 of 15 complete, one instability

The original iteration-4000 model was [frozen](../outputs/umi-source-inertia-train71-20260912/frozen-candidate.json)
after validation and before the final benchmark/test. Every one of the 15 test
IDs was attempted once, without checkpoint selection or recipe changes. The
wrapper exits 2 and marks the summary incomplete.

Test ID 1 first has ground-to-head contact at 0.630 s. Its 5 ms readout contains
31 head-ground samples (155 ms occupied samples), with Head_upper peak
1679.208 N and Head_lower peak 1098.400 N. The root-up dot reaches -0.99984 in
the 5 ms record: this is an actual inversion and uncontrolled head impact.
The same trajectory also contains self-contact; these are not two failed cases.

At 10.525 s, MuJoCo reports `mjWARN_BADQACC` for DOF 21 and internally restarts
its clock to 0.005 s. The existing guard catches this before another trace
callback; the last valid saved endpoint is 10.520 s. The failed prefix is
retained as such and is excluded from full-trajectory averages. Its extreme
joint speed and saturated commands are failure observations, not reliable
post-failure dynamics or evidence for a particular training fix.

The other 14 trajectories complete 17 seconds without inversion or head
contact. Their subset means are 10.263 mm and 0.027938 rad. **There is no
complete 15-trajectory test mean**, and these subset values must not be
reported as the full-test performance. See [test summary](../outputs/umi-source-inertia-train71-20260912/test-model4000/summary.json),
[failed trajectory](../outputs/umi-source-inertia-train71-20260912/test-model4000/trajectory_001.json)
and [its substep summary](../outputs/umi-source-inertia-train71-20260912/test-model4000/trajectory_001-substeps.json).

## Decision and delivery checks

The source-inertia conversion defect is corrected and validated, and the
new split-trained policy learns substantially relative to initialization.
The research milestone now accepts tracking close to the old baseline and
publication with explicit limitations. The earlier strict requirement to
preserve both means exactly and complete every test case had classified this
candidate as not qualified; that historical decision remains in the original
local result snapshots. The current research-preview decision does not change
any measurement, remove the test failure, or establish universal stability.

The framework continues from this completed UMI reproduction stage. Further
UMI robustness work and full RoboDuet learning remain separate experiments;
no automatic extra seed, budget extension or test-driven tuning is included
in publishing this result. Installation, method integration and KMA release
evidence retain their own scopes.

The [comparison video](../outputs/umi-source-inertia-train71-20260912/video/comparison_fixed4000_case06.mp4)
shows old seed 0, historical seed 1 and new train71 seed 0 on the same preselected
case 6. It reconstructs saved states on CPU, with no inference or physics
integration. Target arrays match exactly; input hashes are unchanged. The full
426-frame video decodes successfully and four encoded frames were inspected.
Its 17.04-second container covers the 0–17-second task including both endpoints.
This illustrative case does not substitute for the all-trajectory test result.

Relevant CPU checks pass: 40 tests and 24 subtests, plus three upstream policy
comparisons in the existing reference environment. Independent evidence review
found no material data-identity or aggregation error and confirmed the recorded
trajectory failure. See [final checks](../outputs/implementation-validation-20260912/final-source-inertia-validation.json).

The exported actor has zero-error reload parity and completed an actual
17-second CPU MuJoCo execution on validation ID 7. All seven saved arrays match
the full-checkpoint execution exactly. The actor-only loader retains unknown
training identity; explicit artifact provenance links the export to the frozen
checkpoint without relabeling arbitrary actor-only weights as held out. See
[export consumption](../outputs/umi-source-inertia-train71-20260912/export-consumer-17s/verification.json).
These checks establish package execution; the known trajectory instability
remains disclosed.
