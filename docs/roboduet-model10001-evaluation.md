# RoboDuet model 10001: fixed protocol milestone

The original 50000-update experiment reached checkpoint 10001 and its prescribed
fixed evaluation completed on 2026-09-13. All nine cases terminated for low
body height after 0.10–0.40 seconds of policy control; none completed the requested
20 seconds. The process exited 0 because the prescribed evaluation attempts
completed. This is negative physical evidence, not learned coordination.

The checkpoint is the exact Stage 1 boundary: `next_rollout_stage=2`,
`arm_update_count=0`. Its arm weights remain identical to initialization and
both arm optimizer states are empty, as checked by the training operator.
Subsequent training has entered Stage 2 and remains under the original
50000-update budget. This milestone does not authorize another recipe, seed
or shorter substitute for that endpoint.

## Fixed inputs and retained artifacts

The [existing protocol](roboduet-evaluation.md) and
[`configs/roboduet_evaluation.json`](../configs/roboduet_evaluation.json) were
unchanged. Every checkpoint is evaluated with the arm physically released,
using the original `official_play` arm → plan → dog graph and all five exported
modules. This playback condition is not the frozen-arm Stage 1 training task.

Checkpoint SHA-256:
`78e2068829c3dc1079ee9e69eca1a2c8a476f9c1de3bbfe35738edd4a5b5c1cb`.
Protocol SHA-256:
`bdb7a7c2de34586a24a03e4d68dd1c885c05a323cc74fb887908f685b69c99df`.
The original records are under
`outputs/roboduet-fixed-evaluation-20260912/model_10001/`, including
`summary.json`, the five-module `policy/` package, and each seed's `states.pt`,
`substeps.pt` and `policy_inputs_outputs.pt`. Model 0 and 1600 records remain
alongside it. All nine post-reset root states and joint positions matched
across the three checkpoints exactly; each retained its separate zero-action
initialization record.

## Paired results

Survival below excludes the separately recorded 0.02-second zero-action
initialization. The three-way common prefix is the shortest observed policy
window for that seed/case, including its terminal policy step.

| Seed / case | Model 0 (s) | Model 1600 (s) | Model 10001 (s) | Common prefix (s) |
|---|---:|---:|---:|---:|
| 2027 / standing | 0.14 | 0.16 | 0.10 | 0.10 |
| 2027 / locomotion | 20.00 | 0.24 | 0.14 | 0.14 |
| 2027 / simultaneous | 0.18 | 0.24 | 0.18 | 0.18 |
| 2028 / standing | 0.18 | 0.24 | 0.40 | 0.18 |
| 2028 / locomotion | 0.18 | 0.24 | 0.16 | 0.16 |
| 2028 / simultaneous | 0.18 | 0.30 | 0.16 | 0.16 |
| 2029 / standing | 0.16 | 0.24 | 0.16 | 0.16 |
| 2029 / locomotion | 20.00 | 0.30 | 0.14 | 0.14 |
| 2029 / simultaneous | 0.16 | 0.26 | 0.20 | 0.16 |

The following first averages each metric within that seed/case's common
prefix, then gives equal weight to the nine cases. These values describe a
short shared transient; they do not compare complete 20-second behavior.

| Common-prefix mean error | Model 0 | Model 1600 | Model 10001 |
|---|---:|---:|---:|
| Planar linear velocity (m/s) | 0.410699 | 0.611816 | 0.402679 |
| Yaw rate (rad/s) | 0.190193 | 0.645943 | 1.175843 |
| EE position (m) | 0.415029 | 0.430511 | 0.464000 |
| EE orientation (rad) | 0.139595 | 0.148073 | 0.252213 |

Relative to model 0, model 10001 improves linear velocity in five of nine
cases, yaw rate in none, EE position in none and orientation in one. Relative
to model 1600, those counts are nine, one, zero and one. Its smaller aggregate
linear-velocity error therefore does not establish better coordination.

## Physical failure boundary

All model 10001 initialization records were valid. Every subsequent case
ended through the original `body_height < 0.28 m` condition, with terminal
height 0.2672–0.2790 m. There were no timeouts, reverse terminations or nonfinite
states. The minimum world-up dot product was 0.75795, so these records do not
show inversion. The first substep below the height threshold occurred at
absolute physical time 0.115–0.410 seconds including initialization; the
control-step termination timing is consistent with the source's four-substep
checking interval.

Every case reached the 23.7 Nm sent leg-effort limit. The maximum saved joint
speed was 49.20 rad/s at the right-rear calf in seed 2027's locomotion case,
absolute time 0.135 seconds. Adjacent saved substeps had exactly continuous
joint positions, and all saved substep tensors were finite. These facts expose
the transient's severity without establishing a controller or integration bug.

Three cases already had nonfoot net contact above 1 N in the initialization's
first 5 ms. These net forces can include self-contact; they do not by themselves
identify ground impact or the cause of later termination. No model 10001 case
reached the first 2.5-second target switch. The evidence covers the first-command
transient, not the complete dynamic EE sequence.

An independent read-only review computed the paired comparisons and inspected
the saved 5 ms records. It found no clear defect invalidating the evaluation
and did not rerun physics or the training checkpoint audit. Continue to the
original 50000 endpoint and apply the unchanged protocol there, interpreting
survival, locomotion and EE tracking together. The stage-boundary result alone
does not isolate a causal contribution from Stage 1 learning.

The common-prefix arithmetic is retained as an executable source tool:

```bash
python scripts/report_roboduet_fixed_comparison.py \
  --evaluations outputs/roboduet-fixed-evaluation-20260912 \
  --iterations 0 1600 10001 \
  --output outputs/roboduet-fixed-evaluation-20260912/model10001-comparison.json
```

It uses the recorded control period, verifies identical protocol hashes and
case order, retains initialization deltas and source file identities, and
reports any empty shared prefixes separately. Its actual CPU result matches
the independent review's rounded values above; no simulation is repeated.
