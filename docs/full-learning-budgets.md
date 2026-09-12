# Full learning budgets after bounded method integration

This is a prospective resource decision for the unfinished learning scope in
[the project plan](project-plan.md), not a record of completed training or an
instruction to launch additional jobs. The original RoboDuet 50000-update run
and its final fixed evaluation remain active work. The completed DeepWBC,
UniFP and Learning Force Control short runs established execution, recovery and
export consumers; none established learned control.

## Resolved source workloads

The table uses existing executable configuration and fixed upstream sources.
One transition is one policy step in one environment, not one physics substep.
Each proposed workload is one fresh seed-0 candidate with its original robot,
method, task, rollout length and full update endpoint. No seed repeats, reward
search, hardware work or automatic extension is included.

| Method | Original robot | Environments | Rollout | Updates | Policy transitions |
| --- | --- | ---: | ---: | ---: | ---: |
| UniFP | B2 + Z1 | 4096 | 24 | 60000 | 5,898,240,000 |
| DeepWBC `public_fresh` | Go1 + WidowX 250s | 5000 | 40 | 40000 | 8,000,000,000 |
| Learning Force Control | B1 + Z1 | 4000 | 48 | 100000 | 19,200,000,000 |

Together these candidates represent **33,098,240,000 transitions**. They are
proposed sequential runs after RoboDuet's final evaluation, not three additional
concurrent jobs. Their full-capacity memory use and completion times have not
been measured on this workstation.

UniFP's inherited [base configuration](../pawcerto/methods/unifp/base_robot_config.py)
sets 4096, 24 and 60000; its [method configuration](../pawcerto/methods/unifp/config.py)
sets `force_start_step=8000`. Source force injection becomes active strictly
when `global_steps > 192000`. A synthetic boundary probe is not trained force
control. The fresh candidate would use the corrected runtime already exercised
by the existing evaluation, rather than resume the older model 21 whose
training had the documented gripper-property coverage gap.

DeepWBC's [configuration](../pawcerto/methods/deepwbc/config.py) resolves the
published fresh recipe above. Its separate `paper` option has 10000 updates
and a changed regularization schedule, totaling 2,000,000,000 transitions.
That option is not the proposed published recipe and does not provide the
unimplemented full paper 6D task. The selected sensing remains the explicit
`reconstructed_sensor_wrench` approximation on official, unmodified Lab/PhysX;
its [measured limitations](experiments/deepwbc-reconstructed-runtime-20260912.md)
remain part of any learning result.

Learning Force Control's pinned released entrypoint at revision
`c760e1d74ad165d3c069d4f57ab5d066f6a41eb6` specifies 4000 environments,
48 rollout steps and a 100000-update training call. Its configured `binary`
force/position mode is randomly resampled; this entrypoint does not establish
a later force-training phase. The native source owner checked these values
in external `b1_gym/envs/b1/b1_plus_z1_config.py:104` and
`scripts/train.py:610`. This identifies the released code budget, not the
unverified actual duration of the paper's reported experiment. See
[the source integration](learning-force-control-lab.md).

CPU-resolved configuration values, arithmetic and exact source hashes are
retained in `outputs/full-learning-budget-20260913/source-budgets.json`.
No simulator or optimizer was started for this budget calculation.

## What the existing timings establish

The 16-environment UniFP run completed 20 iterations with mean collection time
1.35330 seconds and mean learning time 0.13857 seconds. These loop components
exclude setup and do not predict the 4096-environment workload.

Learning Force Control's existing 16-environment run used 145.695 seconds for
20 timed iterations, approximately 7.285 seconds per iteration. Its independent
resumed iteration used 7.465 seconds. The timing includes rollout, updates and
finite checks, but excludes initialization, checkpoint/export writing and exit.
At that observed 16-environment rate, 100000 iterations are arithmetically
about 8.43 days while producing only 1/250 of the released transition budget.
This is neither an equivalent recipe nor a credible estimate for 4000 environments.

The existing small runs do not establish full-size initialization, rollout,
optimizer and save peaks. A configured 12 GiB service limit is not a measured
peak. Available memory cannot be inferred simply by subtracting RoboDuet's
service usage from 32 GB: the desktop and other processes also consume memory.
No 4000/4096/5000-environment capacity or reliable full-run duration is claimed.

## Proposed next allocation and evidence

The proposed order is UniFP, DeepWBC, then Learning Force Control, one candidate
at a time after the original RoboDuet experiment closes. The table defines the
requested full workloads. Existing official dependencies and independent user
services remain the execution path; full-size resource use must be observed in
the actual chosen run. A real allocation failure is preserved and repaired at
its cause. A smaller batch or different sample budget requires a new explicit
decision rather than being silently substituted and called equivalent.

For each accepted candidate, retain its own initialized model and final model,
all failed attempts, the original method's stages, complete configuration,
checkpoint identity and actual physical input/output records. Compare final
and initial policies under matched evaluation conditions and report tracking,
force/velocity errors where applicable, resets, falls and numerical failures
together. Exported policies must drive the physical evaluation and match
independent CPU consumers on the actual saved inputs. These are intended
comparisons, not new claims that a current short run has already passed them.

UniFP's [evaluation entrypoint](unifp-evaluation.md) now provides explicit
evaluation-start phase selection for the initial/final force comparison:
blindly restoring each checkpoint's training counter would change external-force
enablement between models. Actual model-21 CPU preflights verified saved / zero /
strict-threshold-plus-one start counters without changing the checkpoint or
resolved configuration; no new physical rollout is claimed. Matching stage and seed does
not imply identical later force histories when policies reset at different
times. The original source sampling, ramps and clock progression must remain
visible in the evaluation record.

MLM and Multi-critic implementation-material decisions, ReLIC full training
and its external-source license boundary, and AS2/RoboDuet transfer remain
separate unfinished items. These three proposed workloads do not close or
replace those parts of the full framework objective.
