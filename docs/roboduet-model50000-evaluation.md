# RoboDuet model 50000: final fixed evaluation

The authorized training and fixed evaluation are complete. Physical learning acceptance is negative: all nine final cases terminate for body height within 0.10–0.28 s; none completes the 20 s protocol. The endpoint is not evidence of learned whole-body cooperation.

Training has exactly 50,000 unique, finite metrics rows (iterations 0–49,999), 4,915,200,000 transitions and 39,999 arm PPO updates. Final weights and all four optimizer states are finite, with CPU uint8 RNG state. Model SHA256: `78e99f1535600fa37ecd075c9674d003a9317713988fb06a067e5f2bc9b6b297`.

The original service journal records normal completion at 2026-09-14 05:01:59 +08:00. Both transient units had already unloaded at observation, so their later default ExecMainStatus=0 is not an independently retained numeric process exit code. Journal completion without a process failure, complete final artifacts, and the CPU audits are the available evidence. Original training attempts and causes remain preserved.

Final evaluation ran 05:02:36–05:02:52 through the unchanged official Isaac Lab path, same nominal plane, fixed three seeds × three cases, official_play five-module exports, 5 ms physics and 20 ms policy steps. Protocol bytes and RoboDuet source closure match the 10,001 evaluation. Checkpoint asset hash matches the actual consumed USD. Five-module reloads and re-evaluation of every recorded actual policy input against the checkpoint match exactly (maximum absolute output difference 0).

| Seed | Standing EE survival s | Locomotion fixed EE survival s | Simultaneous survival s |
|---|---:|---:|---:|
|2027|0.10|0.14|0.20|
|2028|0.16|0.12|0.16|
|2029|0.28|0.12|0.28|

All nine observed prefixes are finite; all nine have body-height termination, no initialization-invalid cases or timeouts. Two cases also reach negative up-dot (seed 2029 standing and simultaneous); the stop reason remains height. Mean survival is 0.1733 s. For comparison, model 0 completes 2/9 and model 10,001 completes 0/9; these are the existing same-protocol results, without rerunning or retuning.

Final per-case EE position mean errors range 0.3873–0.5878 m over each case's own observed prefix (equal-case mean 0.4607 m); these are not 20 s tracking errors or a matched-prefix improvement claim. Mean clipped-action fractions per case are 89.8–96.3%; leg sent effort reaches 23.7 Nm in every case. Actual joint-speed peaks are 30.13–34.55 rad/s. Saved sent effort excludes implicit spring/damper and is not measured solver joint force; body contact vectors are net body forces and do not identify contact pairs. Head-contact summaries remain unavailable rather than zero.

Local evidence under `outputs/roboduet-fixed-evaluation-20260912/`: `model_50000/` contains all three seed summaries, pre-autoreset states, policy input/output traces, 5 ms substeps, policy modules, and copied protocol/config. `model50000-readout.json`, `analyze_model50000.py`, launch source copies/hashes, command, log and service journal retain the consumed path. Training endpoint audits and journal are in `outputs/roboduet-learning-seed0-20260912/`. These local runtime artifacts are not bundled in the source checkout; this page records their measured result.

Stop boundary reached: no additional optimizer updates, tuning, replacement protocol or new AS2 learning was started. No production source was edited for this endpoint.

Protocol and previous comparisons: [fixed evaluation](roboduet-evaluation.md), [model 0/1600](roboduet-evaluation-result.md), [model 10001](roboduet-model10001-evaluation.md). The separate AS2 general-EE learning budget does not extend this RoboDuet candidate.

## Bounded CPU diagnosis of the first saturated actions

A subsequent read-only examination of the nine saved first policy inputs found
no actionable evaluation-port contract mismatch. The 15 relevant source files
match the recorded evaluation launch closure. Initial arm observations have
maximum absolute value 0.5 and dog observations at most 1, yet the final policy's
per-case maximum physical arm output is 55.76–222.68 and dog output is
37.74–164.71. At the first action, 11–17 of 18 dimensions saturate at ±10.
All nine first non-initialization physical substeps contain exactly the saved
policy output clamped to that range. The earliest observed large action is
therefore already in the policy mean, rather than an extra downstream scale
or permutation.

The inherited training/playback difference in the two arm guidance dimensions
was examined on the same saved inputs. Replacing raw guidance with its training
tanh form leaves the six physical arm outputs unchanged and changes dog outputs
by at most 0.0003356934, with unchanged saturation counts: the existing plan
already clips both guidance forms to nearly identical limits. Removing the
duplicate initial arm history frame, which is also present in upstream playback,
instead gives maximum physical arm outputs of 290.49–301.90; all six arm
dimensions and 6–10 dog dimensions still saturate. Neither comparison supports
changing that contract to repair this observed failure.

This uses the recorded `policy_inputs_outputs.pt` and `substeps.pt` for all three
seeds, the existing `official_play`/`training_mean` implementations and the
fixed upstream playback/training sources. No new simulator steps, optimizer
updates or production edits were made. These CPU comparisons identify where
the first failure manifests; they do not establish why training produced the
large means, prove complete physics equivalence, or substitute for a replay of
alternative stochastic trajectories. The negative endpoint and stop boundary
remain unchanged.

## Location of output growth in the existing checkpoints

The same nine final first-step observation/history inputs were also replayed on
four existing checkpoints, on CPU without generating new guidance or physical
states. The severe mean growth lies between checkpoints 10001 and 50000, after
arm optimization begins in Stage2; these four samples do not identify its first
iteration within that interval.

| Checkpoint | Arm updates | Dog max absolute mean | Dog saturated dimensions /108 | Arm physical max absolute mean | Arm saturated dimensions /54 | Dog / arm mean std |
|---|---:|---:|---:|---:|---:|---:|
|0|0|0.1168|0|0.1231|0|1.000 /0.100|
|1600|0|1.6837|0|0.1231|0|1.394 /0.100|
|10001|0|2.8371|0|0.1231|0|2.525 /0.100|
|50000|39999|164.7111|88|222.6786|47|8.093 /73.211|

The final arm std parameters range 65.93–89.62. These are checkpoint parameters,
not reconstructed log values. The shared inputs contain final Stage2 guidance;
the earlier checkpoints' rows are therefore fixed-input diagnostics, not their
own closed-loop or Stage1 behavior results.

Existing training metrics show earlier degradation: during iterations 1600–10000,
dog learning rate is always 1e-5 and mean dog reward is 1.2175e-9. Mean done_count
is 1088.73 per update in that window, 9618.01 in the first 999 Stage2 updates and
12980.05 in the final 1000 updates. Final-window dog/arm mean rewards are about
5.23e-43/9.22e-45; both are zero in the final row. Dog learning rate remains 1e-5
through Stage2; the saved arm rate spans 1e-5–0.01. These are end-of-update rates,
not all minibatch rates. Done_count sums termination events across 4096 environments
and 24 rollout steps, so it is neither unique failed environments nor completed
episode length.

The metrics contain rewards, rates and losses but no actual KL, entropy, std
time series or completed-episode-length series. The second logged loss is the
surrogate loss, not the combined value/entropy objective. A learning-rate floor
does not establish a KL value. The observations locate deterioration and output
growth; they do not establish entropy, reward scale, physics or an implementation
defect as its cause. No algorithm change or further RoboDuet run follows from
this bounded diagnosis.
