# PawCerto: an open reinforcement-learning framework for quadruped manipulation

The user-confirmed final goal is an open-source whole-body-control (WBC)
reinforcement-learning framework for quadruped robots with manipulators,
usable by the broader research community. UMI-on-Legs is the first complete
method integration and validation route, not the limit of the project.
Researchers should be able to obtain public inputs, install supported official
dependencies, reproduce measured baselines, and adapt methods to their robots
and tasks. Training, evaluation and policy deployment must have usable entry
points, documented assumptions and honest behavior evidence.

Build shared robot, task-input, simulation, evaluation and deployment components
when concrete method integrations need them. Preserve each method's observation,
reward and learning semantics. The multi-method route and the rule to extract
shared code after a second working method remain below; this goal does not
authorize premature universal abstractions or change the current experiment
budgets and stop boundaries. KMA remains separate development tooling, not a
dependency for researchers using PawCerto.

Approved on 2026-09-12. This is the current project direction, not a claim that
the remaining work has already passed. The initial UMI implementation, three
fixed 4000-iteration seeds, dual-engine evaluations, exports, and actual Conda
migration checks are complete; retain them as the baseline in
[the current result](umi-current-result.md).

## Current research milestone

The user clarified on 2026-09-12 that the small old/new tracking difference is
acceptable for continuing the project. UMI research reproduction is complete
with real training, fixed-policy dual-engine evaluation, verified policy
exports and transparent limitations. Publish a research preview and proceed
along the method-integration roadmap. A small mean-tracking change or one
disclosed test instability does not classify the entire integration as failed.

Keep every actual inversion, numerical error and early-termination stratum in
the evidence. The current 14/15 complete unseen test result is not universal
stability; its invalid prefix cannot enter a complete-test mean. Earlier local
snapshots retain the prior stricter acceptance decision as historical context.
Calf or task-appropriate nonfoot support is interpreted with tracking and motion,
not classified as failure solely from contact counts or load share.

Use official unmodified Isaac Lab/PhysX and the installed fixed Conda
environments. Keep body-speed-v3 and the explicit 1000 rad/s limit as the
current baseline; change them only when evidence supports the change. Exact
Lab/Gym numerical equality and a PhysX rebuild are not prerequisites.

## Deliverables and current execution state

The RoboDuet second-method entrypoints now complete actual short execution across
Stage 1, Stage 2 and an independent-process resume. All four optimizers and
saved RNG/curriculum/history state were checked; five exported modules in both
inference modes match saved real inputs exactly. This is integration evidence,
not effective learning or identical resumed physics trajectories. Full RoboDuet
training retains a separately chosen budget. See [training](roboduet-training.md)
and the [next complete learning experiment](roboduet-learning-plan.md).

Item 1's original 5 ms contact and warmup diagnosis is complete; the
initialization-only intervention did not remove the head collision and that
branch is closed. The subsequent nonzero-state ground/free comparison found an
actionable source-URDF/USD inertia-axis mismatch. Source-derived correction
passed 25-body tensor checks and native mass-matrix/free-response checks, without
modifying Isaac Lab or PhysX. The complete evidence is in
[the diagnosis](umi-contact-next-step.md).

Item 2 completed one justified candidate under its existing conditional
budget: corrected source-inertia asset, random seed 0, 4096 environments by
24 rollout steps by 4000 updates, body-speed-v3 and joint1000 retained. This run
also uses Item 3's 71 training IDs. Existing policies and old assets remain
unchanged. Final MuJoCo fixed16 tracking regressed, and test ID 1 inverted,
struck its head and triggered BADQACC; only 14 of 15 test cases completed. The
candidate is a research result with this disclosed trajectory limitation. No extra training or
test-driven recipe changes follow automatically. See [the final candidate
result](umi-source-inertia-result.md).

Item 3's grouping, ID selection and checkpoint provenance are implemented
(71 train / 15 validation / 15 test). Full-budget split training completed,
the original iteration-4000 candidate was frozen after validation, and every
test ID was attempted once. The failed test prefix is excluded from complete
trajectory means; all 15 IDs were attempted and the one instability is reported. Item 4's isolated
CPU and Lab online installations and runtime entry checks have completed,
including three short split-training updates across save/resume. Network
recovery and one unintended extra resume update are retained in
[the reproduction record](release-reproduction.md); this is not an uninterrupted
installation or learning-success claim. The original scopes below remain the acceptance
contract, not a list of tasks all still awaiting implementation.

1. Add optional, non-perturbing 5 ms execution/contact readout to the existing
   MuJoCo path. Diagnose final seed 1, paired case 6, with the original
   sample(16,2027), initialization, delays, and complete 17 seconds; compare
   Official and analyze seed 2's different calf-support pattern separately.
   Supplement missing Lab signals only if they change the next decision.
2. Fix the first actionable implementation or learning cause. An execution
   fix is first evaluated with existing fixed policies; a sampling fix does
   not silently change reward or termination timing. If a training change is
   needed, use one justified candidate and the existing 4096 by 24,
   4000-iteration recipe before deciding whether repeats are useful. Preserve
   failed runs and do not automatically extend a failed endpoint.
3. Establish grouped train/validation/test trajectories (approximately
   70/15/15, split seed 2027). Keep equal content or shared source recordings
   together where provenance is supplied. New split-trained policies use
   only train inputs, recipe selection uses validation, and final evaluation
   covers every test trajectory explicitly. Old full-pool checkpoints cannot
   acquire holdout status retrospectively. If a subsequent diagnosis justifies
   a training change, combine its training with the split instead of duplicating
   full-pool runs. The current negative diagnosis does not justify inventing a
   recipe change or starting training.
4. In parallel, make the current result tooling and installation route usable
   from a public checkout. Preserve historical evidence; move active report
   tools out of ignored outputs, correct stale status wording, retain licenses,
   and perform an isolated empty-cache installation and actual entrypoint
   checks. Existing environment migration is not repeated as unfinished work.

## KMA and release

Apply the already corrected KMA guidance in these real deliverables. Each
worker owns execution, observation, and routine recovery for its complete
scope. Master coordinates resources and advances ready work rather than
mirroring worker progress. Shared files and GPU execution have one operator.
Change guidance only for a concrete unresolved gap or conflict; no new
scheduler, telemetry, or mandatory team is needed.

KMA release is independent of robot training success: candidate tests and
exact-commit CI, immutable tag, public distribution and fresh-session discovery
checks, then Release. PawCerto research publication uses the completed behavior,
data-split and reproduction evidence with its known limitations disclosed. Keep
project-local KMA metadata outside PawCerto's public files.

## Later work and stop boundaries

For AS2 EDU / Piper H, transfer proceeds first through UMI-on-Legs, then
through RoboDuet. Preserve each method's own observations, controller, rewards
and training/evaluation path; asset preparation or a short integration run
does not qualify either transfer as learned control. See the current
[UMI robot adaptation](as2-umi-adaptation.md) and [RoboDuet evaluation protocol](roboduet-evaluation.md).

### Updated execution priority (2026-09-12)

The user has placed AS2 EDU / Piper H simulation transfer immediately after
the usable shared framework built from UMI, RoboDuet and DeepWBC. Finish those
methods' actual execution and evaluation paths, then extract only components
with multiple real consumers. Preserve each method's observations, rewards and
learning algorithm. Framework completion requires:

- Robot descriptions that explicitly identify joint order, mounting transform,
  inertia, limits, actuator/control parameters and available sensor signals.
- Working entrypoints selecting method, robot, configuration, checkpoint and
  simulation engine without silently changing the selected method.
- Reproducible evaluation records containing commands, states, errors and
  termination reasons, including incomplete or failed episodes.
- Exported policies that actually execute through their corresponding adapters.
- Public-source installation and external-input instructions that a new
  researcher can execute.

DeepWBC's corrected original-terrain 21-update, export and fixed 500-step
execution, independent CPU policy consumption and local source delivery are
complete. The [final report](experiments/deepwbc-reconstructed-runtime-20260912.md)
retains one signed-roll failure, poor tracking and the approximate six-axis
signal; the older freefall run is excluded from physical-task conclusions.
This bounded completion does not authorize larger DeepWBC training.

AS2/Piper H's authorized candidate completed **fresh seed 0, 4096
environments × 24 transitions × 4000 updates = 393,216,000 transitions**,
with the nominal physical inputs and original UMI supplied tossing task.
Its training service exited 0; RoboDuet remains in training under its original
50000-update budget and assigned GPU operator.
The first full-size attempt failed with OOM; preserving shared collision
geometry repaired initialization without reducing the budget or changing the
method. All 601 nominal control rows matched the previous corrected run exactly.
The earlier inverted-initial-pose update and OOM attempt remain retained.

The [executable AS2 learning protocol](as2-umi-learning-plan.md) preserves
model 0/500/4000 fixed16 evaluations, final paired Lab/MuJoCo comparison and
export. The actual 4096-environment run's model 0 baseline completed 16/16 full
17-second cases with no numerical failure or inversion, but mean tracking error
was 0.51053 m / 1.63250 rad. Its entire model state was measured equal to the
older one-environment baseline; the formal comparison still uses the new run's
own checkpoint. The [model 500 fixed16 evaluation](as2-umi-model500-evaluation.md)
completed with 13 numerical failures and three full but inverted cases.
The [final model 4000 result](as2-umi-learning-result.md) completes the prescribed
comparisons: Lab mean EE error improves from 0.510769 m / 1.625607 rad to
0.119318 m / 0.363301 rad, but 7/16 cases invert and ground support deteriorates.
MuJoCo has 16/16 BADQACC failures and zero complete cases. Actor export matches
the saved weights, but verification and independent consumption fail numerically;
matching invalid prefixes does not qualify a valid exported-policy rollout.
The one-candidate stop boundary is reached with a negative stable-control and
transfer outcome. Do not automatically add AS2 experiments or change parameters.
No extra seeds, reward search, extended training or hardware execution are
automatically included. A robot-name change, completed batch or successful
asset load is not a completed learned transfer.

A subsequent user-approved AS2 assembly investigation registered the official foot-AS2 rails and Piper mounting frame, built an explicit 160×180×6 mm plate candidate, and verified the six assets in official Lab and MuJoCo. The separately approved old-actor adaptation completed seed 0, 1024×24×1000 updates in 55.37 minutes, before its 90-minute training-loop cap, with fresh critic/optimizer. All eight model0/final×nominal/20 kg-body+2 kg-payload×Lab/MuJoCo evaluations reached a reported outcome. Nominal Lab EE error worsened from 0.270458 to 0.554300 m and zero-ground-support fraction from 15.35% to 34.93%; fewer inverted cases (11 to 8 of 16) do not establish improvement. Loaded model0 Lab failed one case and censored 15; final completed 16 but inverted 10. All four MuJoCo groups failed all 16 cases numerically. The [protocol and full result](as2-pretraining-family.md) retain complete/failed/censored and failed-prefix boundaries. This finite experiment did not establish usable transferred control.

The subsequent 22-asset AS2/Piper target distribution has completed official, unmodified Lab conversion and actual 22-environment reset plus 25 zero-action steps, with mass/COM/inertia readback and saved joint/TCP/whole-COM states. This proves actual asset consumption, not learning or demo visits to full-load arm extension; the post-auto-reset endpoint limitation is retained. Target-distribution demo pretraining and its pose coverage remain unfinished, with no extra training budget inferred from the completed six-condition adaptation. The original RoboDuet budget remains independent.

After the completed AS2 single-candidate experiment, UniFP's separately bounded
[official Lab integration](unifp-lab-runtime.md) completed 20+1 updates, a repaired
zero-update strict force-stage probe and one [500-step evaluation/export check](unifp-evaluation.md).
The original resume saved model 21 before its probe failed; that failure remains
preserved. One roll reset and 0.458129 m mean EE error, plus real-input CPU parity,
establish execution rather than learning. Evaluation repaired a gripper rigid-
property gap present during training; those physics configurations are not
claimed identical. No additional optimizer or simulator work follows automatically.

Learning Force Control's [official Lab integration](learning-force-control.md)
now preserves its original 19-output network/PPO, native position drives and
released task. The bounded 16-environment × 48-step × 20-update run exited 0;
independent resume saved model 21/16,128 transitions before its probe exited 1.
A separate zero-update inference-context repair exercised the boundary probe.
The corrected [500-step × 16-environment JIT evaluation](learning-force-control-evaluation.md)
completed 8,000 finite transitions with four orientation resets, position-mode
EE error 0.821928 m and force-mode XYZ component RMS 40.852856 N. The latter
uses known applied force, not wrist/contact-sensor accuracy. Independent CPU
consumption of every actual history passed numerical parity. The first reset
failure consumed no JIT actions; its source-limit-key mapping repair completed
the missing evaluation without repeating optimizer updates. The bounded LFC
assignment is closed, preserving all failures and the poor tracking. Author
weights were not obtained. Source-derived asset syntax repairs and recovered
upstream license provenance support this port without changing the physics engine.
UniFP full force-stage training, full LFC learning and the other expansion
methods remain in the overall unfinished scope. Their larger runs require their
own established budgets; these finite deliveries do not redefine the framework
goal as completed or authorize hardware execution.

The user also requires protecting the 32 GB workstation after memory pressure
terminated the desktop application. Large simulator/training processes run in
an independent systemd user service, outside the desktop application's cgroup.
The initial diagnostic limits were `MemoryHigh=16G`, `MemoryMax=18G`, and
`MemorySwapMax=1G`, verified in the actual cgroup. Preserving shared collision
geometry then allowed the same 4096-environment RoboDuet run to initialize and
perform real updates at about 7.3 GB service memory. The user authorized lifting
the temporary RAM limit once the issue was found; `MemoryHigh` and `MemoryMax`
were changed to infinity on that live service without restarting it. The
independent service and 1 GiB swap limit remain. Observe later-stage resource
use; do not disable system memory protection or blindly repeat an allocation
failure. If a future method cannot fit, state its measured limit and establish
an explicit environment-count/sample-budget adjustment before claiming its
original experiment is running.

The established near-term method sequence is UMI, RoboDuet on Go1/ARX5 with
its two-stage and five-module execution path, and DeepWBC with its actual
supported target dimensions and adaptation/advantage-mixing mechanisms.
Their current execution and learning evidence is distinguished above. MLM waits for sufficient
materials or an explicitly identified reimplementation. Use the completed
reference investigations rather than restarting them. Extract shared code only
when a second real method consumes it. Common-condition research and
AS2/PiperH transfer follow established methods; hardware remains out of scope.

### Approved framework expansion (2026-09-12)

The user has also approved UniFP, Learning Force Control, Multi-critic Twist
Tracking and ReLIC as parts of the final framework scope, with
LeggedManip_Lab as a separate engineering reference. This expands the planned
capabilities beyond pose tracking to force/compliance, end-effector velocity
tracking and interlimb cooperation. UniFP and Learning Force Control now have
bounded Lab execution; full force-control learning and the other integrations remain
uncompleted. Their evidence boundaries are stated above. Source inspections and their dated limits
are recorded in [the candidate research report](../exa-results/wbc-rl-candidates-2026-09-12/report.md).

| Component | Intended place in PawCerto | First integration boundary |
|---|---|---|
| UniFP | Low-level unified end-effector force/position control | Preserved B2/Z1 task, force estimator, supervision and strict stage transition now have bounded Lab execution and actual-input export parity. Full trained force control, cross-engine equivalence and the separate high-level imitation pipeline remain unverified. |
| Learning Force Control | Force tracking and variable-compliance baseline | Preserve B1/Z1 and history adaptation. The paper describes 17 effective controls, but the fixed executable source has 19 outputs and native position-drive control. CPU raw/JIT consumption, bounded Lab 20+1 updates and 500×16 actual JIT transitions are exercised; four orientation resets and poor tracking remain despite all-input CPU parity. Derived-only namespace/material repairs and fixed-upstream license provenance support the port. Author weights were not obtained; full learned force control remains unproved. |
| Multi-critic Twist Tracking | End-effector velocity tracking and separate value learning for leg, arm and contact objectives | Official training code has not been located. Obtain implementation materials or explicitly scope an independent paper-based implementation before coding; do not present its advantages as DeepWBC advantage mixing. |
| ReLIC | Optional research integration for arm/leg manipulation and changing support roles | An [optional original PLAY runtime](relic-external-runtime.md) now executes the 84-input/12-output export on the full 19-joint Spot, recording actual order and exact source downstream target writes. A finite 500-step run had eight ground-contact terminations; measured tracking and actual-input CPU replay do not establish stable performance or training reproduction. Historical contact equivalence and weights' training provenance remain unknown. Noncommercial upstream files stay external; this does not authorize redistribution or relicensing. |
| LeggedManip_Lab | Engineering reference for robot configuration, task entrypoints and deployment interfaces | Inspect working consumers and applicable licenses before reusing specific code. Robot configurations do not count as paper-method reproductions, and version compatibility still needs execution evidence. |

The existing UMI → RoboDuet → DeepWBC route remains the near-term sequence;
MLM retains its material-availability gate. UniFP is the first force-control integration with bounded Lab execution,
followed by Learning Force Control with bounded Lab integration as a comparison.
Multi-critic and ReLIC can advance when their material and permission
dependencies are ready, without blocking other methods. Method-specific plans
must establish their training budgets before starting large runs. The
[source-derived full learning budgets](full-learning-budgets.md) now make the
UniFP, DeepWBC and Learning Force Control workloads reviewable; these are
prospective allocations, not started experiments.

ReLIC's optional external runtime completed its bounded 500-step, zero-optimizer
assignment using the original PLAY task and downstream arm-seven/selected-leg-
three target overwrites. The actual physical tracking errors and eight contact
terminations remain part of its result, alongside all earlier failed attempts.
The original actuator compute path is retained through a local constructor
bridge; current normal-contact scalars are not proven equivalent to historical
vector/total-force signals. PLAY's Phase1 inheritance does not identify the
supplied weights' training phase. Full learning reproduction and stable ReLIC
performance remain unfinished; this delivery does not authorize further
simulation, training, redistribution of noncommercial inputs or hardware work.

Evaluate each new capability using its actual commands and execution path:
force tracking needs force error, transients, contact establishment/loss and
position/orientation tracking under interaction; compliance needs response to
specified disturbances; twist tracking needs linear and angular velocity errors
alongside pose drift and locomotion; interlimb cooperation needs role changes,
task completion and support behavior. Report falls, unintended head impact,
numerical failures, saturation and full-duration completion alongside these
metrics. Set numeric acceptance thresholds with the relevant method protocol,
not by transferring UMI's pose-error thresholds to different tasks.

Shared components should represent only the commands, sensors and action
semantics that working methods actually consume. Required force signals must
come from supported public APIs or explicitly validated reconstruction; UMI's
foot-force reconstruction is not automatically equivalent to end-effector
external-force supervision. Preserve original-method results and common-task
comparisons separately. The default installation remains official unmodified
Isaac Lab/PhysX, with no mandatory engine rebuild.

Stop uninformative repeated probes, unsupported reward searches, automatic
budget extensions, and test-set tuning. Failed independent tests block a
behavior-qualified PawCerto release. Repair installation/publication failures
at their actual owner and repeat only missing evidence. A new method's large
training budget, materially greater resource use, or hardware execution needs
a separately established boundary.
