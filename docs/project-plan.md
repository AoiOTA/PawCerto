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

## Current acceptance

Preserve the existing seed-matched position and orientation means under the
same engine, target set, and protocol while correcting uncontrolled behavior.
Complete every fixed full-duration case without inversion, numerical failure,
automatic reset, or the known head-impact event; author-protocol evaluation
must no longer contain inverted episodes. Retain all early-termination strata.
Do not introduce a requirement to match every official-policy metric.

Calf contact can be useful support in a low posture. Neither nonfoot contact
count, load share, nor zero supported feet alone is failure. Interpret contact
establishment, tracking, forces, slip, actuator saturation, and the subsequent
motion together. Exercise a posture exit only if the task requires it.

Use official unmodified Isaac Lab/PhysX and the installed fixed Conda
environments. Keep body-speed-v3 and the explicit 1000 rad/s limit as the
current baseline; change them only when evidence supports the change. Exact
Lab/Gym numerical equality and a PhysX rebuild are not prerequisites.

## Deliverables and current execution state

Updated on 2026-09-12 after the bounded diagnosis. Item 1 is complete: the
non-perturbing 5 ms readers, matching Lab signals, ground/free comparison and
one actual Lab PD-warmup-state intervention are documented in
[the diagnosis](umi-contact-next-step.md). The intervention did not remove
the head collision. No actionable control implementation defect or justified
recipe change was identified; this diagnostic branch is stopped and no new
long training is started. Item 2 therefore currently has a negative result,
not a behavior-qualified repair.

Item 3's grouping, ID selection and checkpoint provenance are implemented
(71 train / 15 validation / 15 test), but split training and independent
held-out behavior evaluation remain unfinished. Item 4's public reproduction
checks continue independently. The original scopes below remain the acceptance
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
checks, then Release. PawCerto release additionally requires the behavior and
independent-data evidence above, plus the public reproduction route. Keep
project-local KMA metadata outside PawCerto's public files.

## Later work and stop boundaries

After UMI's current stage, continue RoboDuet on Go1/ARX5 with its real two-stage
and five-module execution path, then DeepWBC with its actual supported target
dimensions and adaptation/advantage-mixing mechanisms. MLM waits for sufficient
materials or an explicitly identified reimplementation. Use the completed
reference investigations rather than restarting them. Extract shared code only
when a second real method consumes it. Common-condition research and
AS2/PiperH transfer follow established methods; hardware remains out of scope.

### Approved framework expansion (2026-09-12)

The user has also approved UniFP, Learning Force Control, Multi-critic Twist
Tracking and ReLIC as parts of the final framework scope, with
LeggedManip_Lab as a separate engineering reference. This expands the planned
capabilities beyond pose tracking to force/compliance, end-effector velocity
tracking and interlimb cooperation. These are planned integrations, not
implemented or reproduced methods. Source inspections and their dated limits
are recorded in [the candidate research report](../exa-results/wbc-rl-candidates-2026-09-12/report.md).

| Component | Intended place in PawCerto | First integration boundary |
|---|---|---|
| UniFP | Low-level unified end-effector force/position control | Preserve the B2/Z1 task, force estimator, supervision and stage transitions. Public training code exists; policy availability and the separate high-level imitation pipeline remain unverified. |
| Learning Force Control | Force tracking and variable-compliance baseline | Preserve B1/Z1, history adaptation and the actual 17-action control boundary. Verify local checkpoint loading and distinguish actuator-network files from policy weights. |
| Multi-critic Twist Tracking | End-effector velocity tracking and separate value learning for leg, arm and contact objectives | Official training code has not been located. Obtain implementation materials or explicitly scope an independent paper-based implementation before coding; do not present its advantages as DeepWBC advantage mixing. |
| ReLIC | Optional research integration for arm/leg manipulation and changing support roles | Retain model-based manipulation plus learned support. The inspected code has a non-commercial research license; keep it outside the default broadly reusable distribution unless appropriate permission is established. Including the method in the plan does not authorize relicensing its code. |
| LeggedManip_Lab | Engineering reference for robot configuration, task entrypoints and deployment interfaces | Inspect working consumers and applicable licenses before reusing specific code. Robot configurations do not count as paper-method reproductions, and version compatibility still needs execution evidence. |

The existing UMI → RoboDuet → DeepWBC route remains the near-term sequence;
MLM retains its material-availability gate. UniFP is the first proposed force
control integration, followed by Learning Force Control as a comparison.
Multi-critic and ReLIC can advance when their material and permission
dependencies are ready, without blocking other methods. Method-specific plans
must establish their training budgets before starting large runs.

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
