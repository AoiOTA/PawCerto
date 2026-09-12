# PawCerto and KMA: continuation from the completed UMI baseline

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

Stop uninformative repeated probes, unsupported reward searches, automatic
budget extensions, and test-set tuning. Failed independent tests block a
behavior-qualified PawCerto release. Repair installation/publication failures
at their actual owner and repeat only missing evidence. A new method's large
training budget, materially greater resource use, or hardware execution needs
a separately established boundary.
