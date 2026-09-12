# Learning Force Control: B1/Z1 integration

PawCerto now provides the pinned original Learning Force Control CPU components
and a bounded official, unmodified Isaac Lab/PhysX task/runtime integration.
This is not full learned force control, author-policy reproduction or hardware
acceptance. Author weights were not obtained by the recorded anonymous W&B
request; its `project:null` response does not establish that they do not exist.

## Researcher entrypoints

- [CPU source acquisition, raw-checkpoint and two-JIT consumption](learning-force-control-implementation.md)
  preserves the original raw-state interface and records its historical tests.
- [Asset conversion, terrain and runtime](learning-force-control-physics.md)
  documents the real B1/Z1 physics and public state/force interfaces.
- [Released task, training and independent resume](learning-force-control-lab.md)
  describes the full training checkpoint, source lifecycle and bounded execution.
- [Full-checkpoint export and fixed JIT-driven evaluation](learning-force-control-evaluation.md)
  records the distinct export contract, actual-input comparisons and failure accounting.

Run scripts from the source checkout with the documented official Lab Python;
the wheel does not include scripts, original vendor assets or local run outputs.
The commands in the linked reports document the bounded work and do not
authorize another simulator, optimizer run or larger training budget.

## Actual source semantics

The fixed source uses **19 network outputs and 19 movable joints**, despite the
paper-level description of 17 effective controls. Action 18 is overwritten
with -0.1 during control. The released task sends position targets through
native PhysX position drives. Its separately computed `source_pd_torques`
remain reward/termination inputs, not measured or submitted drive effort.
The released default has no instantiated door/ball object.

Original Gym lookup for `gripperStator` returns -1, causing the original tensor
code to measure the final body, `gripperMover`. The port explicitly names that
measurement body; external force still targets `link06`. This preserves measured
source execution without claiming the negative lookup was the authors' intent.
Force commands, known applied spring force and contact signals remain distinct;
none becomes an independently measured wrist-force sensor by naming convention.

The original URDF bytes remain intact. Conversion repairs only the derived
input's missing xacro namespace, unused includes and duplicate material name;
source body/joint/mass comparisons are retained in the physics report. Referenced
legged_gym and rsl_rl license texts were obtained at fixed upstream commits
identified by the released README, with explicit provenance. They are not
claimed to be copies recovered from the missing LFC directories. Optional
object-task and unused recurrent branches were not automatically reinstated.

## Bounded execution status

The 16-environment × 48-step × 20-update run exited 0. Separate-process resume
saved model 21 with 16,128 training transitions, then its post-save boundary
probe exited 1 because of an inference-context write. A separate recovery
exercised the repaired probe with zero optimizer updates. The saved failure
remains part of the execution record; resume state does not include hidden
PhysX solver/contact caches and cannot establish bitwise physical replay.

The [corrected fixed evaluation](learning-force-control-evaluation.md) exited 0:
500 policy steps × 16 environments produced 8,000 finite, actual JIT-driven
transitions, with exact device checkpoint/JIT action and latent parity. Four
body-orientation terminations caused automatic resets; this is not a no-fall
trajectory. Position-mode EE mean error was 0.821928 m across 4,555 samples.
Force-mode XYZ component RMS error was 40.852856 N across 3,445 samples, or
48.492836 N across 2,445 constrained force-mode samples. Base XY velocity
component RMS error was 0.247447 m/s and minimum up-dot was 0.282486.
Force errors compare the source's known applied force to its commands, not an
independent measured wrist/contact force.

A separate CPU process consumed all 8,000 saved histories with the same JIT
pair. CPU checkpoint/JIT actions and latents matched exactly; differences
from saved GPU actions/latents were at most 3.278255e-7 / 1.335144e-5, passing
the combined atol=1e-6, rtol=1e-5 comparison. This proves numerical consumption
on those inputs, not policy quality or a new physical rollout.

The first evaluation failed at reset with zero JIT consumption: the evaluator
used nonexistent velocity-limit keys. Correcting the mapping to source
`limit_vel_x/y/yaw` and testing the real initializer/sampler on CPU resolved
the cause. The failed output remains separate; the new attempt completed the
missing 500 steps without another optimizer update. The bounded assignment is
closed, with poor tracking and all failures retained.
These 21 updates are a bounded integration budget, not the original 100,000-
update experiment, learned force/compliance acceptance, sim2sim or hardware readiness.
