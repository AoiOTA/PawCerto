# Next RoboDuet learning experiment

The next development target is an actual two-stage learning result on Go1/ARX5.
The existing corrected two-update plus one-resume execution establishes
integration, recovery and five-module export, not learning. UMI is retained as
a completed research baseline with its known instability disclosed; small UMI
tracking differences do not block this next method.

## Two concrete implementation tasks

1. Align fresh-run episode-age initialization with the fixed upstream training
   entry. Upstream `auto_train.py` calls `learn(init_at_random_ep_len=True)` and
   randomizes `episode_length_buf`; the current local runner resets then reads
   observations without that randomization. Apply it to fresh initialization
   while preserving checkpoint-restored state. Check the same seeded tensor
   equation on CPU; no new short training campaign is needed for this change.
2. Add a standalone RoboDuet physical evaluation entry using the existing
   observer, controller, task/runtime and five-module policy. Current export
   verification uses saved inputs; UMI evaluators cannot establish RoboDuet
   behavior. Record errors and termination state before automatic reset.

## First complete candidate

- Fresh seed 0, original Go1/ARX5 and current corrected runtime.
- 4096 environments, 24 control steps per update, five PPO epochs and four
  minibatches; 50000 updates at the documented paper scale.
- Keep the original switch after zero-based iteration 10000: 10001 Stage 1
  updates followed by 39999 Stage 2 updates, 4915200000 transitions in total.
- Preserve the actual original observations, rewards, command curriculum,
  randomization and arm/dog rollout ordering. Do not retune against final tests.
- Save iteration 0, the Stage 1 boundary at next_iteration 10001, and 50000;
  save ordinary recovery checkpoints every 400 updates. At the switch endpoint
  arm_update_count is still zero; the planned final count is 39999.

This single seed-0 candidate is now running at the stated 4096-environment,
50000-update budget. Initialization failures, the shared-geometry repair and
the independent-service resource boundary remain recorded in the
[current status](status.md); the measured optimization and checkpoint resume
are in the [throughput record](roboduet-throughput.md). This is not a completed
learning result or three-seed reproduction. The local CLI still defaults to
2048 environments and needs an explicit additional-iteration budget; use the
recorded command when reproducing this candidate. Actual run measurements,
not the earlier 16-environment file timestamps, establish observed throughput
and memory. Concurrent AS2 training changes throughput, so the single-run
measurement is not a fixed completion-time prediction.

## Fixed evaluation

Use fixed seeds and commands within the supported ranges, covering standing
with EE motion, locomotion with a fixed EE goal, and simultaneous leg/arm
commands. Preserve original LPY/RPY and trajectory-time semantics. Report
`official_play` as the main playback mode, with any `training_mean` comparison
explicitly separated. This is a new engineering evaluation protocol because
the inspected upstream playback sources do not define a quantitative benchmark
or pass threshold.

Compare the initialization, stage boundary and final endpoint. Report leg
linear/angular velocity error, EE position/orientation error, non-timeout
termination, survival time, and video of simultaneous-command coordination.
Reward increase, finite loss or fewer falls alone is not learning success.
Retain early terminations and invalid prefixes instead of averaging them as
complete trajectories. Interpret measured improvement without inventing an
unapproved numerical acceptance threshold.

After this second method has a working learning/evaluation route, consolidate
only the robot/configuration, evaluation or deployment interfaces actually
shared by UMI and RoboDuet. Continue to DeepWBC and the documented force-control
methods afterwards; robot transfer follows usable method baselines.

References: [training implementation](roboduet-training.md),
[upstream evidence and budgets](roboduet-reference.md),
[framework roadmap](project-plan.md).
