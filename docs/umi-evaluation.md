# UMI evaluation conditions

The original paired 16-case, 17-second evaluations were diagnostics. They used `sample(16, seed=0)` from the tossing training pool with the same targets across checkpoints. Nominal and push/transport results remain separate. They do not constitute an independent trajectory holdout or reproduce the author's evaluation command verbatim.

## Stage-one evaluation choice

Before inspecting the from-scratch iteration-4000 result, the evaluation is fixed to **seed 2026**, 250 parallel environments and a nominal 500 completed episodes from the same tossing pool. Use the actual checkpoint configuration as the base so that observation dimensions, scales and control settings match the trained actor. Apply the author's evaluation behavior explicitly: deterministic actions, dense task/history updates, evaluation initialization, no random target-height offset, and disabled push/transport. Retain the prescribed physical randomization and observation noise. Save the resolved evaluation configuration with the results.

This is **checkpoint configuration plus author evaluation semantics**. It is not an exact reproduction of the current upstream starter command, and new random initial conditions do not make the training trajectories held out. The upstream runner stops when global completed episodes exceed the nominal target, with a four-episode-length step limit. Report actual completed and unfinished counts, the stopping reason and duration, rather than assuming exactly 500 episodes or two per environment.

Report episode mean position error in metres, geometric orientation error in radians, timeout fraction as the author's survival proxy, and electrical power. Compute per-episode time means before the aggregate across completed episodes. The author retains only the latest nominal-target count of completed episodes in a bounded deque; retain this reported window and also save all completion records and the number evicted. Include support and instability diagnostics separately. A terminated episode's tracking mean describes its executed prefix; report termination/completion alongside it. Neither survival nor a small error on a selected trajectory alone establishes WBC success.

The author's legacy evaluation runner also carries trial-step critic values into reward timeout bootstrapping. That reporting behavior does not define the position, orientation, survival or power metrics. Any raw environment reward report must be distinguished from that legacy runner's reward/return statistic.

## Source boundaries

The [paper, section 10.2](https://arxiv.org/html/2407.10353v1#S10.SS2), describes checkpoint 4000, 500 episodes, random dataset trajectories, randomized initial pose and physics, and a 17-second survival criterion. It does not define a position/orientation threshold with a required holding period or specify a held-out trajectory split.

In the local original source under `third_party/umi-on-legs/mani-centric-wbc/`:

- `docs/starter.md:62` and `docs/wbc.md:7` use the same tossing file for evaluation and training; `legged_gym/env/isaacgym/pose_sequence.py` samples the entire supplied pool.
- `config/eval.yaml` and `legged_gym/rsl_rl/runners/on_policy_runner.py` define dense evaluation, initialization, episode collection and disabled external disturbances.
- `legged_gym/rsl_rl/runners/utils.py` aggregates completed episodes; `legged_gym/env/isaacgym/task.py` defines the position and orientation errors.

Current upstream Hydra defaults do not automatically load the saved checkpoint configuration. The composed starter configuration differs from the saved `ours` run in observation scaling, pose latency and root-height settings. Dense evaluation also updates history at the physics cadence, unlike training's control cadence. These differences must remain visible when comparing results; do not silently substitute current defaults for the trained policy's inputs or claim complete numerical equivalence to the paper.

## Data and remaining validation

The published local files contain 101 tossing trajectories, 200 pushing trajectories and 1090 cup trajectories. The other two are separate tasks, not a tossing test split. No split marker or additional test archive was found in the supplied data or official download documentation. A true unseen-tossing test requires trajectories that were not used for training; selecting some of the already used 101 trajectories afterward cannot provide it.

The current body-speed-v3 stage completed three fixed 4000-iteration training seeds and paired `sample(16, seed=2027)` final evaluations. Native nominal16 and author latest500 also use seed 2027; the seed 2026 protocol above belongs to the historical stage. All three final nominal MuJoCo groups completed, but seed 1 head impact and two author-protocol inversions remain. Repetitions on the same pool do not provide independent data validation. This document does not set a new universal accuracy threshold or narrow the original whole-body-control objective.
