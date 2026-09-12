# AS2/Piper-H UMI: one-candidate learning and evaluation protocol

**Recorded status (2026-09-13): the repaired full-size run entered PPO updates; its actual model 0 fixed16 baseline completed.** The candidate is fresh seed 0, **4096 environments × 24 transitions × 4000 PPO updates**, using the current nominal AS2 EDU/Piper-H robot and the original supplied tossing task. This document specifies the executable run and its comparisons. It does not report learned AS2 control or hardware readiness.

The preceding [adaptation work](as2-umi-adaptation.md) established corrected initial orientation, source/FK agreement, 3-second nominal Lab basic control and one complete fresh PPO update. The earlier inverted-initial-pose run remains an invalid adaptation record. Neither the corrected short update nor zero-action PD establishes trajectory learning.

## Public checkout and local evidence

Run commands from a PawCerto source checkout. In the commands below, `python`
means the existing CPU/MuJoCo interpreter for preparation, target packing,
MuJoCo evaluation and export, or the official Lab interpreter for training
and native Lab evaluation. Follow [environment setup](isaaclab-install.md),
[UMI inputs](../pawcerto/mujoco/README.md) and
[AS2 source acquisition/build](as2-piper-assets.md#reproduce-on-cpu) before
preparing the [official Lab asset](as2-piper-isaac.md#reproduce).
Neither patched PhysX nor a custom Isaac Lab build is required.

The Python wheel supplies modules, not checkout scripts, robot assets,
checkpoints or experiment results. `reference/`, `runs/` and `outputs/`
paths below identify local inputs and recorded evidence, not files shipped
with the public source. With the documented external inputs available,
create the training configuration on CPU:

```bash
python scripts/prepare_as2_umi.py --output runs/as2_umi_adaptation/config.json
```

Rebuild generated asset paths on another machine; saved configurations can
contain absolute paths. The following training/evaluation commands document
the separately scheduled experiment, not a requirement to start GPU work
while installing or inspecting the source.

## Candidate and launch boundary

Use the prepared `runs/as2_umi_adaptation/config.json` and the official, unmodified Lab/PhysX installation. Preserve:

- AS2/Piper nominal source assembly, stock gripper fixed open 40 mm, 0.12 m nominal mount and `piper_gripper_base + (0,0,.138)` TCP; source-model assumptions remain uncalibrated hardware inputs.
- 18 actions in the saved FL/FR/RL/RR leg then Piper joint order; actor 132 / critic 269; the corrected Lab 3 xyzw initialization.
- Current offsets, gains, source torque/velocity limits, mass/COM/material randomization and contact/termination bodies. The preparation selects source velocity limits, with no legacy Go2 1000 rad/s override.
- Original UMI observation/history, 20 ms control delay, four 5 ms physics steps, supplied `reference/data/tossing.pkl` trajectories and sampler, reward formulas/weights, curriculum and PPO. No reward search, task simplification or alternate robot is included.
- A fresh random initialization: **no `--resume`, no `--weights`** for initial launch. Only this seed and candidate are authorized.

Start only after **DeepWBC's currently assigned 21-update/export/evaluation work and the shared source installation delivery are complete**, and the **single GPU operator has checked actual resource headroom** and arranged the independent training service. Do not launch early, silently reduce 4096 environments, or overlap another GPU simulator outside that operator's schedule. Insufficient resources mean wait/reassign resources through the coordinator, not a smaller substitute experiment. This preparation task performs CPU work only.

The full budget is 393,216,000 policy transitions. Save every 500 updates, including the trainer's actual initial `model_0.pt`, and stop at update 4000. Preserve logs, exit status, config and asset/source identities. Observe any asynchronous failure and retain its cause; a valid resume of this same run must retain its optimizer/RNG/config and the original total-update budget. A changed physical recipe, extra seed, reward change or longer run is a new candidate and is not automatic recovery.

## Startup failure and bounded representation repair

The first 4096-environment launch was killed by the memory cgroup on 2026-09-13 before producing the run directory or `model_0.pt`. The kernel recorded SIGKILL for the simulator process with 12,523,760 KiB anonymous resident memory under a 12 GiB RAM / 1 GiB swap cgroup. The application service peak did not capture that kernel-observed usage. No learning or checkpoint is claimed from this attempt.

The AS2 runtime previously removed mesh instancing and authored identical physics attributes after all environments had been cloned. The repair prepares the same contact/rest offsets in the shared USD geometry and applies the existing collision/body/contact-report settings to the source robot before cloning. AS2 retains the matching geometry instances; the existing Go2 path is unchanged. This changes representation and when identical settings are authored, not the 4096-environment budget, controller, task, physics recipe or PPO candidate.

The original asset is archived at `outputs/as2-umi-learning-execution-20260913/asset-before-shared-offsets`. CPU validation found 42 collisions with 11 instance proxies retained; 2586 existing attributes other than the two declared collision offsets, relationships, prim types and instancing flags were unchanged. Source inertial/FK/joint/shape validation and 31 relevant CPU tests passed. Reports are `shared-offset-source-validation.json` and `shared-offset-equivalence.json` in the same output directory. The subsequent real 3-second run exited 0. Across all 601 recorded rows, joint position/velocity, commanded/applied torque, TCP pose, root position/up-dot, net contacts and ground support were exactly equal to the prior corrected run (`runs/as2_umi_nominal_lab_3s_fixed_xyzw`), with maximum absolute difference zero for every compared array. The config, controller and URDF identities also matched. The comparison and PD/FK audit are in `runs/as2_umi_nominal_lab_3s_shared_geometry/{shared-geometry-control-comparison.json,paired-cpu-audit.json}`. This demonstrates unchanged observed nominal single-environment control; it does not yet prove full-size memory capacity or learned tracking. The second full-size launch entered PPO updates under the same 12 GiB RAM / 1 GiB swap cap. Its successful startup is capacity evidence; the original OOM and source/asset snapshots remain preserved.

## Completed pretraining CPU baseline

The requested existing **training-before-update** checkpoint, `runs/as2_umi_runtime_update_fixed_xyzw/model_0.pt`, was evaluated with one shared `sample(16, 2027)` draw, each case covering the full 17 seconds. This is an untrained network from the corrected **one-environment integration run**, not its `model_1.pt`.

```bash
python -m pawcerto.mujoco.evaluate \
  --checkpoint runs/as2_umi_runtime_update_fixed_xyzw/model_0.pt \
  --trajectory reference/data/tossing.pkl --cases 16 --seconds 17 --seed 2027 \
  --output runs/as2_umi_model0_fixed16_seed2027
```

This command has executed and exited 0. Results:

| Measure | Existing model0 baseline |
|---|---:|
| Full cases / requested cases | 16 / 16 |
| Numerical failures / inverted cases | 0 / 0 |
| Mean EE position error | 0.5105255 m |
| RMS EE position error | 0.5408080 m |
| Mean EE orientation error | 1.6324972 rad |
| Mean ground-supported feet | 3.941696 |
| Endpoints with zero supported feet | 1.0601% |
| Cases with sampled nonfoot contact above 1 N | 0 |
| Minimum root up-dot | 0.9863633 |

Across the 16 cases, root XY net displacement is 0.04047–0.04542 m (median 0.04160 m), while sampled XY path length is 0.29894–0.31201 m. The saved audit also retains per-case arm joint ranges and EE motion in base coordinates. These full-episode motion values include the release transient and possible oscillation; they do not establish coordinated gait.

Each case retains JSON and NPZ with 849 measured policy endpoints after the one-step reset warmup, ending at 17 seconds (3400 physical steps including warmup). Numeric arrays were checked finite. Contact absence above is at saved 20 ms endpoints, not continuous contact-quality proof. The actor's approximately half-metre tracking error remains the relevant untrained baseline; standing without inversion is not trajectory success.

Artifacts are `runs/as2_umi_model0_fixed16_seed2027/{summary.json,baseline-audit.json,case_*.json,case_*.npz,fixed16_targets.npz}`. The frozen targets contain `positions[16,T,3]` and `rotations[16,T,3,3]`; their SHA-256 is `d118cdc555bcfa661a0afa566b2a838f1513416e580109efa93fcd48eb1337d5`. The supplied trajectory archive SHA-256 is `a71400c1b7bf97b78d80a81a1a6505aace1b25d19eac55c7cff3f12cdddf3815`. This is a supplied-training-pool diagnostic, not held-out evaluation.

**The main learning comparison must use the new 4096-environment run's own `model_0.pt`.** Runtime randomization consumes random numbers before actor construction, so the same seed with 1 versus 4096 environments does not imply identical initial weights. Retain the completed baseline above as a pretraining engineering reference; do not silently substitute it for the actual training-run baseline.

## Training command — executing attempt 2

Run from the repository root in the established Lab environment, after the launch conditions above:

```bash
python scripts/train_umi.py \
  --config runs/as2_umi_adaptation/config.json \
  --usd-path reference/isaac/as2_piper/merged/merged.usda \
  --urdf-path reference/isaac/as2_piper/merged.urdf \
  --trajectory reference/data/tossing.pkl \
  --num-envs 4096 --iterations 4000 --seed 0 \
  --force-signal reconstructed-solver --save-every 500 \
  --output runs/as2_umi_seed0_4096_4000 \
  --visualizer none --device cuda:0
```

The GPU operator owns the service and resource allocation; the command is not authorization to start a second uncoordinated simulator. The trainer writes `config.json`, `metrics.jsonl`, `model_0.pt` and subsequent checkpoints. Record actual source revision/snapshots at launch. Do not overwrite an earlier attempt's output with a fresh run.

## Actual 4096-environment model 0 baseline

The full run's actual `model_0.pt` was independently evaluated with the fixed16 command and exited 0: all 16 cases completed 17 seconds, with zero numerical failures, inverted cases or sampled nonfoot-contact cases above 1 N. Mean EE error is 0.5105255 m / 1.6324972 rad; mean ground-supported feet is 3.941696 and zero-support endpoints are 1.0601%. Every saved target exactly equals the frozen fixed16 array, and every saved numeric array is finite.

The checkpoint records iteration 0, zero transitions and 4096 environments. Its SHA-256 is `2f8a8b5519edc062486a994a023c4af267ee17a9b19b49c7a81900ebdd2e7718`. Actual tensor comparison found its entire model state equal to the earlier one-environment model 0, explaining the identical rollout metrics; this equality was measured rather than inferred from seed equality. The formal comparison uses this new run's checkpoint and artifacts: `runs/as2_umi_seed0_4096_4000_fixed16_0/{summary.json,fixed16-audit.json,case_*.json,case_*.npz}`. Model 500, model 4000 and final Lab/export results remain pending.

## Fixed checkpoints: 0, 500 and 4000

After the formal model0 fixed16 evaluation, freeze its actual saved targets
for paired Lab evaluation using this CPU-only command (no simulator import):

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

root = Path('runs/as2_umi_seed0_4096_4000_fixed16_0')
positions, rotations = [], []
for case in range(16):
    with np.load(root / f'case_{case:02d}.npz', allow_pickle=False) as saved:
        positions.append(saved['sampled_target_positions'].copy())
        rotations.append(saved['sampled_target_rotations'].copy())
positions = np.concatenate(positions, axis=0)
rotations = np.concatenate(rotations, axis=0)
assert positions.shape[0] == 16 and positions.shape[2:] == (3,)
assert rotations.shape == positions.shape[:2] + (3, 3)
assert np.isfinite(positions).all() and np.isfinite(rotations).all()
np.savez_compressed(root / 'fixed16_targets.npz', positions=positions, rotations=rotations)
PY
```

Case order is exactly 00 through 15. This derives the target arrays from the
formal run's evaluation, without resampling or depending on the historical
one-environment baseline. Compare target arrays, not NPZ container hashes.

Evaluate the exact new run's checkpoints using unchanged MuJoCo model, config and `sample(16,2027)`, each for the full 17 seconds. CPU evaluation can run when a saved checkpoint is complete and the operator's resource budget permits it; do not modify the checkpoint or training config. This loop is executable once all three checkpoints exist; at milestones the same command can be run for the available single checkpoint:

```bash
as2_eval_status=0
for as2_iteration in 0 500 4000; do
  python -m pawcerto.mujoco.evaluate \
    --checkpoint "runs/as2_umi_seed0_4096_4000/model_${as2_iteration}.pt" \
    --trajectory reference/data/tossing.pkl --cases 16 --seconds 17 --seed 2027 \
    --output "runs/as2_umi_seed0_4096_4000_fixed16_${as2_iteration}" || as2_eval_status=$?
done
test "$as2_eval_status" -eq 0
```

Verify each output's saved sampled targets equal the corresponding case in `runs/as2_umi_seed0_4096_4000_fixed16_0/fixed16_targets.npz` before making paired comparisons; identical seeds alone do not certify unchanged inputs. The evaluator retains invalid numerical prefixes separately, reports complete-case denominators and exits 2 for an incomplete batch. Preserve that exit and failure detail. An exception before any valid prefix also remains a failure; do not relabel a shorter rollout as a complete 17-second case. If complete subsets differ, show per-case outcomes and compare shared elapsed prefixes separately from complete-episode statistics; never hide failures by averaging only the convenient cases.

At all three points report position **and** orientation error, full-case count, inversion, root height/up-dot, ground versus net support, nonfoot contacts and numerical failures. Use the saved `physical` columns for root translation/rotation, arm joint motion, targets and TCP motion to describe leg/arm behavior; base motion alone does not establish coordinated control. Preserve all cases, including checkpoint 500 if checkpoint 4000 regresses. A favourable training reward or finite PPO update is not the outcome; the report must make the tracking/support/contact tradeoff explicit. No new binary tracking-success threshold is introduced here.

## Final Lab fixed-policy comparison and export

After training, the single GPU operator executes the original and final policies on the same frozen 16 targets, nominal physics and 850 policy periods (=17 seconds including reset warmup):

```bash
python scripts/run_umi_isaac.py \
  --checkpoint runs/as2_umi_seed0_4096_4000/model_0.pt \
  --compare-checkpoint runs/as2_umi_seed0_4096_4000/model_4000.pt \
  --target-sequences runs/as2_umi_seed0_4096_4000_fixed16_0/fixed16_targets.npz \
  --trajectory reference/data/tossing.pkl \
  --num-envs 16 --steps 850 --seed 2027 --contact-summary \
  --output runs/as2_umi_seed0_4096_4000_lab_fixed16.json \
  --visualizer none --device cuda:0
```

This is a fixed-policy, nominal same-target comparison. It does not add a different author-evaluation reset distribution or stochastic success protocol. The contact summary samples every existing 5 ms step; MuJoCo's standard saved contact report samples at 20 ms endpoints, so do not compare those occupancy counts as if their sampling were identical. Report cross-engine trajectory errors and physical failure differences without treating agreement as hardware validation.

Export the final checkpoint and validate the independent consumer:

```bash
python scripts/export_umi.py \
  --checkpoint runs/as2_umi_seed0_4096_4000/model_4000.pt \
  --output runs/as2_umi_seed0_4000_export

python scripts/verify_umi_export.py \
  --checkpoint runs/as2_umi_seed0_4096_4000/model_4000.pt \
  --export runs/as2_umi_seed0_4000_export \
  --trajectory reference/data/tossing.pkl --seconds 17 --seed 2027 \
  --output runs/as2_umi_seed0_4000_export_verification
```

This export/verification path was exercised on the existing model0 while preparing the protocol: its first verification correctly exposed a remaining Go2-only joint-list check. The source-delivery owner changed that check to the saved robot binding; the original failure remains in `runs/as2_umi_model0_export_verify_before`. After repair, both independent 17-second CPU consumers exited 0 and all saved arrays matched exactly, recorded in `runs/as2_umi_model0_export_verify_after/verification.json`. This validates the consumer path using untrained weights, not the future model4000.

The export keeps the actual AS2 binding, joint order, controller, actor and source provenance. The verifier checks actor identity and independent-process original/exported rollout parity on one sampled case; it is package validation, not the fixed16 learning conclusion. The final 16-case MuJoCo result above remains required. Numerical failure in export replay must remain reported, even if source and export fail similarly.

## Completion boundary

Deliver the training log/config/identities and model 0/500/4000 checkpoints; fixed16 per-case arrays, metrics and failure accounting at all three points; final paired Lab readout; export and consumer-verification evidence; and a short interpretation of tracking, physical support, contact and engine differences. Stop at this one candidate and 4000 updates. Failed learning is a valid result to report; it does not automatically authorize more seeds, longer training, altered rewards, a simpler task or hardware execution.
