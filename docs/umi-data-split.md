# UMI tossing grouped trajectory split

The existing three full-pool seeds remain valid historical training and sim2sim results. They are **not** held-out models. No split training or final test evaluation has been performed by this change.

`configs/umi_tossing_split_seed2027.json` identifies the unchanged upstream `reference/data/tossing.pkl` by SHA-256, source list index (zero-based trajectory ID), trajectory-content hash, and group. It contains no upstream trajectory arrays. Seed 2027 gives **71 train / 15 validation / 15 test**, covering all 101 IDs exactly once.

- Validation IDs: `7,22,32,40,43,50,56,59,64,73,78,80,82,83,92`.
- Test IDs: `1,8,13,20,28,36,47,48,51,53,54,76,81,88,94`.
- Train IDs: the remaining IDs, explicitly listed in the manifest.
- Source SHA-256: `a71400c1b7bf97b78d80a81a1a6505aace1b25d19eac55c7cff3f12cdddf3815`.

Grouping joins identical `ee_pos` and `ee_axis_angle` arrays and any shared explicit `recording_id`, `source_recording_id`, or `source_id`, including transitive links. This source contains 101 unique task trajectories and **no recording identity fields**. Thus list IDs and exact-content separation are verified; independence of fragments from an unknown common recording is unverified. Additional recording provenance would require regrouping and a new manifest identity before training; it must not silently change an existing experiment.

Groups are shuffled using Python `random.Random(2027)`, then assigned whole to the partition with the largest remaining trajectory-count deficit (train, validation, test tie order). Nominal targets use rounded 70% and 15%, with the remainder assigned to test; larger groups can make sizes approximate. The checked-in manifest is the source of truth for consumers.

## Commands and consumption

From the repository root with the documented environments installed:

```bash
# Regenerate for inspection under a new name; existing files are never overwritten.
conda run -n pawcerto-mujoco python scripts/split_umi_trajectories.py \
  --trajectory reference/data/tossing.pkl --seed 2027 \
  --output outputs/data-split/regenerated.json

# Only when the diagnosed recipe is ready: add this flag to the existing from-scratch train command.
# --split-manifest configs/umi_tossing_split_seed2027.json

# Development: every validation ID once, no random replacement.
conda run -n pawcerto-mujoco python scripts/eval_umi_split_mujoco.py \
  --checkpoint runs/SPLIT_RUN/model_4000.pt \
  --split-manifest configs/umi_tossing_split_seed2027.json \
  --partition validation --output outputs/SPLIT_RUN/validation

# After configuration/checkpoint selection is frozen, run the same command
# with --partition test and a fresh output directory. Do not tune on those results.
```

`scripts/train_umi.py --split-manifest ...` always selects train. The resolved source and manifest hashes, IDs, partition and seed are embedded in `sequence_sampler.trajectory_selection` in the saved config and trainer checkpoint. The existing trainer already persists that config. Resume and weight initialization must preserve the same selection; legacy full-pool initialization into a split is rejected. A saved split config requires the manifest flag to avoid accidentally reverting to full-pool training. No flag on a legacy config retains historical full-pool sampling.

`PoseSequence.select_ids(ids, seed)` supplies exactly the specified source IDs using the existing centering, padding and height processing. It does not draw replacement trajectory IDs. The new MuJoCo entry defaults to evaluation seed 2027; the historical Lab author-protocol default remains 2026. After saving all individual results and the summary, invalid or incomplete aggregate status exits with code 2. The MuJoCo script supplies those positions and rotations to the ordinary evaluator and saves each source ID plus the aggregate. Old policies remain explicitly marked `held_out_from_recorded_training_partition=false`. Training identity is read from the full weight checkpoint itself, never from adjacent execution config that another run could overwrite. Missing checkpoint metadata and actor.ts-only bundles are unqualified even if their config lists a train partition. Matching recorded training-partition metadata supports only that qualified partition claim; it cannot certify that users never inspected test results.

The author-protocol Lab entry `scripts/eval_umi.py` accepts the same manifest plus `--partition validation|test`; validation is the default. It preserves the original author sampling/reset protocol, so it **does not promise exhaustive coverage**. Its report now describes the actual selected pool and checkpoint training identity instead of hard-coding “same training pool.” Use the explicit-ID MuJoCo entry for exhaustive fixed-trajectory coverage; this change does not claim exhaustive Lab evaluation.

## Verification and limits

- `pawcerto-mujoco` CPU: 7 split tests passed (determinism, duplicate and recording grouping, coverage, cross-group leakage rejection, changed-source rejection, selected-ID consumption, original sampler behavior, legacy identity, nonzero CLI exit after preserving every failed ID and the aggregate).
- Actual source: all train/validation/test tensors loaded in source-ID order; all 101 unique groups assigned once; source checksum unchanged. Evidence: `outputs/data-split/actual_source_check.json` and `tests.log`.
- Entry integration: official legacy weights, 15 validation IDs, 0.2 s each, completed through the real MuJoCo evaluator; evidence: `outputs/data-split/validation_entry_smoke_02s/`. This is an entry smoke check, not full-trajectory behavior or generalization evidence. No policy was run on test IDs.
- An initial 0.02 s probe failed because the evaluator spends one policy step on reset and then had no recorded steps. The wrapper now rejects durations shorter than reset plus one policy step. The original failure log is retained.
- The broad existing policy test initially could not import `pytorch3d` in the MuJoCo environment; that dependency failure is retained separately, not reported as a failed split behavior. Reference-environment result is reported in `policy_regression_reference.log`.
- Fresh-environment Lab execution now consumes the 71 train IDs: 16 environments, one initial update/save, then two additional resume updates, all processes exit 0. The resume operator intended one additional update but passed `--iterations 2`, so the actual total is three; execution is stopped and the extra update is retained. See [the reproduction record](release-reproduction.md) and `outputs/release-reproduction-20260912/lab-smoke-execution.json`. This is an installation/entry check, not a learned baseline or test-set evaluation.
- Full-budget split training, formal test evaluation and acceptance remain pending. Data preparation and finite entry updates do not satisfy project generalization acceptance.
