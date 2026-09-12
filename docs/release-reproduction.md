# Public reproduction and candidate release

The source is being prepared for release. The current three training seeds and
48 full MuJoCo evaluation cases are documented in [current results](umi-current-result.md).
Seed 1 head impact, author-protocol inversion, and lack of an independent
trajectory holdout prevent a behavior-qualified release. Package parity and
installation checks do not remove those limitations.

## What the source provides

The repository contains UMI adaptation, training/evaluation/export entry points,
fixed environment requirements, robot joint order, input fetching and conversion,
and result analysis. Original code uses the root MIT license; [NOTICE](../NOTICE)
identifies third-party ownership. Neither downloaded upstream assets nor NVIDIA
packages acquire the root license. KISS My Agent is optional local development
tooling and is not required to install or execute PawCerto.

`reference/`, `third_party/`, `runs/`, `outputs/` and generated MJCF/USD files are
not source deliverables. Run `scripts/fetch_umi.py` and `scripts/build_mujoco.py`
from the README to obtain official inputs and regenerate paths. Historical
links into `outputs/` refer to separate research evidence, not files supplied by
a source checkout. No published PawCerto weight/evidence download is claimed
until the release assets actually exist.

## Maintained analysis commands

The following scripts were migrated from the current experiment outputs. They
read supplied records and use the active Python interpreter; none selects a
PawWeaver environment. Run with the MuJoCo environment described in the README.
Use a new output directory to preserve previous evidence.

```bash
python scripts/readout_umi_contacts.py \
  --policy path/to/candidate/model_4000 \
  --reference path/to/official_ours --output new/contact-readout
python scripts/readout_umi_coordination.py \
  --policy path/to/candidate/model_4000 \
  --reference path/to/official_ours --output new/coordination-readout \
  --policy-config path/to/candidate/config.json \
  --reference-config reference/checkpoints/tossing/ours/config.json
python scripts/verify_umi_export.py \
  --checkpoint path/to/model_4000.pt --export path/to/exported_bundle \
  --trajectory reference/data/tossing.pkl --output new/export-verification
```

Contact and coordination readers require all 16 saved cases to be complete;
incomplete prefixes cannot be compared as though they were full trajectories.
They preserve the original 1 N and 20 ms sampling definitions: sampled force
sums are not continuous impulses and adjacent supported foot positions do not
prove continuous sliding. The exporter verifier checks actor/config/joint-order
identity, launches original and exported policies in separate processes, and
compares saved arrays and JSON. It retains actual nonzero process exits; it
cannot pass on an incomplete rollout. The default one-trajectory protocol is
package consumption only, not the fixed16 behavior evaluation.

For the historical three-seed result-tree layout:

```bash
python scripts/summarize_umi_three_seeds.py --root path/to/evidence-root \
  --output new/three-seed-summary
python scripts/summarize_umi_native.py --root path/to/evidence-root \
  --summary-dir new/three-seed-summary
```

These preserve all 0/500/4000 nodes, invalid prefixes, training rows and native
termination strata. They validate and aggregate saved results; they do not
re-run historical source-code checks against a changed checkout. Add `--plot`
to the first command after installing `pip install -e '.[analysis]'` for plots.
The native command amends only the new summary directory.

## Validation on 2026-09-12

The maintained readers reproduced the saved seed 2 contact metrics and the
three-seed nodes, training bands and native summaries exactly. The maintained
export verifier completed two independent 17-second seed 2 original/exported
executions with exact saved-array parity. Failure tests reject equal but
incomplete rollouts and differing arrays.

The CPU empty-cache network installation completed with private Conda and pip
caches: official pinned UMI source, checkpoint and data archives were fetched;
the MJCF was generated and an official policy completed 17 seconds (849 records,
8.921 mm mean position error, 0.029082 rad orientation error, no inversion).
Isolated imports and `pip check` passed. An inherited ROS metadata warning in the
first pip invocation disappeared under isolated Python; no extra ROS dependency
was installed to mask it. Repeating the rollout under isolated Python produced
identical values, apart from input path spelling.

The data/checkpoint ZIP and URDF hashes match the previous inputs. Generated
config JSON differs in formatting but has equal values. This is one installation
condition, not a new fixed16 performance result. The initial candidate source
snapshot was preserved before synchronizing the latest split interfaces for
subsequent checks; old results retain their original identity.

The separate Lab empty-cache installation is still downloading its pinned
packages. Documentation/test-only Git LFS media stalled during checkout; that
attempt and exit 128 were retained, and the same official source SHA was restored
with LFS smudging disabled. No old package cache was substituted and no Lab source
was patched. Final logs and command/status records live under
`outputs/release-reproduction-20260912/`. Existing environments and caches remain
intact. The source snapshot is a local candidate, not a fetched public PawCerto
release, because publication is still pending.
