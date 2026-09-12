# Public reproduction and candidate release

The UMI research reproduction stage is complete. This research preview provides
source, own-trained policy exports and measured evaluation results. The
[split-trained result](umi-source-inertia-result.md) has tracking close to the
earlier baseline and 14/15 complete unseen test trajectories; ID 1 inverted,
struck its head and triggered a numerical error. The historical three-seed
full-pool results also retain their head-contact and author-inversion cases.
These are disclosed research limitations, not a block on research publication.
Package parity and installation checks do not establish universal stability.

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
a source checkout. Download own-trained policies and selected evidence from
[research preview v0.1.0-research.1](https://github.com/AoiOTA/PawCerto/releases/tag/v0.1.0-research.1). The source tag and
artifact README identify the matching version and external inputs. For native
checkpoint evaluation on another machine, pass `--usd-path` to a locally
generated USD instead of the historical path recorded in the checkpoint.

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
Pass the original configuration used to generate each saved record to the
coordination reader. These readers preserve the original 1 N and 20 ms sampling definitions: sampled force
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

The Lab fresh online installation completed with initially empty private Conda
and uv caches. Isolated CPU checks passed, CUDA remained uninitialized, and all
275 distribution versions exactly match the existing Conda environment. `uv pip
check` still reports the same nine known metadata incompatibilities; no extra
package or source patch was introduced to hide them.

Two transport interruptions are retained: documentation/test LFS smudging stalled
(the checkout was restored at the same source SHA with smudging disabled), and
uv's shared PyPI connection delivered about 85.6 KB/s. The latter was interrupted
with exit 130. Only 39 incomplete wheels were recovered from their original
official URLs using bounded HTTP/1.1 ranges; all 39 complete SHA256 hashes passed.
The final sync used a transport-only requirements copy and the completed entries
in the newly populated cache. This is fresh online acquisition with recovery,
not an uninterrupted default-command claim. No previous package cache was used.

The final runtime candidate is source commit `a380b39`. Its original robot USD
conversion, first training update/save, and separate-process resume all exited 0
(17.98, 129.14 and 20.36 seconds respectively). With 16 environments × 24 steps,
`model_1.pt`, `model_2.pt` and `model_3.pt` retain 384, 768 and 1152 cumulative
transitions. Each checkpoint has the same 71-ID training selection and manifest
hash, finite model tensors, optimizer/environment state, and CPU uint8 RNG state.
The resumed metrics start at iteration 2 and are finite.

The intended resume endpoint was iteration 2. The validation driver mistakenly
passed `--iterations 2`, which means two **additional** updates; the actual endpoint
is 3. The extra update and original command are retained, with no further training.
Use `--iterations 1` to resume one update. This bounded check demonstrates real
update/save/resume consumption, not uninterrupted-trajectory equivalence, learning
or test-set quality. Detailed checkpoint hashes and assertions are in
`lab-runtime-validation.json`; total new disk usage is about 35 GiB, below the
100 GiB assignment bound.

Actual commands, exit statuses, source identities, network recovery and package
inventories are retained in `outputs/release-reproduction-20260912/`. Existing
environments and caches remain intact. The source is a local candidate, not a
fetched public PawCerto release. Empty-cache checks are parallel release work;
routine research uses the already verified environments.

## UniFP and Learning Force Control source package, 2026-09-13

Source commit `b4379a55b96d0a706bd6770528eff8ce448b8e4d` was archived from Git,
built into a wheel without dependency resolution, and unpacked outside the
checkout. An isolated Python process imported seven new UniFP/LFC modules
from that unpacked wheel and consumed the existing exported policies. Required
method license files and UniFP runtime/training sources were present.

The wheel consumer ran all 500 saved UniFP Lab input histories; its maximum
action difference from saved Lab outputs was 4.172325e-7, within the recorded
tolerance. LFC consumed three zero histories and returned finite 19-value
actions using the explicitly local initialized policy. This does not make it
an author-trained checkpoint. Both build and consumption exited 0.

The wheel contains 106 entries and is 263,668 bytes, SHA-256
`905f226a7216dd9f90cd385168600d7fe4e79b9f618ea7dce59f543f57a016c3`.
Commands, logs and validation are under
`outputs/framework-package-unifp-lfc-20260913/`. This is exact-commit source
packaging and CPU consumption evidence. It is not a fresh Lab installation,
new physical validation, release publication or completion of the remaining
methods. The separate UniFP report retains the roll termination and the
training/evaluation physical-property difference.
