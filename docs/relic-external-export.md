# Optional external ReLIC actor consumption

`pawcerto.methods.relic.external_export` is an authored CPU adapter for an
explicitly supplied, trusted TorchScript or ONNX export. It consumes an already
packed observation and returns raw actor actions. It does not download anything,
build robot observations, simulate, train, or implement the complete controller.
The default wheel contains no upstream ReLIC code, assets, license, or weights.
ONNX consumption requires an existing `onnxruntime` installation; it is not a
default PawCerto dependency. Missing dependencies and model failures propagate.

## External provenance and license

The investigated upstream revision is
[`rai-opensource/relic@27f8033c5064d32f049a17accb71cd1091422878`](https://github.com/rai-opensource/relic/tree/27f8033c5064d32f049a17accb71cd1091422878).
Its actual [root license](https://github.com/rai-opensource/relic/blob/27f8033c5064d32f049a17accb71cd1091422878/LICENSE)
is the RAI Institute Research License, with noncommercial research use terms.
The README MIT badge does not describe that license. Keep external files and
their license together, outside the package. Local evidence is under the ignored
`artifacts/relic-external-cpu/` directory.

| External file | SHA-256 |
| --- | --- |
| [policy.pt](https://github.com/rai-opensource/relic/blob/27f8033c5064d32f049a17accb71cd1091422878/source/relic/relic/assets/spot/pretrained/policy.pt) | `14ed0614f507e1330a5c66af1615f27558b58b6c999443b42d1bb6c3d45ae3b4` |
| [policy.onnx](https://github.com/rai-opensource/relic/blob/27f8033c5064d32f049a17accb71cd1091422878/source/relic/relic/assets/spot/pretrained/policy.onnx) | `039b542d7e833961b7602b4f933c10415589aa1ff1fa1a8295979f6a20c2f000` |
| LICENSE | `0075bd6499d10cd336e89d43da1a07a089083d7c0f8a44d6e23ddb8d00015c3f` |

The upstream play path loads a full runner checkpoint and exports these formats;
it does not supply an exported-policy consumer. The pretrained export's training
stage, seed, and update count are unknown. PLAY inherits Phase1 configuration;
that inheritance does not identify the supplied weights' training stage.

## Input and output boundary

Both adapter paths require finite NumPy `float32` input of shape `(1, 84)` and
return finite `float32` output of shape `(1, 12)`. ONNX uses `obs` and `actions`,
fixed batch 1, opset 11. The inspected TorchScript actor has layers
84 → 512 ELU → 256 ELU → 128 ELU → 12, identity normalization, and no history.

The source-level observation slices (Python half-open indexing) are:

| Slice | Contents |
| --- | --- |
| `0:3` | Body-frame base linear velocity |
| `3:6` | Body-frame base angular velocity |
| `6:9` | Projected gravity |
| `9:12` | Base command |
| `12:34` | 22-dimensional command: arm 7, legs 12, torso roll/pitch/height 3 |
| `34:53` | 19 relative joint positions |
| `53:72` | 19 relative joint velocities |
| `72:84` | Last 12 raw actor actions |

Actual runtime joint ordering has not been verified. Callers must supply already
packed observations; this API cannot turn raw robot sensors into valid inputs.
The output is 12 raw leg actor values, not 19 joint targets. Upstream initially
scales actions by 0.2 and adds default positions, then overwrites the arm's seven
targets and the selected manipulation leg's three targets through command
interpolation. That downstream behavior is outside this consumer.

Compatibility between upstream Isaac Lab 2.1 / Isaac Sim 4.5 and the local
Isaac Lab 3 / Isaac Sim 6.1 stack is unverified. This CPU path uses neither.

## Run and observed evidence

From the repository root, using a Python environment with NumPy and Torch:

```bash
PYTHONPATH=. /path/to/python scripts/consume_relic_export.py \
  --policy /external/policy.pt \
  --observations /external/packed-observations.npy \
  --output /external/forward.json
```

Add `--compare-policy /external/policy.onnx` when ONNX Runtime is installed.
The JSON retains model/input hashes, raw outputs, maximum absolute difference,
and `allclose(rtol=1e-5, atol=1e-6)`. A failed comparison writes the evidence then
exits nonzero. Only load trusted TorchScript files, which are executable content.

On 2026-09-13 the existing `.venvs/isaaclab-sim610/bin/python` environment
(Torch 2.11.0+cu128, ONNX Runtime 1.26.0) ran both original exports on CPU, without
installation or GPU use. Three synthetic, finite, nonzero 84-value inputs were
used: a [-0.5, 0.5] ramp, a 0.3-amplitude sine sequence, and NumPy seed-0 uniform
[-0.5, 0.5] values. All three comparisons passed:

| Synthetic input | Maximum absolute PT/ONNX difference |
| --- | --- |
| ramp | `3.0994415283203125e-06` |
| sine | `2.6226043701171875e-06` |
| random_seed0 | `2.384185791015625e-06` |

The corresponding `.npy`, `.json`, and `.log` files plus `environment.json` are
retained in `artifacts/relic-external-cpu/`. These are ABI and numerical agreement
checks on three synthetic samples, not valid robot trajectories, whole-body
control evaluation, training reproduction, or proof of agreement on all inputs.

The authored adapter tests passed with:

```bash
PYTHONPATH=. /home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python \
  -m pytest -q tests/test_relic_external_export.py
```

Result: 6 passed; four TorchScript deprecation warnings. Tests cover malformed
input shape/type/nonfinite values, an explicit CPU scripted model, and rejection
of wrong output shape. The original pretrained weights are exercised separately
by the retained real forward artifacts, not vendored as test fixtures.
