# Installing the selected Isaac Lab environment

PawCerto currently selects the official, unmodified Isaac Lab development snapshot
`412fb31b30ee605b4ffec4327436fc0fe53281d8` with Isaac Sim `6.1.0.0`,
Python 3.12 and Torch `2.11.0+cu128`. This is a fixed development snapshot,
not a stable Lab release. No PhysX rebuild is required.

This recipe targets Linux x86_64; the local check used Ubuntu 24.04 and Python
3.12.14. Other platforms are not covered by this installation validation, even
where upstream dependency markers mention them. See
[environment versions](environment-versions.md) for the historical environments
and separate simulation evidence.

## Install into a new Conda environment

Conda is the default route here. A venv is **not required by Isaac Lab**: the
fixed official [installation instructions](https://github.com/isaac-sim/IsaacLab/blob/412fb31b30ee605b4ffec4327436fc0fe53281d8/docs/source/setup/installation/index.rst)
also support creating a Conda environment with Python 3.12. uv is used below only
as a package installer; it does not create or select a venv.

Run from your PawCerto checkout. If this machine already has the verified
`pawcerto-lab-sim610` environment, activate it and skip creation and synchronization
when you only want to run it. For a new installation, use a new environment name; `uv pip sync`
removes packages outside the supplied requirements, so do not target an active
training or general-purpose environment.

```bash
export PAWCERTO_ROOT="$(pwd -P)"
export PAWCERTO_LAB="$PAWCERTO_ROOT/third_party/IsaacLab-develop-sim610"
# Use a new name if this environment already exists.
conda env list
conda create --name pawcerto-lab-sim610 python=3.12.14
conda activate pawcerto-lab-sim610
export PAWCERTO_PYTHON="$CONDA_PREFIX/bin/python"
export UV_CACHE_DIR="$PAWCERTO_ROOT/.cache/isaaclab-sim610"
mkdir -p "$PAWCERTO_ROOT/third_party" "$PAWCERTO_ROOT/.tools/uv-0.12.13"

# For a new Lab checkout. Keep an existing checkout intact.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/isaac-sim/IsaacLab.git "$PAWCERTO_LAB"
GIT_LFS_SKIP_SMUDGE=1 git -C "$PAWCERTO_LAB" checkout --detach 412fb31b30ee605b4ffec4327436fc0fe53281d8

curl --fail --location \
  https://github.com/astral-sh/uv/releases/download/0.12.13/uv-x86_64-unknown-linux-gnu.tar.gz \
  --output "$PAWCERTO_ROOT/.tools/uv-0.12.13/uv.tar.gz"
(cd "$PAWCERTO_ROOT/.tools/uv-0.12.13" && \
  echo '745765a3b6e360ad76743599ae5c42e9278c7edf8bbff9fc76d05bf2623a04dd  uv.tar.gz' | sha256sum --check && \
  tar -xzf uv.tar.gz)
export PAWCERTO_UV="$PAWCERTO_ROOT/.tools/uv-0.12.13/uv-x86_64-unknown-linux-gnu/uv"

cd "$PAWCERTO_LAB"
"$PAWCERTO_UV" pip sync --python "$PAWCERTO_PYTHON" \
  --index https://pypi.nvidia.com \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  "$PAWCERTO_ROOT/configs/isaaclab-sim610-requirements.txt"
```

The clone leaves Git LFS documentation media and upstream test assets as pointers;
the PawCerto Go2+ARX5 path fetches its robot separately and does not consume these
files. If running upstream tests or building its full documentation, fetch the
required LFS assets separately. No source or physics implementation is patched.

Execute these commands in order and stop at any nonzero exit. Check free space
before installation; this environment contains large NVIDIA/CUDA packages, and
cache plus environment can occupy tens of GiB. The artifact URLs require normal
HTTPS access to GitHub, PyPI, NVIDIA and PyTorch hosts. Access restrictions and
vendor license terms still apply.

The [requirements](../configs/isaaclab-sim610-requirements.txt) preserve the
frozen third-party versions and platform selection. For the 214 public wheels
selected in the original installation they use official `uv.lock` URLs and one
exact lock hash per wheel, replacing the original machine's absolute `file://`
paths. The remaining requirements retain their original frozen hashes. The 12
editable entries intentionally resolve from **the Lab checkout as the working
directory**, and obtain their versions from that fixed source.

Do not replace this command with `uv sync --locked`: this snapshot has nine
stale editable-package versions in its upstream lock. Neither `uv.lock` nor
`pyproject.toml` needs patching. The prior `uv sync --frozen` attempt was
interrupted by network failure, so it is not the demonstrated installation path.
For this Conda route, run the installed interpreter directly. In the
[README](../README.md) conversion, training and evaluation examples, use
`"$PAWCERTO_PYTHON"` in place of the entire legacy prefix
`"$PAWCERTO_UV" run --no-sync --extra isaacsim python`, retaining the same script
path and arguments. Those scripts import the installed editable Lab and construct
its `AppLauncher` themselves; they do not require uv to launch Python. This choice
cannot silently select the historical `.venvs/isaaclab-sim610` environment.

For a later shell session:

```bash
conda activate pawcerto-lab-sim610
export PAWCERTO_PYTHON="$CONDA_PREFIX/bin/python"
"$PAWCERTO_PYTHON" -I -c 'import sys; print(sys.executable)'
# Also set PAWCERTO_ROOT and PAWCERTO_LAB to this checkout before using script paths.
```

Keep the legacy venv launcher block only for reproducing the historical run.
Conda activation alone does not override a README block that explicitly sets
`UV_PROJECT_ENVIRONMENT` to the old venv; avoid using that block with this route.
Training still requires the separate PawCerto assets, simulator license acceptance
and launch arguments in the README. The separate [native Conda execution](../outputs/isaac/conda-runtime-migration/validation.json)
has now completed: direct Conda Python ran the existing final seed2 nominal16
condition with exit 0, and all 849 rows and ten recorded fields exactly matched
the historical environment. This is separate from the CPU-only installation check.

## CPU-only installation check

This reads package metadata and exercises a small CPU import/tensor path. It does
not start Isaac Sim, validate a GPU driver, accept the simulator EULA, or establish
simulation/reward correctness.

```bash
CUDA_VISIBLE_DEVICES='' \
  "$PAWCERTO_PYTHON" -I -B - <<'PY'
import importlib.metadata as md
import importlib.util
import numpy as np
import torch
import isaaclab
for name, expected in {
    'isaaclab': '24.2.0', 'isaaclab-physx': '7.1.2',
    'isaacsim': '6.1.0.0', 'torch': '2.11.0+cu128', 'warp-lang': '1.17.0',
}.items():
    actual = md.version(name)
    assert actual == expected, (name, actual, expected)
    print(name, actual)
assert importlib.util.find_spec('isaacsim') is not None
assert torch.tensor(np.array([1., 2.]), device='cpu').sum().item() == 3.
assert not torch.cuda.is_initialized()
print('CPU package check passed; no CUDA context initialized')
PY
```

## Validation scope and local cached replay

The default Conda route was exercised on 2026-09-12 in the new environment
`/home/lyb/miniconda3/envs/pawcerto-lab-sim610`. Conda supplied Python **3.12.14**;
`uv pip sync --python <that-prefix>/bin/python` installed all **274 required
packages**, exit 0. After an actual `conda activate pawcerto-lab-sim610`, the CPU
check exited 0 and verified both `sys.prefix` and `sys.executable` point to this
Conda environment. Every required package version matches the original inventory;
the only extra Python distribution is Conda's `pip=26.2.1` (275 total).
CUDA remained uninitialized. The environment occupies approximately 29 GiB.

Conda creates the interpreter and native base packages; the requirements then
control the Python runtime packages, replacing Conda's bootstrap packaging and
setuptools versions where necessary. The local creation reused cached Conda
packages from defaults and conda-forge in offline mode; the exact records and
commands are retained in [Conda replay evidence](../outputs/isaac/lab-install-reproducibility/conda/result.json).
A new machine's Conda base-library selection can differ. Both Conda creation and
Python dependency installation reused existing read-only cache inputs, with a
private writable cache and networking disabled. This does not verify an
empty-cache online installation. The later native comparison linked above
verifies the exercised frozen-policy condition; it does not cover every simulator
consumer or repair the known dependency metadata issues.


The earlier venv check remains retained as historical evidence. On 2026-09-12, installation into the initially empty
`.venvs/isaaclab-reinstall-check` succeeded: **274 packages installed, exit 0**.
All 274 distribution versions exactly match the prior installation inventory.
The CPU check above passed with `torch.cuda.is_initialized() == False`.
The copy-mode environment occupies approximately 29 GiB. Python `-I -B` avoids
inherited `PYTHONPATH` packages (the first inventory attempt picked up unrelated
ROS packages) and avoids writing bytecode into the editable checkout.

**Dependency metadata validation is not clean:** the earlier venv
`uv pip check --no-cache` returned exit 1 with nine incompatibilities. The same
274 runtime versions were retained in Conda; those known issues were not repaired
or re-audited in the Conda check:

| Installed consumer | Unmet metadata requirement | Installed selection |
|---|---|---|
| mujoco-usd-converter | MuJoCo >=3.11.0,<3.12 | 3.12.0 |
| numba | NumPy >=1.22,<2.5 | 2.5.1 |
| isaacsim-core | MuJoCo ==3.11.0 | 3.12.0 |
| isaacsim-core | mujoco-warp ==3.11.0 | 3.12.0 |
| isaacsim-core | newton[sim] ==1.5.0 | 1.6.0rc1 |
| cmeel-boost | NumPy >=2.3,<2.4 on Python >=3.11 | 2.5.1 |
| isaacsim-kernel | NumPy ==2.3.1 | 2.5.1 |
| open3d | ipywidgets >=8.0.4 | absent |
| isaacsim-robot | onnxruntime-gpu ==1.26.0 | absent |

The fixed upstream `pyproject.toml` explicitly overrides NumPy, MuJoCo,
MuJoCo Warp and Newton requirements; that explains the selected versions but
is not evidence that every affected consumer works. The two missing packages
remain unresolved in this restored frozen selection. This task did not change
the lock or add dependencies to the running experiment. Installation and the
small CPU check therefore establish restoration of the selected package set,
not general dependency consistency or complete simulator functionality.


A newly created Conda environment (or a clean venv) with reused, hash-checked local wheels and populated
cache is different from an empty-cache installation on a new machine. The latter
network route has **not been demonstrated**. The portable requirements remove
absolute machine paths but do not turn the existing 214-wheel directory into a
complete offline bundle: NVIDIA/Torch and other cached artifacts are still needed.

The local replay replaces only official public-wheel URLs with the matching
wheel filenames in an existing wheel directory; all versions, markers, editable
entries and hashes remain identical. To produce that transport-only input:

```bash
export PAWCERTO_WHEELHOUSE="$PAWCERTO_ROOT/outputs/isaac/lab-develop-sim610-validation/wheelhouse"
export PAWCERTO_REPLAY="$PAWCERTO_ROOT/outputs/isaac/lab-install-reproducibility"
mkdir -p "$PAWCERTO_REPLAY"
python3 - <<'PY'
import os, re
from pathlib import Path
from urllib.parse import unquote, urlsplit
root = Path(os.environ['PAWCERTO_ROOT'])
wheelhouse = Path(os.environ['PAWCERTO_WHEELHOUSE']).resolve()
text = (root / 'configs/isaaclab-sim610-requirements.txt').read_text()
def local(match):
    wheel = wheelhouse / unquote(urlsplit(match.group()).path).split('/')[-1]
    assert wheel.is_file(), wheel
    return wheel.as_uri()
text, count = re.subn(r'https://files\.pythonhosted\.org/[^\s]+\.whl', local, text)
assert count == 214, count
(Path(os.environ['PAWCERTO_REPLAY']) / 'requirements-local.txt').write_text(text)
PY
```

Pass this derived file to the same `uv pip sync` command with `--offline`, a new
Conda environment and a suitable populated cache. Do not share a writable
cache with active work. Our replay used private cache metadata and read-only
bind mounts of existing archive directories, with network access disabled and
`--link-mode copy`. The original Lab checkout, environment and cache were
mounted read-only. This resource isolation is specific to the local validation;
it is not an additional installation dependency for researchers.

Actual commands, exit statuses and checks are retained in
[the replay evidence directory](../outputs/isaac/lab-install-reproducibility/).
The initial missing-index failure established why the NVIDIA/PyTorch indexes are
explicit above. A symlink-based archive reuse attempt also failed in uv's copy
step; directory bind mounts replaced those symlinks. Both failures are retained.
