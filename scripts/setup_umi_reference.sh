#!/usr/bin/env bash
# Isolated original UMI-on-Legs runtime. Does not run simulation or touch other envs.
set -euo pipefail
# Do not inherit the host ROS Python packages into this Python 3.8 environment.
unset PYTHONPATH
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
conda_bin=${CONDA_EXE:-$(command -v conda || true)}
if [[ -z "$conda_bin" ]] || ! command -v "$conda_bin" >/dev/null 2>&1; then
  echo "Conda was not found. Install Conda and set CONDA_EXE or put conda on PATH." >&2
  exit 1
fi
if [[ -n "${PAWCERTO_UMI_ENV:-}" ]]; then
  prefix=$PAWCERTO_UMI_ENV
else
  conda_base=$("$conda_bin" info --base)
  if [[ -z "$conda_base" ]]; then
    echo "Conda returned an empty base path; set PAWCERTO_UMI_ENV explicitly." >&2
    exit 1
  fi
  prefix="$conda_base/envs/pawcerto-umi-reference"
fi
if [[ ! -x "$prefix/bin/python" ]]; then
  "$conda_bin" create -y -p "$prefix" python=3.8 pip --override-channels -c conda-forge
fi
mkdir -p "$root/downloads"
if [[ ! -d "$root/downloads/isaacgym/python" ]]; then
  if [[ ! -f "$root/downloads/IsaacGym_Preview_4_Package.tar.gz" ]]; then
    curl -fL --max-time 600 https://developer.nvidia.com/isaac-gym-preview-4 \
      -o "$root/downloads/IsaacGym_Preview_4_Package.tar.gz"
  fi
  tar -xzf "$root/downloads/IsaacGym_Preview_4_Package.tar.gz" -C "$root/downloads"
fi
"$conda_bin" install -y -p "$prefix" pytorch=2.1.0 torchvision=0.16.0 \
  pytorch-cuda=12.1 numpy=1.23.4 --override-channels -c pytorch -c nvidia -c conda-forge
"$conda_bin" install -y -p "$prefix" --freeze-installed \
  hydra-core=1.3.2 pydantic=1.10.13 scipy=1.10.1 matplotlib-base=3.7.3 \
  rich=12.6.0 wandb=0.15.12 imageio-ffmpeg=0.4.9 \
  zarr=2.16.1 numcodecs=0.12.1 transforms3d=0.4.1 gitpython ninja fvcore iopath \
  --override-channels -c pytorch3d -c pytorch -c nvidia -c conda-forge
p3d=pytorch3d-0.7.5-py38_cu121_pyt210.tar.bz2
if [[ ! -f "$root/downloads/$p3d" ]]; then
  curl -fL --max-time 120 "https://conda.anaconda.org/pytorch3d/linux-64/$p3d" \
    -o "$root/downloads/$p3d"
fi
# Official matching binary; runtime dependencies above are installed explicitly.
"$conda_bin" install -y -p "$prefix" --offline --no-deps "$root/downloads/$p3d"
"$prefix/bin/python" -m pip install imageio==2.31.6 fsspec==2023.10.0 ninja==1.11.1.1
"$prefix/bin/python" -m pip install --no-deps --no-build-isolation -e "$root/downloads/isaacgym/python" \
  -e "$root/third_party/umi-on-legs/mani-centric-wbc"
# Import and CLI parsing only. The Master owns subsequent GPU simulation.
export PATH="$prefix/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=""
export MAX_JOBS=2
"$prefix/bin/python" "$root/third_party/umi-on-legs/mani-centric-wbc/scripts/play.py" --help
