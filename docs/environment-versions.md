# Environment versions checked on 2026-09-12

The selected corrected-training environment is the isolated **Isaac Lab `develop@412fb31b30ee605b4ffec4327436fc0fe53281d8` + Isaac Sim 6.1.0.0**, using PhysX. Independent evaluation uses MuJoCo 3.13.0. These selections followed actual validation below; they do not imply a stable Lab release or proven learning. Existing environments remain intact for historical reproduction and latest-release fallback.

## Current default: Conda

At the user's request, future PawCerto execution defaults to **Conda `pawcerto-lab-sim610`** for Lab and **Conda `pawcerto-mujoco`** for independent MuJoCo 3.13. Both environments were actually created and activated; the README now calls their Python interpreters directly, without a `uv run` prefix that could select the historical venv. uv is only a package installation tool in the new Lab instructions. The completed three training chains and original final evaluations retain their historical environment records.

The [Lab Conda native comparison](../outputs/isaac/conda-runtime-migration/validation.json) ran the same final seed-2 nominal16 condition with exit 0: all 849 rows and ten recorded fields exactly match the old environment. The [MuJoCo Conda comparison](../outputs/mujoco/conda-runtime-migration/validation.json) ran the existing seed-0 exported policy's separate 17-second package condition with exit 0: all seven arrays and all result JSON fields match the previous interpreter. These establish actual consumption in the migrated environments, not new training or replacement performance evaluations. See the [Conda Lab installation guide](isaaclab-install.md) and [MuJoCo usage](../pawcerto/mujoco/README.md).

A separate 2026-09-12 CPU empty-cache installation used fresh Conda/pip caches and official network inputs. It built the original MJCF and completed the 17-second `sample(1,0)` official-policy condition (849 records, no inversion). Isolated imports resolved into the new environment, and isolated `pip check` passed. This is installation validation; the published-data source hashes match the earlier inputs. [Details and candidate-source boundary](release-reproduction.md).

Lab's 274 required runtime package versions match the original environment; its separate MuJoCo 3.12 dependency does not replace the independent evaluator's 3.13.0. The nine known Lab dependency metadata incompatibilities and empty-cache network installation limitation remain documented. Neither old environments nor previous experiment records were deleted or rewritten.

## Preserved release environment

| Component | Actual local version | Official release check |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS, x86_64 | Isaac Sim 6.1 lists Ubuntu 22.04/24.04 support. |
| Isaac Sim | Python package 6.0.1.0; build `6.0.1-rc.7+release.42383.32955d8d.gl` | Download page lists 6.1.0, September 2026. |
| Isaac Lab | Git tag `v3.0.0-beta2.patch1`, commit `ffff603eafc6b74264a5261cc0183d6a65390d78` | GitHub Latest points to the same Beta-series tag. It explicitly adds Sim 6.0.1 compatibility. |
| Lab component packages | `isaaclab=6.1.14`, `isaaclab-physx=1.1.3`, `isaaclab-ov=0.4.2` | Component package numbering is distinct from the Lab product release tag. |
| MuJoCo | 3.8.0 in both current training and CPU evaluation environments | 3.13.0 is the latest published release, September 2026. |
| PyTorch | Training: 2.11.0+cu128; CPU evaluation: 2.11.0+cpu | Latest Torch compatibility was not audited in this check. |

The preserved training environment is `/home/lyb/miniconda3/envs/pawweaver-train`; historical CPU evaluation used `/home/lyb/miniconda3/envs/pawweaver-runtime`. Lab is an editable checkout at `/home/lyb/pawweaver/.deps/IsaacLab`. These paths reflect this machine, not portable installation requirements.

The Lab checkout has two existing installation-metadata adjustments: `source/isaaclab/setup.py` pins coverage to 7.4.4 instead of 7.6.1, and `source/isaaclab_rl/setup.py` pins packaging to 26.0 instead of `<24`; two corresponding local changelog files are untracked. No Lab simulation-source changes appear in the current Git diff. The active PhysX GPU library is the installed official `omni.physx.gpu-110.1.13` library. Thus “stock physics” here describes the simulation code and engine library; it must not be used to claim that the entire editable checkout is pristine.

## Relevance of newer versions

- **Confirmed upstream fix:** [Isaac Lab PR #6377](https://github.com/isaac-sim/IsaacLab/pull/6377), merged into `develop` on July 17, fixes nested rigid-body/mass properties stopping below the first body. The installed tag predates it. PawCerto currently applies properties per physical link through the official API; it does not patch installed Lab code.
- **MuJoCo comparison completed, local causality unproved:** 3.13 restores protection for non-positive pivots in sparse inertia factorization and the associated warning. In an isolated fixed-input comparison, the saved iteration-4000 case 3 completed 17 seconds with no warning or automatic reset under 3.13, whereas 3.8 failed at 16.88 seconds. The policy still inverted and had 471.934 mm mean EE error; numerical completion did not repair control. The official policy's same-case arrays were identical across versions. Differences in the learned policy began at 7.30 seconds, and the specific upstream fix is not established as the cause. See the [two-case comparison](../outputs/mujoco/version-3.13-check/README.md). Both original environments remain on 3.8; this is not a full upgrade regression.
- **Original sensor still a separate issue:** latest Sim 6.1 articulation-sensor documentation describes incoming and projected joint force. The checked release notes do not establish restoration of the old forward-dynamics/solver sensor flags. Updating versions cannot yet be treated as a demonstrated fix for the reward-signal mismatch.
- **Compatibility must be checked as a combination:** a newer standalone Sim release does not establish compatibility with a particular Lab tag or identical behavior of old models.

Official sources: [Sim downloads](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html), [Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html), [Lab release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1), [MuJoCo release](https://github.com/google-deepmind/mujoco/releases/tag/3.13.0), [Sim articulation sensors](https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_physics_articulation_force.html).

## Isolated upgrade validation

Following the user's request to pursue newer versions, the selected candidate is **Lab `develop@412fb31b30ee605b4ffec4327436fc0fe53281d8` plus Sim `6.1.0.0`**. This is a fixed development-branch snapshot, not a newly released stable Lab tag. Its [README](https://github.com/isaac-sim/IsaacLab/blob/412fb31b30ee605b4ffec4327436fc0fe53281d8/README.md) explicitly targets Sim 6.1; its [locked project configuration](https://github.com/isaac-sim/IsaacLab/blob/412fb31b30ee605b4ffec4327436fc0fe53281d8/pyproject.toml) specifies Python 3.12, Torch 2.11, Warp 1.17 and USD Exchange 2.3. The candidate includes the nested-property fix. Its entry must explicitly select `isaacsim_physx`, since the development branch's defaults differ from the current environment.

Installation and [native validation](../outputs/isaac/lab-develop-sim610-validation/result.md) are complete. Source is at `third_party/IsaacLab-develop-sim610`; the independent environment is `.venvs/isaaclab-sim610`. The official checkout is clean. `uv sync --locked --extra isaacsim` exposed nine stale editable-package versions in the upstream lock, so installation retained the locked third-party versions after separately checking dependency constraints. All 214 downloaded wheel hashes matched the official lock; 274 installed distributions matched the lock or actual editable source. The final installation used offline `uv pip sync`; execution uses `uv run --no-sync --extra isaacsim`. See the retained installation and dependency-comparison records in the validation directory.

Actual packages include `isaaclab=24.2.0`, `isaaclab-physx=7.1.2`, `isaacsim=6.1.0.0`, Torch `2.11.0+cu128` and Warp `1.17.0`. Component version numbers are not Lab product-release numbers. The first native attempt exposed a changed external-force default; explicitly retaining `enable_external_forces_every_iteration=False` restored the supported branch. The new launcher uses `--viz none`, having removed `--headless`. Neither adjustment changes the original reward or engine source.

### Reusing the installed Lab environment

The [installation guide](isaaclab-install.md) now provides fixed-source/tool acquisition and [portable requirements](../configs/isaaclab-sim610-requirements.txt). A new empty virtual environment was actually restored from hash-checked wheels and populated cache: 274 packages, exit 0, all versions matching the prior inventory, and a passing CPU check without CUDA initialization. A successful empty-cache network installation on a new machine has **not** been demonstrated. Independent dependency metadata checking returned nine incompatibilities, including seven selected-version conflicts associated with upstream overrides and two missing packages; the guide records these without claiming general dependency consistency. The original running environment, cache and Lab source remained read-only during this validation.

The README training commands can still reuse this workspace's installed environment. The historical `uv sync --locked --extra isaacsim` failed on editable-package metadata; the subsequent `uv sync --frozen --extra isaacsim` was interrupted by network failures. Neither is a verified one-command installation recipe. The successful historical recovery installed 274 packages using downloaded wheels **and previously populated uv cache**.

The fixed source is [IsaacLab at 412fb31](https://github.com/isaac-sim/IsaacLab/tree/412fb31b30ee605b4ffec4327436fc0fe53281d8). The exercised tool is [uv 0.12.13 for Linux x86_64](https://github.com/astral-sh/uv/releases/download/0.12.13/uv-x86_64-unknown-linux-gnu.tar.gz); its archive SHA-256 is `745765a3b6e360ad76743599ae5c42e9278c7edf8bbff9fc76d05bf2623a04dd`. Current checkout, lock and tool hashes were compared with the [installation record](../outputs/isaac/lab-develop-sim610-validation/installation.json). The commands below express the recorded recovery with explicit paths; they were not rerun during documentation repair and should not be used to change an environment while training is active.

```bash
# First set PAWCERTO_ROOT, PAWCERTO_UV and UV_* as in the README.
PAWCERTO_INSTALL_RECORD="$PAWCERTO_ROOT/outputs/isaac/lab-develop-sim610-validation"
cd "$PAWCERTO_ROOT/third_party/IsaacLab-develop-sim610"

"$PAWCERTO_UV" export --frozen --extra isaacsim --no-dev \
  --format requirements-txt \
  --output-file "$PAWCERTO_INSTALL_RECORD/requirements-frozen.txt"

"$PAWCERTO_UV" pip sync \
  --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  --offline --find-links "$PAWCERTO_INSTALL_RECORD/wheelhouse" \
  "$PAWCERTO_INSTALL_RECORD/requirements-frozen-local-exact-hashes.txt"
```

The export produces `requirements-frozen.txt`; it does **not** generate the derived `requirements-frozen-local-exact-hashes.txt`. The latter already exists in this research workspace, maps 214 selected wheels to absolute local `file://` URLs, and retains each selected wheel's official lock hash. Other dependencies, including NVIDIA/Torch packages, still require the existing `UV_CACHE_DIR`; the wheel directory is not a complete offline bundle. Its 12 editable entries use `./source/...`, so synchronization must run from the Lab checkout. Local absolute paths also require adaptation after relocating this workspace; the new portable requirements instead use the matching official artifact URLs. See the [historical synchronization log](../outputs/isaac/lab-develop-sim610-validation/uv-pip-sync-exact-hashes.log), [selected-dependency comparison](../outputs/isaac/lab-develop-sim610-validation/selected-dependency-comparison.json), and [new empty-venv restoration result](../outputs/isaac/lab-install-reproducibility/result.json). Empty-cache network installation and the reported dependency metadata incompatibilities remain limitations, separate from the completed training and evaluation in the original environment.

### Executed validation

With PawCerto's per-link compatibility calls omitted in the candidate, the upstream helper applied all eight checked properties on all 25 bodies. Public C/acceleration interfaces, legacy friction and reward freshness passed; all 450 components in three free-dynamics samples passed the same tolerance. A real 4096-environment original-PPO iteration completed 98,304 transitions with 256 Adam updates and finite data; all non-timing metrics matched the release smoke run. Its 2.056-second iteration and 5814 MiB sampled peak are single-run measurements, not general performance claims. Independent review found no additional issue. Main code retains the idempotent per-link calls for release fallback and explicitly records its own provenance for long training.

The [isolated MuJoCo 3.13 regression](../outputs/mujoco/version-3.13-regression/README.md) completed official `ours` and both historical selected training policies on all 16 fixed `sample(16,2027)` cases each. All 48 actual processes exited 0 with full 17-second records, no warning/reset/inversion, small tracking differences and identical sampled ground-support/contact counts. On this evidence, **3.13.0 is the explicit baseline for the corrected training experiment's independent evaluation**, using `third_party/mujoco-3.13-probe`. Eight existing asset/contact/failure tests also passed, including an actual induced QACC reset propagated as a failure. Existing 3.8 environments and historical results remain unchanged.

The user clarified that fallback means the latest published **release, including beta**, relative to `develop`; it does not require an older non-beta series. If substantive training, physics-interface or reward-semantics incompatibility emerges, return to the already exercised `v3.0.0-beta2.patch1` + Sim 6.0.1 combination instead of accumulating development-branch repairs. A fallback must preserve run identity and explicitly account for the engine change.
