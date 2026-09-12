# PawCerto

PawCerto aims to be an open-source reinforcement-learning framework for whole-body control of quadrupedal manipulators, usable by the wider research community across multiple methods and robots. **The currently implemented route is UMI-on-Legs on the original Go2 + ARX5**: train with Isaac Lab / PhysX, evaluate fixed weights independently in MuJoCo, and export a CPU TorchScript actor with its execution configuration. RoboDuet on Go1 + ARX5 now also has policy, runtime, two-stage training, recovery and export entrypoints, with short integration evidence; its learning result remains unverified.

Start with the [researcher guide](docs/researcher-guide.md) for the implemented workflow and extension points. For the second method, see [RoboDuet training and export](docs/roboduet-training.md), [task semantics](docs/roboduet-task.md), and [runtime evidence](docs/roboduet-runtime.md). The [local research candidate and current-policy videos](docs/release-evidence.md) are review artifacts, not a public or behavior-qualified release.

The latest [corrected-inertia, split-trained candidate](docs/umi-source-inertia-result.md) completed 4000 updates but **failed behavior acceptance**: MuJoCo fixed16 tracking regressed and test ID 1 inverted, hit its head and triggered a numerical error (14/15 test cases complete). No public release follows.

The previous three independently initialized policies completed the fixed 4000-update recipe (393,216,000 environment transitions each). Their final MuJoCo evaluations completed **48/48 full 17-second cases**, with mean end-effector position errors of **10.169 / 12.300 / 12.499 mm**. All three have native Lab evaluations and separately verified CPU exports. These are training-pool results: seed 1 still exhibits head impact and two inverted author-protocol episodes. The current stage is behavior-quality improvement and release preparation, not completed reliable WBC. [Current results and limitations](docs/umi-current-result.md).

This result uses **body-speed-v3 solver-force reconstruction and an explicit joint-speed limit of 1000 rad/s**. The limit is a deliberate physical configuration change from the original URDF limits. The original robot, published trajectory pool, observation/control path, EMD formula and PPO are retained. Default installation uses official, unmodified Isaac Lab and PhysX simulation code; no physics-engine rebuild is required. Earlier isolated PhysX patches are historical experiments, not dependencies.

| Current evaluation, seed 2027 | Final seed 0 / iteration 4000 | Official `ours` |
|---|---:|---:|
| MuJoCo EE mean, all 16 full cases | 10.169 mm | 8.657 mm |
| MuJoCo orientation mean | 0.024536 rad | 0.025493 rad |
| MuJoCo inverted / invalid cases | 0 / 0 | 0 / 0 |
| Native Lab nominal16 EE mean | 12.910 mm | 9.940 mm |
| Lab author protocol, latest500 EE mean | 21.455 mm | 20.076 mm |
| Lab author timeout completion fraction | 95.0% | 97.8% |

These evaluations use the supplied training-trajectory pool, not an independent trajectory holdout. Author-protocol means include early terminations; timeout completion is a survival proxy, not task success. Residual nonfoot external contact remains: 395 sampled endpoints above 1 N versus 38 for the official reference. Exact Lab/Gym numerical equivalence is not an acceptance requirement; reconstructed-force residuals and near-zero EMD branch sensitivity remain documented differences. Learning, transfer, failure and contact behavior must still be measured honestly. See [contact quality](outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_seed2027/contact-quality-readout.md) and [saved-state coordination](outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_seed2027/coordination_readout.md).

The fixed [seed-1 and seed-2 repetitions](docs/umi-relaxed-seed-repetitions.md) are now complete. All three final4000 policies completed all16 MuJoCo cases without inversion or numerical failure. The [three-seed report](outputs/mujoco/umi_relaxed_body_speed_three_seed_summary/README.md) retains every0/500/4000 result, all training rows, native evaluation strata and final exports.

| Final training seed | MuJoCo EE mean / orientation | External nonfoot endpoints | Lab author latest500 EE mean | Timeout fraction | Inverted author episodes |
|---|---:|---:|---:|---:|---:|
| 0 | 10.169 mm /0.024536 rad | 395 | 21.455 mm | 95.0% | 0 |
| 1 | 12.300 mm /0.027440 rad | 1140 | 21.614 mm | 94.8% | 2 |
| 2 | 12.499 mm /0.024825 rad | 1509 | 23.760 mm | 91.2% | 0 |

Tracking improved in all three seeds, but no inversion on nominal16 does not establish contact quality or universal stability. Seed1 case6 includes two sampled `Head_lower` ground-contact endpoints with a maximum force norm779.108N; its maximum sampled nonfoot fraction of upward ground force reaches68.0593%. Seed1's two inverted author episodes remain in its26 early terminations; seed2 has44 early terminations. Intermediate inversions at500 are also retained (seed0 cases6/14, seed1 case14). Three seeds on the same trajectory pool do not establish an independent holdout or full WBC acceptance.

Only UMI-on-Legs currently has completed training and fixed-policy evaluation evidence. RoboDuet's original Go1 + ARX5 integration completed two actual updates and one resumed update, plus five-module export comparisons on saved real inputs. This verifies execution, recovery and export, not learning or uninterrupted physics recovery. DeepWBC, MLM, further robots, real hardware, ROS 2 and VLA remain later work. KISS My Agent supports development and is not a runtime dependency.

The approved [project roadmap](docs/project-plan.md) also includes UniFP and Learning Force Control for force/position compliance, Multi-critic Twist Tracking for velocity tracking and conflicting objectives once its materials are available, ReLIC as an optional research integration subject to its license restrictions, and LeggedManip_Lab as an engineering reference for multiple robot bodies. These are planned directions, not implemented methods; the current UMI scope and budget remain unchanged.

A subsequent source-URDF comparison found incorrect principal inertia axes in the imported UMI USD. The converter now writes source-derived inertia tensors using official USD APIs. The corrected asset passed native dynamics checks, and a fresh seed-0 run completed all 4000 updates on 71 training trajectories. **This candidate failed behavior acceptance**: fixed16 MuJoCo tracking regressed to 11.818 mm / 0.025356 rad, and the frozen-model test completed 14 of 15 trajectories; test ID 1 inverted, struck its head on the ground and triggered a numerical error. The older results above retain their original asset identity. See [the split-trained result and failure evidence](docs/umi-source-inertia-result.md).

## Inputs and environments

From the repository root:

```bash
python scripts/fetch_umi.py
```

This fetches the pinned [official UMI repository](https://github.com/real-stanford/umi-on-legs), original robot assets, published trajectory data and checkpoints. Local inputs in `third_party/`, `downloads/` and `reference/`, and generated `runs/` and `outputs/`, are not committed. Historical output links require the corresponding evidence archive; the public source alone does not contain trained bundles or raw runs. [Release reproduction status](docs/release-reproduction.md) lists what is actually available. Existing inputs are preserved. Hardware SDK submodules are not required for simulation. Attribution stays with copied method code; downloading upstream assets grants no additional redistribution rights.

The exercised training combination is **Isaac Lab `develop@412fb31b30ee605b4ffec4327436fc0fe53281d8`, Isaac Sim 6.1.0.0, Python 3.12 and PyTorch 2.11 / CUDA12.8**, using PhysX. This is a fixed development snapshot, not a stable release tag. The new default environment is Conda `pawcerto-lab-sim610`; source is `third_party/IsaacLab-develop-sim610`. The completed training runs used the preserved `.venvs/isaaclab-sim610` environment, as their execution records state. [Environment versions and installation evidence](docs/environment-versions.md) records dependencies, launcher setup and the preserved release fallback. Select the exercised `isaacsim_physx` backend. No PawWeaver code is imported.

For a new installation, follow the [Isaac Lab installation guide](docs/isaaclab-install.md) and its [portable pinned requirements](configs/isaaclab-sim610-requirements.txt). The guide creates a Conda environment and uses versioned, hash-checked `uv` only to install packages into its explicit interpreter. All274 required package versions match the original environment; actual Conda activation and CPU checks passed. A separate [actual Conda native comparison](outputs/isaac/conda-runtime-migration/validation.json) also completed with the same849 rows and all10 recorded fields exactly equal to the prior environment; a separate empty-cache online acquisition and installation has now passed with hash-verified transport recovery; see [fresh reproduction](docs/release-reproduction.md). Dependency metadata remains unclean: `uv pip check` reports nine incompatibilities, including upstream version overrides and missing `ipywidgets` / `onnxruntime-gpu`; the guide retains the full results.

Independent CPU evaluation now defaults to **Conda `pawcerto-mujoco`**, with Python3.12.14, Torch2.11.0+cpu, NumPy2.5.3, SciPy1.18.1 and **MuJoCo3.13.0**. Its [actual migration check](outputs/mujoco/conda-runtime-migration/README.md) completed the same17-second exported-policy condition with all seven arrays and all JSON fields exactly equal to the previous interpreter. The original fixed16 evaluations retain their actual `third_party/mujoco-3.13-probe/bin/python` provenance; old environments remain intact. Lab's separate Conda environment contains MuJoCo3.12 for its own pinned dependencies, so do not use it for independent3.13 evaluation.

For the installed local CPU environment:

```bash
conda activate pawcerto-mujoco
python scripts/build_mujoco.py
```

For a new environment on another machine, the corresponding package-selection commands are below. Skip creation/installation when using the already installed environment. The original migration reused cached packages. A separate empty-cache network installation now fetched the official source/data/checkpoints, built the MJCF and completed a 17-second official-policy execution; isolated package checks passed. This validates the CPU installation path, not a new performance benchmark. See [fresh reproduction](docs/release-reproduction.md).

```bash
conda create --name pawcerto-mujoco python=3.12.14
conda activate pawcerto-mujoco
python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[mujoco]' 'mujoco==3.13.0' 'numpy==2.5.3' 'scipy==1.18.1'
python scripts/build_mujoco.py
```

Rebuild the MJCF after relocating the workspace because mesh references are absolute. See the [MuJoCo implementation](pawcerto/mujoco/README.md) and [48-case version regression](outputs/mujoco/version-3.13-regression/README.md).

## Train the current physical variant in Isaac Lab

The current recipe uses 4096 environments ×24 rollout steps,64 PPO epochs ×four minibatches, eight previews,132/261 actor/critic inputs and a fixed4000-iteration endpoint (393,216,000 transitions). No `--weights` or `--resume` means random initialization. Training/evaluation default to reconstructed solver force for EMD; the original formula is unchanged. Missing or stale reconstruction fails explicitly. [Runtime and resume semantics](docs/umi-reconstructed-force-training.md) explains the saved contract.

Activate the installed Conda Lab environment from the repository root and run its interpreter directly:

```bash
export PAWCERTO_ROOT="$PWD"
conda activate pawcerto-lab-sim610
export PAWCERTO_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=yes
cd "$PAWCERTO_ROOT/third_party/IsaacLab-develop-sim610"
```

A new workspace first converts the original robot with `scripts/convert_umi_usd.py`; the script records its generated USD location. The measured original policy joint order is committed at `configs/umi_go2_arx5_joint_names.json`, so a new checkout does not need an Isaac Gym joint-order query.

```bash
"$PAWCERTO_PYTHON" \
  "$PAWCERTO_ROOT/scripts/convert_umi_usd.py" --viz none --device cuda:0
```

The following examples use a **new output name**, preserving existing experiment records. Generate the input from the released recipe and explicitly add the1000rad/s override; omitting it retains URDF limits:

```bash
cd "$PAWCERTO_ROOT"
python3 - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('reference/checkpoints/tossing/ours/config.json').read_text())
config['joint_velocity_limit_override_rad_s'] = 1000.0
out = Path('runs/umi_v3_example')
out.mkdir(parents=True, exist_ok=False)
(out / 'input-config.json').write_text(json.dumps(config, indent=2) + '\n')
PY
cd "$PAWCERTO_ROOT/third_party/IsaacLab-develop-sim610"
"$PAWCERTO_PYTHON" "$PAWCERTO_ROOT/scripts/train_umi.py" \
  --config "$PAWCERTO_ROOT/runs/umi_v3_example/input-config.json" \
  --joint-names "$PAWCERTO_ROOT/configs/umi_go2_arx5_joint_names.json" \
  --num-envs 4096 --iterations 4000 --save-every 100 --seed 0 \
  --force-signal reconstructed-solver \
  --output "$PAWCERTO_ROOT/runs/umi_v3_example" --viz none --device cuda:0
```

The new development launcher uses `--viz none`; the preserved release launcher uses `--headless`. The examples follow the exercised training entry; the completed seed0 additionally recorded native limits, initialization and exception state in its [actual execution](runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4096/execution.json) and [run validation](runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4096/run-validation.json). A completed iteration budget alone does not establish learning.

Author-style final evaluation uses250 environments and the latest500 completed episodes, retaining timeout and early-termination strata:

```bash
"$PAWCERTO_PYTHON" "$PAWCERTO_ROOT/scripts/eval_umi.py" \
  --checkpoint "$PAWCERTO_ROOT/runs/umi_v3_example/model_4000.pt" \
  --joint-names "$PAWCERTO_ROOT/configs/umi_go2_arx5_joint_names.json" \
  --seed 2027 --force-signal reconstructed-solver \
  --output "$PAWCERTO_ROOT/outputs/isaac/umi_v3_example_author" --viz none --device cuda:0
```

`scripts/run_umi_isaac.py` also executes frozen policies on nominal trajectories; [current seed0 results](docs/umi-relaxed-scratch-result.md) include its nominal16 comparison. Keep a single GPU operator for training/evaluation. W&B remains offline. The separate original Python3.8 Isaac Gym environment and released-policy reference run are documented in [reference runtime setup](docs/reference-runtime.md).

## Evaluate and export in MuJoCo

Run these commands from the repository root after `conda activate pawcerto-mujoco`, using its MuJoCo3.13 CPU interpreter. The evaluator reads actual MuJoCo state, builds UMI observations, infers actions and applies the original explicit PD and delays; it does not replay recorded joint positions.

For an official-policy execution, or a fixed16 evaluation of the example's final checkpoint:

```bash
python -m pawcerto.mujoco.evaluate \
  --checkpoint reference/checkpoints/tossing/ours \
  --trajectory reference/data/tossing.pkl --seconds 17 \
  --output outputs/mujoco/official_example.json

python -m pawcerto.mujoco.evaluate \
  --checkpoint runs/umi_v3_example/model_4000.pt \
  --trajectory reference/data/tossing.pkl --cases 16 --seed 2027 --seconds 17 \
  --output outputs/mujoco/umi_v3_example_final16

python scripts/export_umi.py \
  --checkpoint runs/umi_v3_example/model_4000.pt \
  --joint-names configs/umi_go2_arx5_joint_names.json \
  --output outputs/export/umi_v3_example_4000
```

The export contains `actor.ts`, `config.json` and `joint_names.json`. Observation assembly, pose history, action delays/scales and PD remain in the adapter/runtime. The verified [current seed0 export](outputs/export/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4000/README.md) can be consumed directly:

```bash
python -m pawcerto.mujoco.evaluate \
  --checkpoint outputs/export/umi_reconstructed_solver_relaxed_body_speed_scratch_seed0_4000 \
  --trajectory reference/data/tossing.pkl --seconds 17 --seed 0 \
  --output outputs/mujoco/scratch_v3_4000_deployment.json
```

That bundle passed exact actor-tensor/parity checks and two independent full17-second original/exported processes with all saved arrays identical. This historical `sample(1,0)` package test is separate from the complete `sample(16,2027)` performance evaluation. Every fixed repetition retains the preselected0/500/4000 nodes, all16 cases and all failures; no successful single case replaces the full group.

## Method details and experiment history

- Policy joint order is FL, FR, RL, RR, then six arm joints; simulator arrays are mapped by name.
- State observations follow the original constructor's sorted keys: joint position, joint velocity, projected gravity, angular velocity. JSON insertion order differs.
- Targets remain in the task/world frame; the actor sees them relative to the delayed end-effector pose, with preview positions before rotation6D values.
- The200Hz trajectory/physics clock and50Hz policy clock, reset zero-action step, pose history and per-joint action delays are preserved.
- `ours` and `ours-real` differ in observation dimensions, previews and execution settings; each checkpoint uses its own configuration.

Tests support geometry, PD/delay, observations/actor parity, PPO and resume mechanisms. They do not replace measured control behavior. [Public-API force reconstruction](docs/umi-stock-solver-force.md) and [body-speed evidence](outputs/diagnostics/umi-body-speed-damping/candidate-readout.md) retain the remaining numerical differences without requiring exact Lab/Gym equality for acceptance.

Earlier results remain accessible with their original conditions and commands:

- [Normal-contact training stage](docs/umi-scratch-stage1.md), [checkpoint selection](docs/umi-checkpoint-selection.md) and [independent historical seed](docs/umi-repeated-seeds.md): selected seed0 / fixed seed1 iteration3500 policies achieved14.585 /15.782mm MuJoCo EE means on all16 cases with no inversion. Those runs precede current physics fixes and use a different EMD input; they are not current-variant repetitions. [Historical seed0 export](outputs/export/umi_lab_scratch_3500) and [seed1 export](outputs/export/umi_lab_scratch_seed1_3500/README.md) preserve package execution.
- [Stock-property regression](docs/umi-stock-properties-regression.md), [reconstructed-force4000 result](docs/umi-reconstructed-learning-result.md) and [weights-only adaptation](docs/umi-relaxed-velocity-adaptation.md): failed transfer and intermediate improvements remain documented. The URDF-limit final run had15 complete cases (five inverted) and one numerical failure; it did not trigger additional seeds.
- [Evaluation definitions](docs/umi-evaluation.md), [original reference setup](docs/reference-runtime.md) and [status/history](docs/status.md) retain protocols, commands and earlier execution failures. Historical comparison videos are [iteration3500](outputs/videos/scratch3500_vs_official_case0.mp4), [iteration500](outputs/videos/scratch500_vs_official_case0.mp4) and [body-speed adaptation](outputs/videos/umi_relaxed_body_speed_adapt_seed0_case0_seed2027/README.md); each illustrates one preselected training-pool case.
