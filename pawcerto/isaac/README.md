# Go2 + ARX5 in Isaac Lab

This adapter uses the original UMI-on-Legs URDF, released policy and task
semantics. The current environment is Conda `pawcerto-lab-sim610`, using the
fixed official Isaac Lab development snapshot
`412fb31b30ee605b4ffec4327436fc0fe53281d8` and Isaac Sim 6.1.0.0.
Follow the [installation guide](../../docs/isaaclab-install.md); no PawWeaver
installation or physics-engine rebuild is required. Its quaternions are **xyzw**.
The earlier Isaac Sim 6.0.1/PawWeaver environment was a historical porting
runtime, not the current launcher.

From the PawCerto root:

```bash
export PAWCERTO_ROOT="$PWD"
conda activate pawcerto-lab-sim610
export PAWCERTO_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=yes
cd "$PAWCERTO_ROOT/third_party/IsaacLab-develop-sim610"
"$PAWCERTO_PYTHON" "$PAWCERTO_ROOT/scripts/convert_umi_usd.py" --viz none --device cuda:0
"$PAWCERTO_PYTHON" "$PAWCERTO_ROOT/scripts/run_umi_isaac.py" --viz none --device cuda:0 \
  --joint-names "$PAWCERTO_ROOT/configs/umi_go2_arx5_joint_names.json" \
  --num-envs 16 --steps 850 --output "$PAWCERTO_ROOT/outputs/isaac/reference_example.json"
```

Conversion resolves upstream mesh paths in a derived URDF. It retains
collision-bearing `dont_collapse` links, merges massless marker frames and
reconstructs the end-effector frame from `link6` using the original fixed
transform. This avoids Isaac Sim's default 1 kg body for a massless marker.
The current converter ignores its cylinder-to-capsule option, so the script
applies that change to generated collision primitives. `usd_path.txt` holds
the actual returned USD path; the importer creates numbered directories on
reconversion.

Original Gym's measured DOF order is FL, FR, RL, RR, then arm joints 1–6.
Isaac Lab enumerates joints by tree depth. The runtime maps by name before
passing q, qdot or torque to the UMI modules. It runs the original 5 ms physics
step and four-step explicit PD control, including action delays. Evaluation
includes the original zero-action reset period before the first policy action.
Episode poses and target trajectories use each environment's local world origin.

After producing the example run in the [root training guide](../../README.md),
use one or more `--compare-checkpoint` arguments.
Each policy receives the same sampled trajectories and a fresh robot reset.
The output retains EE position/orientation error, root height/up vector and
solver contact support throughout the full trajectory; it does not label a
short successful trajectory as general training success.

```bash
"$PAWCERTO_PYTHON" "$PAWCERTO_ROOT/scripts/run_umi_isaac.py" --viz none --device cuda:0 \
  --checkpoint "$PAWCERTO_ROOT/runs/umi_v3_example/model_0.pt" \
  --compare-checkpoint "$PAWCERTO_ROOT/runs/umi_v3_example/model_4000.pt" \
  --joint-names "$PAWCERTO_ROOT/configs/umi_go2_arx5_joint_names.json" \
  --num-envs 16 --steps 850 --output "$PAWCERTO_ROOT/outputs/isaac/paired_example.json"
```

`training=True` enables the original reset perturbations and physical
randomization. It exposes actual body masses/COMs, joint damping/friction,
shape friction and solver contact forces to the original reward/critic adapter.
Mass/COM randomization occurs at construction; the task adapter randomizes
controller gains on reset. Pushes and transports follow the original intervals.
Only one process should operate the GPU at a time.
