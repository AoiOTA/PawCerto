# Nominal AS2 EDU + Piper-H UMI adaptation

This adapts UMI-on-Legs to the nominal AS2/Piper-H model. CPU checks, a corrected 3-second nominal Lab observation and one fresh complete Isaac Lab PPO update have executed. The earlier update was disqualified as correct-robot validation because of an erroneous initial orientation and remains preserved below. This is **not** a learned WBC policy, verified sim2sim transfer, identified hardware model, or real-robot readiness result. No hardware is connected. The original Go2/ARX5 configuration and default execution behavior remain available.

`pawcerto/methods/umi_on_legs/robot_binding.py` supplies the two concrete robot bindings. The existing runtime class names (`Go2Arx5Isaac`, `Go2Arx5Mujoco`) are retained for compatibility; their execution now consumes the saved binding. `scripts/prepare_as2_umi.py` writes a complete configuration consumed by training, evaluation, actor export and MuJoCo fixed-policy replay. It does not launch a simulator.

## Physical inputs and simulation choices

The source is [configs/as2_piper.json](../configs/as2_piper.json) and the assembled [AS2 asset](as2-piper-assets.md), with input SHA-256 identities embedded in `pawcerto_robot`. The source model has 22.347 kg total mass, a nominal 0.12 m mount, 18 controlled joints, and stock gripper joints frozen at 40 mm opening. These values are source-model assumptions, not measured installation or actuator calibration.

- Control order: FL, FR, RL, RR, each hip/thigh/calf; then `piper_joint1` through `piper_joint6`.
- Root body: `base_link`. TCP: `piper_gripper_base` plus local `(0, 0, 0.138)` m with identity orientation. MuJoCo also validates against its independently authored `tcp` site and source URDF FK.
- Foot force order remains FR, FL, RR, RL. The four corresponding foot bodies are retained in the prepared Lab URDF. Collision sensors cover actual source collision links. Termination uses base, hips, thighs and Piper links 1–5; collision penalty covers source collision bodies except feet, finger/gripper bodies and terminal wrist `piper_link6`. There are no Go2 head bodies; head diagnostic metadata explicitly reports an empty body list.
- Mass and COM randomization target `base_link`, `piper_link1`, `piper_link3`, `piper_link5`, `piper_link6`. The original randomization distributions remain. Actual randomized values feed the critic; no Go2 nominal masses are substituted.
- Source effort limits become controller torque bounds (legs 60/60/90 Nm, Piper source 100 Nm each). These are URDF limits, not certification of hardware limits. Lab reads the source velocity limits; MuJoCo torque actuators do not enforce velocity limits.
- Nominal position offset comes from `configs/as2_piper.json`. Gains: each leg `kp=[60,60,60]`, `kd=[1,1,.25]`; arm `kp=[80,80,60,15,10,8]`, `kd=[2,2,1,.03,.15,.02]`. Action scale remains 0.25. Gains are explicit simulation choices, not identified hardware parameters.
- Initial root release height is 0.55 m; the root-height reward target is 0.30 m, matching the observed supported nominal pose. The config and the currently supported official Lab 3 API both use xyzw; the runtime passes this rotation unchanged. An initial AS2-only conversion to wxyz was erroneous and was removed after the actual nominal Lab observation exposed an inverted initial pose. The original Go2 initialization path is preserved.
- The Lab asset owner preserves all 28 positive-mass bodies and fixed joints. The source has 42 collision shapes. The critic shape-friction block therefore changes from 34 to 42 values: critic input 269 rather than 261; actor input remains 132 and actions remain 18. The runtime checks source/Lab shape counts and uses source depth-first ordering for the critic.
- UMI runtime retains its existing contact/rest offsets, 4 position / 0 velocity solver iterations, 5 ms physics step and self-collision setting. Those are the existing UMI simulation choices, not the converter's standalone defaults.

UMI task/history, trajectory pool and sampling, pose latency, 20 ms control delay, four 5 ms substeps per policy action, action clipping, reward formulas/weights, curriculum, PPO and evaluation semantics are retained. Physical joint-limit arrays, observation offsets, relevant named-body references and the root-height target are adapted explicitly. `aligned_body_ee` uses Piper joints 1 and 5 with their nominal offsets, rather than Go2 zero assumptions.

## Commands

Prepare without Isaac or GPU:

```bash
/home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python scripts/prepare_as2_umi.py \
  --output runs/as2_umi_adaptation/config.json
```

After the separately owned official USD conversion has produced `reference/isaac/as2_piper/usd_path.txt` (currently `merged/merged.usda`) and `merged.urdf`, the existing trainer consumes that configuration:

```bash
python scripts/train_umi.py --config runs/as2_umi_adaptation/config.json \
  --trajectory reference/data/tossing.pkl --output runs/as2_umi_runtime_update_fixed_xyzw \
  --num-envs 1 --iterations 1 --seed 0 --force-signal reconstructed-solver \
  --visualizer none
```

The designated GPU operator executed this bounded runtime/update command after the quaternion repair, with explicit matching USD and URDF paths, and exited successfully. It initializes fresh weights and does not resume the earlier invalid-initial-pose checkpoint. The adaptation owner did not launch GPU simulation. One environment and one complete PPO update exercise the runtime path; they do not establish learning. Training resource and iteration budgets remain with the coordinating operator. Alternatively `--robot as2_piper` adapts the selected original config in memory. `--joint-names` is optional; when supplied its exact order must match the selected binding. Resume/weight initialization rejects a different robot binding. Go2 weights are not an AS2 policy.

Full-checkpoint `eval_umi.py`, `run_umi_isaac.py` and `export_umi.py` read the saved binding and choose its paths/order. Export retains the full execution config and joint names. `python -m pawcerto.mujoco.evaluate` consumes the configuration beside the checkpoint/export and selects the matching MuJoCo model. Explicit `--usd-path` / `--urdf-path` retain their existing provenance behavior. Moving machines may require these explicit path overrides; saved paths are absolute.

## Evidence and outstanding work

CPU tests cover method-input invariants, actual AS2 source FK/TCP and velocity, 132-D observation, 269-D critic construction, real MuJoCo per-substep delayed PD parity, selected reward body/joint resolution and rejection of mismatched Go2 joint order/weights. The original Go2 runtime also produced exactly identical state values to the pre-change HEAD implementation over 30 policy / 120 physics steps with identical deterministic actions.

The initial damping choice (legs 2; arm 3,3,2,.8,.6,.5) failed at 0.11 s with `mjWARN_BADQACC`, DOF 18, and engine time reset to 0.005 s. The runtime raised `SimulationInstability`; it did not hide the engine's finite auto-reset. The nominal mass matrix showed effective inertia approximately 0.000343 kg m² at the final wrist joint, making the original trial's 0.5 damping too large for this explicit 5 ms PD path. The revised gains above completed 150 policy / 600 physical steps (3 s) of zero-action PD: minimum up-dot 0.98879; final root height 0.29986 m; final four ground-supported feet and zero nonfoot contacts; no numerical failure. The release transient reached a minimum root height of 0.22282 m. Artifact: `runs/as2_umi_adaptation/mujoco_basic_control.json`.

The MuJoCo result is a short basic-control observation, not WBC learning or long-run stability. Subsequently the designated GPU operator completed the authorized 1-environment / 1-update Isaac Lab run at `runs/as2_umi_runtime_update`, with 24 transitions and `reconstructed-solver` selected. This exercised actual USD prim paths, randomized body/shape setup, the 269-D critic, control and reconstructed-force runtime calls without contract errors. It did not establish numerical identity of the force reconstruction to an external reference.

The CPU audit in `runs/as2_umi_runtime_update/cpu-artifact-audit.json` verifies checkpoint iteration 0→1, matching embedded/execution configs and current asset hashes, finite saved model/optimizer/environment-training tensors (20 tensors before, 71 after), and actual actor and critic parameter changes (maximum absolute change 0.0155683). All reported numeric metrics are finite. The run reports termination mean **0.75**, normal-contact supported-feet mean **0.0**, position error **0.59181 m**, orientation error **1.71594 rad**, and final policy KL **0.10067**. These do not demonstrate supported tracking or learning. Simulator state and force tensors were not saved; checkpoint finiteness does not establish finiteness of every internal physics tensor. No further GPU run or hardware operation is part of this bounded audit. Original-Gym pointwise force identity and AS2 sim2sim success are not claimed.


### Corrected initial orientation after deterministic Lab observation

The first Lab nominal 3 s run (`runs/as2_umi_nominal_lab_3s`) completed 600 finite physics steps, but started with up-dot **-1** and TCP z **-0.065838 m**, while the matched MuJoCo initial TCP z was **1.165838 m**. This isolated an adaptation error before dynamics: an unnecessary AS2-only xyzw→wxyz conversion had been added despite the current official Lab 3 `matrix_from_quat` and articulation initialization API using xyzw. The prior observation file also mislabeled its raw API quaternion as wxyz; the corrected recorder explicitly converts API xyzw to its named output convention. The runtime conversion has been removed, preserving the existing Go2 path.

The earlier `runs/as2_umi_runtime_update` remains an erroneous-initial-pose execution record, **not correct AS2 runtime validation**, despite finite tensors and actual parameter changes. Its audit now records this invalidating discovery without deleting the original metrics. The authorized correction repeats the same 3 s nominal observation and then, if suitable, a fresh one-update check; it does not resume the invalid checkpoint or establish learning.

The deterministic observation entry point is:

```bash
python scripts/observe_as2_umi_control.py \
  --config runs/as2_umi_adaptation/config.json \
  --output runs/as2_umi_nominal_lab_3s_fixed_xyzw --visualizer none
```

It fixes the budget at one environment and 600 existing 5 ms steps, calls the actual runtime reset and UMI controller, disables domain randomization, and writes the initial state plus every physical-step sample. It records root pose/up-dot, q/qd, commanded and PhysX-readback efforts, TCP, per-body normal contact forces, ground-filtered foot force and reconstructed solver force. No actor or reward is called. Output directories must be new so unsuccessful observations remain available. The paired MuJoCo record is `runs/as2_umi_adaptation/mujoco_paired_steps.jsonl`; its post-integration TCP uses independently validated source URDF FK without changing the running dynamics. Contact sample phases differ between engines and are explicitly described in the metadata.


### Verified correction and fresh update

The corrected nominal observation in `runs/as2_umi_nominal_lab_3s_fixed_xyzw` completed **600 steps / 3 seconds**. Its initial up-dot is **1**, initial joint values differ from MuJoCo by at most **4.77e-8 rad**, and the initial TCP transform differs from source/MuJoCo FK by at most **4.50e-7** per matrix component. PhysX-readback effort equals commanded effort exactly; independently recomputing PD from the previous sample's q/qd differs by at most **7.28e-6 Nm**. All 601 recorded states (including reset) are finite. The last second has four ground-supported feet in every sample, minimum up-dot **0.999937**, final root height **0.31613 m**, final maximum joint speed **0.07830 rad/s**, and no nonfoot net-contact body above 1 N at the endpoint.

The release transient still differs from MuJoCo: Lab reaches minimum root height **0.16078 m** and up-dot **0.96798**, versus MuJoCo **0.22252 m** and **0.98844**. Across the full 3 seconds, RMS root-position distance is **0.06641 m** and RMS TCP-position distance is **0.18056 m**; endpoint TCP distance is approximately **0.0203 m**. These are retained physical differences, not claimed engine equivalence or learned WBC. Read `paired-cpu-audit.json` in the corrected observation directory alongside both engines' step records.

The subsequent fresh run `runs/as2_umi_runtime_update_fixed_xyzw` completed **one environment / one full PPO update / 24 transitions** with actor **132**, critic **269**, and reconstructed-solver force selected. Its snapshot uses the exact same repaired runtime as the nominal observation (SHA-256 `7b9cde8c36b722b5f155204a729f83da20d9eebc0f289ee7daefe393bfc56c54`). CPU audit verifies current asset identities, identical embedded/execution configurations, finite saved training tensors and actual actor/critic changes (maximum **0.0169058**). Termination and collision metrics are **0**, mean normal-contact supported feet **2.4583**, position error **0.70481 m**, orientation error **1.65493 rad**, and final policy KL **0.07291**. This closes the bounded corrected-runtime update check; it is not an AS2 learning or sim2sim result. The audit is `runs/as2_umi_runtime_update_fixed_xyzw/cpu-artifact-audit.json`. No further training or hardware execution is included.
