# Learning Force Control through official Isaac Lab

This port follows `Improbable-AI/learning-compliance@c760e1d74ad165d3c069d4f57ab5d066f6a41eb6`. It retains the released B1+Z1 task, source network/PPO-CSE, command distribution, rewards, history, control timing and domain-randomization labels. It uses official unmodified Isaac Lab/PhysX. The authorized execution is a bounded integration run: **16 environments × 48 policy steps × 20 updates, followed by one independent resumed update**, a separately labeled three-step force-schedule boundary probe, and a coordinator-owned fixed JIT evaluation of at most 500 policy steps. It is not the source's 100,000-update training.

## The released task, rather than a name inferred from comments

`training/config.py` resolves the complete original `legged_robot_config.py`, `b1_plus_z1_config.py`, and the **actual final** `scripts/train.py:configure_env` assignments. The copied configuration body is preserved in `released_config.py`; no final override is inferred from an earlier assignment or comment. `default_config()` returns the complete JSON-compatible configuration and `resolve_config()` provides its source attribute structure.

- The robot is B1+Z1: 19 joints and 19 policy outputs. Policy action 18 is overwritten with -0.1 on each physics substep. Joint 17 is only overridden in teleop, which the released configuration disables. The paper's 17 effective-control description does not authorize changing this checkpoint ABI.
- `env.add_balls=False`. There is **no actual door/ball object** in the released default. `inverse_IK_door_opening=True` is a retained source command-distribution flag; its name is not evidence of an instantiated door task.
- The task combines commanded base longitudinal velocity/yaw with spherical end-effector trajectories and binary force/position mode. Command vector width is 23. Radius spans [0.3, 0.9] m, pitch ±2π/5, yaw ±3π/5, trajectory duration [1, 4] s. Base x velocity spans [-1, 1] m/s and yaw [-1.5, 1.5] rad/s. Mode is reselected from the original binary distribution when commands resample.
- Terrain is the source `boxes_tm`: 20 × 20 tiles, each 5 m square, source Perlin roughness [0, 0.25], horizontal/vertical scales 0.1/0.005 m. The source ground-friction range [0, 0.01] is retained. It is not replaced with a plane. The physics owner implements and validates this terrain.
- Physics dt is 0.005 s with four substeps per policy action: policy dt 0.02 s. Source rollout length is 48, episode length 20 s, command resampling every 10 s. The source runner initially randomizes episode counters; the independent resume retains saved counters instead of resampling them.
- Actor observation is 87 = gravity 3 + commands 23 + joint positions 19 + joint velocities 19 + applied action buffer 19 + gait clocks 4. History is 10 frames (870 inputs), stride 1. The 16 privileged values remain base velocity 3, joint-dynamics labels 3, applied gripper force 3, friction 1, gripper position 3, and target position 3.

## Preserve the actual control and force semantics

The default is PhysX **position drive mode 1**. The original `_compute_torques` computes lagged/scaled position targets and a clipped PD torque buffer. Only the **position targets** are submitted to the simulator in this mode. The torque buffer remains used by the original rewards and termination logic; it must be labeled `source_pd_torques`, not measured or commanded drive effort. The source motor-strength and Kd-factor randomization labels do not change the position-drive gains through that unused torque buffer. The port does not invent such a coupling. Actual implicit drive gains come from source `commands.p_gains_*`/`d_gains_*` through the physics runtime.

The original `_push_gripper` remains active from the beginning in hybrid mode. There is **no UniFP-style delayed force-training threshold**. Each environment samples its source push interval/duration and a freed/not-freed draw. Force-command components follow the source ramp/hold/ramp timing; physical external force is a randomized spring/damper toward the source Cartesian target, clamped componentwise to [-70, 70] N and zeroed for position/freed environments. Random spring kp is [25, 400]; the final source `prop_kd=0.1` makes kd proportional to kp. Source base velocity impulses and gravity/COM/friction randomization are retained.

The actor sees force **commands**, with force-mode position command indices 15:18 masked as in `RCSensor`. Privileged force and force rewards use the source externally **applied** force tensor, not an independently measured wrist/contact force. Source force rewards rotate x/y into base yaw while command generation comments describe world components; the source equations are preserved rather than silently reconciling this convention. ContactSensor data remains separate physics evidence.

## Measured source body lookup behavior

The fixed source assigns `gripper_stator_index` to the body named `link06` for external force and privileged-force access. Other methods call Gym's lookup for `gripperStator` to obtain measurement position/orientation. Actual original Gym loading of both the original URDF and its syntax-only derivative established:

- 24 rigid bodies, 19 joints; base is body 0.
- Feet appear FL, FR, RL, RR.
- `link06` is body 22; `gripperMover` is the last body (23).
- `find_actor_rigid_body_handle(..., 'gripperStator')` returns **-1**; it is not an alias.

Thus original tensor indexing reads `gripperMover` through -1. The port explicitly binds `ee_measurement_index` to **gripperMover**, preserving this proven execution result without relying on Lab body order or negative indexing. `gripper_stator_index` still names **link06**. It does not claim that this lookup bug represents the authors' intended frame. No additional fixed gripper body or new FK measurement point is added to the default. `env.gripper_position` remains the source link06 quantity; position-evaluation consumers must use `rigid_body_state[:, ee_measurement_index, :3]` for the source reward measurement.

See the physics owner's source probe and implementation report for body, asset, contact, and state-readback artifacts.

## Source ownership and license provenance

The root MIT notice remains intact. `LICENSES/legged_gym/LICENSE` and `LICENSES/rsl_rl/LICENSE` contain original BSD texts fetched at fixed commits from the exact upstream projects identified by the released LFC README. `LICENSES/PROVENANCE.md` gives immutable URLs and the code correspondence. These are not claimed to be copies recovered from LFC's missing directories. The rsl_rl helper correspondence is particularly strong: one function is AST-identical and the other differs only in a docstring character.

Original per-file headers are retained. `base_config.py`, `math_utils.py`, and the extracted task lifecycle inherit the original legged_gym header. Reward, sensor, command-curriculum, logging, and released robot/train-configuration files retain the LFC root attribution. The disabled optional object/NPZ task and unused recurrent-storage branch are not added back merely because upstream license sources were found.

`lifecycle.py` contains the original reachable task equations. Gym tensor acquisition, simulation calls, root/DOF writes and body lookup are replaced by the physics owner's real `B1Z1Isaac` methods. `torch_math.py` supplies tensor-only XYZW quaternion algebra. The port adds a read-only `transition_observer(env)` callback after original termination/reward computation and **before reset**, for evaluation of the state that actually produced those outcomes.

Known original behavior is exposed:

- The released reward scale `dof_pos_limits=-10` has no matching reward function. Original code warns and skips it. The port preserves that warning and behavior; separate arm/leg position-limit reward functions remain enabled.
- The original contact-termination code is commented out; `check_termination` initializes `contact_buf` false. Configuring `terminate_after_contacts_on=['gripperMover']` therefore does not activate that missing code. The port preserves actual termination logic rather than presenting the config string as an active rule.
- Source history and command-update ordering is retained, including command interpolation after observation construction. There is no observation rewrite to make the timing appear cleaner.
- Original root reset and velocity-impulse writes use the runtime's true state interface. Camera/video/logging frontends are excluded from the training loop; no training sensor consumes them.

## Entrypoints and checkpoints

```bash
python scripts/train_learning_force_control.py \
  --num-envs 16 --iterations 20 --seed 0 --device cuda:0 \
  --output outputs/learning-force-control-execution-20260913/train

python scripts/train_learning_force_control.py \
  --resume outputs/learning-force-control-execution-20260913/train/model_20.pt \
  --iterations 1 --force-boundary-probe --device cuda:0 \
  --output outputs/learning-force-control-execution-20260913/resume
```

The physics asset defaults to `reference/isaac/b1_z1/usd_path.txt`. Resuming uses the saved configuration and checks its USD/derived URDF identity; it does not accept a changed environment count. `--iterations 0 --resume ... --force-boundary-probe` runs only the explicitly labeled boundary diagnostic, without consuming another optimizer update.

The original raw state-dictionary format remains available as `ac_weights_latest.pt` with the source `body_latest.jit` and `adaptation_module_latest.jit` pair. The coordinator's separate training-checkpoint export API consumes the new full format without changing the raw-loader contract.

Full checkpoint format is `pawcerto.learning_force_control.training.v1`, including:

- strict model weights; both PPO and adaptation Adam states and current learning rate;
- `next_iteration` (the next update number) and `global_steps` (training transitions only);
- source `common_step_counter`, episode counters, commands and curriculum weights/RNG, force timing/gains, lag buffers, history, reward sums and last observations;
- exposed runtime root/DOF state, material, mass/inertia/COM, gravity and runtime time, plus static terrain identity;
- CPU Torch RNG ByteTensor, CUDA RNG states, NumPy and Python RNG state.

`env.common_step_counter` counts actual policy steps including the one zero-action source reset step. `env.global_steps` is that count × environment count; runner `global_steps` counts only rollout transitions. `last_observations` is restored directly, avoiding an extra history append at resume.

The runtime restores publicly exposed physical state, not hidden PhysX contact/solver warm-start caches. This is a real optimizer/task-state continuation, not a claim of bitwise physical replay across separate simulator processes. Source checkpoint defaults save every 400 updates and at the final boundary. Original exceptions remain failures, with traceback and completed/partial counters saved in `failure.json`; no silent retry is embedded in the training code.

## Validation and bounded evidence

CPU validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
  /home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python -m pytest -q \
  tests/test_learning_force_control.py tests/test_learning_force_control_training.py
```

The first three tests cover original network/PPO/storage and raw/JIT consumption. Four additional tests execute and compare the complete original configuration, compare unchanged lifecycle AST statements, run actual-source versus port force schedules through multiple timing boundaries and hybrid masks, and check XYZW rotation/Euler conventions. These are tensor/source checks, not a synthetic simulator or learning evidence.

Actual execution, independent resume, and fixed JIT evaluation results are recorded below when their processes complete. The execution is owned by the single shared GPU operator, with independent service logs and source/asset snapshots. No long-training acceptance or generalization conclusion follows from this budget.

### Completed execution: 2026-09-13

Artifacts are under `outputs/learning-force-control-execution-20260913/`, with separate operator command/source snapshots and service logs.

| Checkpoint | Completed updates | Training samples | Source policy counter | PPO Adam step | Adaptation Adam step |
|---|---:|---:|---:|---:|---:|
| `train/model_0.pt` | 0 | 0 | 1 | uninitialized | uninitialized |
| `train/model_20.pt` | 20 | 15,360 | 961 | 400 | 400 |
| `resume/model_21.pt` | 21 | 16,128 | 1,009 | 420 | 420 |

`checkpoint-audit.json` records hashes and CPU verification of these artifacts. All 23 model parameter tensors changed during the initial run and during the resumed update; all checkpoint tensors tested finite. Maximum absolute parameter change was 0.420953 from initialization to update 20 and 0.0513402 from update 20 to 21. The CPU RNG remained a CPU `torch.uint8` tensor. The original one-bin curriculum weight remained [1.0]; this is not evidence of curriculum advancement. Final checkpoint SHA-256 is `5f1e09891536758039cdb3c6839324380f104b8a4793b2a9c3fcd613314876e2`.

The initial training process completed successfully. The independent resume completed its one update and saved checkpoint/export, **then failed in the post-save boundary diagnostic**: a `torch.no_grad()` block tried to modify observation buffers created in inference mode. `resume/failure.json` and `resume.log` preserve the error and completed counters; it was not reported as a successful whole process. The fix changes only the boundary block to `torch.inference_mode()`.

The operator then ran `--iterations 0` from `model_21.pt` into `boundary-recovery/`, which exited successfully and recorded all three diagnostic policy steps. No optimizer update was repeated. One diagnostic step's physics had already occurred before the original observation-write failure; this failed attempt remains part of the execution history.

`boundary-audit.json` summarizes the source force-schedule check: all three records had 4 force-mode and 12 position-mode environments; position-mode applied force was identically zero. On the source schedule boundary one force-mode environment was freed, leaving 3 with applied spring force. Force commands were zero at the boundary and rose to a maximum absolute 0.930014 N on the following policy step; the independently computed applied spring force reached the original 70 N component limit. The pre-boundary record retained the preceding schedule's state, as expected from moving only the diagnostic counters. This is an explicit timing/mode execution probe, not a learned force-control stage or force-error acceptance test.

The coordinator's combined affected CPU suite passed **10 tests** (`cpu-validation.log`), including the original 3 CPU method tests, 4 training/source tests and 3 export/evaluation tests. Actual checkpoint 21 also passed the coordinator's CPU export/preflight before its separately owned fixed JIT evaluation. See `docs/learning-force-control-evaluation.md` for that policy-consumption result; no evaluation outcome is inferred here from training completion.
