# UniFP B2/Z1 CPU implementation

The low-level UniFP method is implemented as CPU-compatible network/PPO/storage/runner code and tensor equations, with the official B2/Z1 configuration and assets. It is **not yet an Isaac Lab task or a learned policy**. No GPU simulation or training was launched for this integration.

Source is [unified-force/UniFP at 68847a070f88d731058c3d8476929bc3b205f5bd](https://github.com/unified-force/UniFP/tree/68847a070f88d731058c3d8476929bc3b205f5bd). `scripts/fetch_unifp.py` downloads the complete method source and B2/Z1 robot assets into ignored `third_party/unifp-reference`, verifies all 62 downloaded files against their Git blob SHA-1 identities, and retains the full pinned Git tree in `.source-tree.json`. Existing differing files cause a failure. The fixed tree contains no policy checkpoint and no full high-level imitation pipeline. Root BSD-3-Clause (Unitree) and `legged_gym/LICENSE` (ETH/NVIDIA) are preserved in the reference and method package; file-level notices in algorithm sources remain intact.

## Components and executable entry points

- `config.py`, `base_robot_config.py`, `base_config.py`: original nested configuration and inheritance; only import paths changed.
- `equations.py`: 50 upstream method bodies, including observation construction, all 27 nonzero reward terms, their sum and time scaling, 17-action/19-joint PD, gait, EE goal interpolation/resampling/collision checking, termination, and commanded/external force schedules. Bodies remain verbatim and are compared against the pinned source AST. `UniFPEquations` is a mixin: its runtime supplies tensors and owns simulator state; it does not substitute a mock simulator.
- `math.py`: original sphere/Euler helpers plus plain PyTorch xyzw quaternion and random helpers replacing Isaac Gym imports. A Lab adapter must convert its quaternion ordering explicitly.
- `history.py`: zero-initialized 32-frame actor and 3-frame critic buffers, per-environment history reset, and strict force-stage predicate.
- `algorithm/`: original actor/critic, concurrent state estimator, PPO, rollout storage and on-policy runner. `from pawcerto.methods.unifp.algorithm import ActorCritic, PPO, OnPolicyRunner` is usable without Isaac Gym, W&B, or params_proto.

The actor configuration is `ActorCritic(2336, 447, 12, 73, 17, actor_hidden_dims=[512,256,128], critic_hidden_dims=[512,256,128])`. Supply `{"obs": history}` to `act_inference`, or the history tensor to `act_student`. `get_student_latent(history)` returns the 12 supervised estimates. The actor consumes the current 73 values plus the encoder's 64-dimensional latent; it does not directly concatenate the 12 decoded estimates.

## Exact observations and hardware implications

A policy frame has 73 values in this order:

| Slice | Values | Frame / scale |
| --- | --- | --- |
| 0:2 | Base roll, pitch | radians, from base orientation |
| 2:5 | Base angular velocity | base frame × 0.25 |
| 5:22 | First 17 joint positions minus nominal | radians × 1 |
| 22:39 | First 17 joint velocities | rad/s × 0.05 |
| 39:56 | Previous action applied during the last step | unscaled action |
| 56:58 | Gait sine and cosine | phase at 1 Hz |
| 58:73 | 15 commands | scaling below |

The 15 commands are base vx/vy/wz (×2, ×2, ×0.25), EE spherical radius/pitch/yaw (×0.5, ×1, ×1.3), EE roll/pitch/yaw (×0.5 each), EE force xyz (×0.01 each), base force xyz (×0.01 each). The default task does not write the sampled EE orientation offsets into command indices 6:9; preserve the actual producer, not comments implying all fields are exercised.

History is oldest to newest, zeroed on reset before appending a newly observed frame. Actor input is 32×73=2336. Critic is 3×149=447. Each privileged frame contains base velocity 3, EE spherical position 3, injected EE force 3, injected base force 3, leg reference error 12, mass/COM randomization 22, friction 1, motor strength offsets 17, stance mask 4, foot contact mask 4, gravity 3, base angular velocity 3, joint offsets 17, joint velocity 17, action 17, gait 2, commands 15, force-offset EE goal 3.

The 12 supervised values are scaled base linear velocity, EE spherical position, yaw-frame injected EE force, and yaw-frame injected base force (three each). Loss weights are [0.2,0.2,1,1], multiplied into prediction and target before MSE (thus squared in the loss). The estimator has encoder [512,256,128]→64 and decoder [128,64]→12.

**Inference does not require measured joint torque/current or an external force/torque sensor.** It requires joint positions/velocities, base orientation/angular velocity, action history and commands. This follows the executable `compute_observations`, `act_student` and `get_student_latent` source, rather than any assertion about AS2, Piper-H, or their stock grippers. Those hardware products' sensor availability and quality remain unverified here. The training force labels are the simulator's known applied external-force buffer, not actual joint torque, foot contact forces, a gripper F/T reading, or UMI solver-reconstructed foot force. Changing the training target to any of those signals changes the method.

## Control and force-stage semantics

Joint order is FL, FR, RL, RR with hip/thigh/calf each, then `z1_waist`, `z1_shoulder`, `z1_elbow`, `z1_wrist_angle`, `z1_forearm_roll`, `z1_wrist_rotate`, `z1_jointGripper`. The policy controls the first 17 joints. The last two joints retain default targets with Kp=64, Kd=1.5. All 19 torques are saturated at actual asset effort limits. First-17 torque is `Kp * (0.25 * action * motor_strength + q_default - q) - Kd * qdot`; motor strength scales the action offset, not the final torque. Default gains are 300/7.5 for hips/thighs, 500/12.5 calves, and the configured Z1 gains. Clip incoming action at ±100, not ±1.

Physics dt=0.005 s and decimation=4 give a 0.02 s policy step. The source's active loop evaluates `global_steps > force_start_step * 24` before every substep, with `force_start_step=8000`; it increments global_steps only after a policy step. Thus the switch is strictly after 192000 policy steps, and the literal multiplier 24 must not silently track a changed rollout length. `force_stage_active` preserves that expression. For active force stages, run `_push_gripper` each substep, apply `forces` in WORLD/GLOBAL_SPACE, then simulate and refresh joints. `_push_robot_base` is implemented but its call is commented out in upstream `step` even though configuration enables it. Base injected force remains zero on that path.

Commanded and external gripper forces are separate ramp/hold/release schedules, each selected at probability 0.8. Both configured ranges are [-60,60] N per component; intervals are [3.5,9] s, ramp durations [1,3] s, settling 1 s, converted with policy dt. The loops use strict inequalities, so the plateau retains the last ramp sample rather than explicitly assigning the exact target at the boundary. This discretization is preserved and tested. The actual source randomization/reset producers remain authoritative; no force schedule has been changed to emulate a sensor.

EE reward computes world target `goal + (F_injected_world + R_yaw * F_command_local) / Kp_force` with default Kp_force=200 N/m, then `exp(-2 * sum(abs(EE_position - offset_target)) / tracking_ee_sigma)`. This is compliance-offset position tracking, not an independent measured-force-error reward. The active base velocity reward similarly offsets velocity by yaw-local injected+commanded force / Kd_force (200), with the source's velocity deadband. Direct force-error reward functions exist upstream but are not active configured rewards.

After physics, refresh base/body/contact tensors; increment episode counters; update gait/base commands and EE goal; calculate termination and reward; reset done environments; append observations; then record previous actions, joint velocity, root velocity, body state and torques. Reset must clear both histories and the force schedule/command/force buffers. The original reset randomization, terrain construction and simulator calls are retained in the reference source for the future Lab owner; they are not claimed implemented by this mixin. Zero-velocity probability is 0.3 in the actual `_resample_commands`; the separately configured `zero_vel_cmd_prob_after_force=0.8` is not consumed there. Source's earlier EE-goal stage checks compare against `0 * 24` and are inactive for nonnegative global steps.

## Runtime signals to bind in an official Isaac Lab port

| Required data / owner | Consumers and semantics |
| --- | --- |
| Root state [N,13], joint q/qdot [N,19], rigid-body state [N,B,13] | Base orientation/velocity, EE position, foot/thigh positions/velocities, torque, goals and observations; upstream quaternion xyzw, world positions and body-state velocities |
| Named-body indices: four feet, four thighs, gripper, base, penalized and termination contacts | Contact masks/rewards/termination, force injection, EE goals; bind actual imported asset names, never assume Lab indices |
| Net contact force [N,B,3] world | Foot thresholds 1/5 N, collision thresholds 0.1/1 N, drag and contact-force rewards; use supported ContactSensor; distinct from `forces` |
| Joint effort limits [19], soft position limits [19,2], gains [17], nominal q [1,19] | Saturation, joint-limit costs, PD and reference posture; read actual asset limits and apply configured 0.8 soft-position fraction |
| Base local velocity/angular velocity, projected gravity, yaw quaternion, Euler angles | Recompute after physics from measured state; not predictions in critic supervision |
| Known WORLD applied force `forces` [N,B,3] | Training labels, force-offset rewards, actual external-force application; preserve body and reference frame and do not double-apply |
| Mass/COM randomization [N,22], friction [N,1], motor strengths [N,17] | Exact privileged observation and PD offset scaling; lab randomization values must match what physics receives |
| Policy actions, previous actions, previous qdot, foot air-time/last contact | Reward history, observation semantics and episode reset |
| Phase, EE goal timers/spherical trajectories/orientation offsets, command and force schedule buffers | Invoke tensor methods with upstream initialization/resampling order; `_init_buffers`, `_parse_cfg`, `reset_idx` in reference define values and lifecycle |

`UniFPEquations` deliberately uses these upstream attribute names so the simulation owner can bind views and invoke real equations directly. No public physics-engine patches are required by this CPU implementation. Supported Lab force application/contact collection, equivalent timing, asset import, reset/randomization semantics and learned performance still require runtime work and verification.

## PPO/runner preservation and repairs

PPO preserves clipping, GAE, timeout bootstrap, adaptive KL learning-rate adjustment, and a second Adam update of the concurrent estimator after each PPO minibatch. Both optimizers originally own all actor-critic parameters; only parameters receiving gradients are updated. Adaptation loss reporting retains upstream's extra division by 64, despite using the full minibatch. Recurrent helper code is not exercised because this network is feed-forward.

Integration repairs are explicitly local: remove params_proto as a configuration-only dependency; use per-Normal-instance validation settings instead of mutating PyTorch global state; use `eval()` for test mode; optional lazy TensorBoard and no W&B side effects; permit no-log-dir execution; report/save completed iteration count; save/restore both Adam states and adaptive learning rate; avoid mutating storage dones when obtaining statistics. Exact upstream optimization/state-output comparisons cover changes on the tested feed-forward path. A checkpoint does not restore simulator or random-generator state, so it does not imply deterministic mid-rollout replay.

Runner expects environment attributes `num_envs`, `num_obs`, `num_privileged_obs`, `num_pred_obs`, `num_single_obs`, `num_actions`; `reset()` returning a pair; `get_observations()` returning `obs`, `privileged_obs`, `obs_pred`; and `step(actions)` returning that dictionary, rewards, dones, infos. Timeout flags use `infos['time_outs']`. It can consume these CPU components once an actual environment supplies that protocol; no fake training environment has been added as a product path.

## Validation and limits

Run:

```bash
python3 scripts/fetch_unifp.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python -m unittest discover -s tests -p 'test_unifp*.py' -v
```

On 2026-09-12, all **11 tests passed** in `pawcerto-mujoco` (0.614 s), including the full 2336/447/12/73/17 network. All 22 URDF mesh references resolve, and the URDF actuated-joint order matches the 19-joint contract above. The tests compare initialized upstream/local network state and inference outputs, original PPO/adaptation updates including both Adam states, GAE/timeouts, checkpoint restore, observation history/reset, 27-term reward execution, force schedules, PD saturation and quaternion frames. Core tests directly compile the corresponding methods from the pinned downloaded source without importing its simulator. They use explicit tensor fixtures, not a physics simulation. An initial bytecode-level source-copy test was unsuitable across import compilation contexts; it was replaced by source AST equality, with executable parity retained.

Evidence level: CPU source/equation/optimizer parity. Outstanding: actual Lab B2/Z1 runtime, force and contact signal/timing parity, complete reset/domain-randomization/terrain port, trained low-level policy, sim-to-sim/real deployment, and unavailable full high-level imitation pipeline. Passing these tests proves none of those outcomes.
