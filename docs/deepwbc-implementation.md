# DeepWBC CPU implementation

The original network, observation assembly, and public fixed-gain control path are implemented in `pawcerto/methods/deepwbc`. CPU comparisons pass against executable source at [8159e4ed8695b2d3f62a40d2ab8d88205ac5021a](https://github.com/MarkFzp/Deep-Whole-Body-Control/tree/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a). This is component equivalence evidence using initialized original networks; no released DeepWBC policy weights were found in the reference audit, and this delivery does not establish learned control, simulator transfer, or paper reproduction. See [reference audit](deepwbc-reference.md) for original-method distinctions.

## Fetch and verification

```bash
python3 scripts/fetch_deepwbc.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest tests.test_deepwbc_policy -v
```

The fetcher verifies all 74 selected source/asset files against their Git blob hashes, including the original Go1/WidowX URDF and all 15 unique referenced meshes. It uses the fixed complete recursive tree, preserves mismatching existing files by failing, and only promotes verified partial files. The first asynchronous fetch process handle became unavailable across an app interruption; a subsequent complete verification reported 74 verified files and zero missing downloads. No dependency installation or simulator startup was performed. `third_party/deepwbc-reference` remains ignored; robot assets and external model files are not included in package redistribution. Adapted source retains the upstream BSD copyright and license in the method directory. The upstream tree does not provide a separate WidowGo1 mesh-license file; obtaining files from upstream does not establish a new asset redistribution grant.

## Configuration and policy API

```python
from pawcerto.methods.deepwbc.config import default_config
from pawcerto.methods.deepwbc.policy import build_model

cfg = default_config("public_fresh")
model = build_model(cfg, device="cpu")
# observations: [N,860]
# model.act_inference(observations, hist_encoding=True) -> [N,18]
# model.evaluate(observations) -> [N,2]
```

`default_config()` contains all resolved upstream environment sections and top-level `policy`, `algorithm`, `runner`, `seed`, and `runner_class_name`. Both fresh recipes set the original module's RESUME branch to false before resolving schedules. This differs from changing only `runner.resume` after import.

| Recipe | Iterations | Privileged regularization | Advantage mixing beta |
|---|---:|---|---|
| `public_fresh` | 40000 | 0 to 0.1, iterations 3000–10000 | 0 to 1, iterations 0–3000 |
| `paper` | 10000 | 0 to 1, iterations 5000–10000 | Public-code ramp retained: 0–3000 |

Both retain 5000 environments × 40 steps, 5 PPO epochs, 4 minibatches, LR 2e-4, and every-20-iteration DAgger. The `paper` option adopts the documented paper budget/regularizer only; it is **not** a complete paper configuration. Both retain the public position-only target task, 18 policy outputs but only 15 nonzero target scales, and zero orientation target ranges/reward weights. Full independent 6D manipulation is not established by either preset.

`ActorCritic` preserves the original state dictionary including `std`, `actor.priv_encoder`, `actor.history_encoder`, `actor.actor_backbone`, separate leg/arm heads, and two critic heads. The 10-frame convolutional history encoder maps 760 history values to 20 latent values; the privileged encoder maps 24 values to the same width. The full original API remains available to the algorithm owner, including grouped `[N,2]` log probabilities and entropy. Only debug prints and the upstream accidental global overwrite of `Normal.set_default_validate_args` were removed. There is no UMI checkpoint substitution or generic single-reward PPO.

## Observation and control boundary

`DeepWBCObserver(num_envs, dof_names, config=None, device='cpu')` takes all 20 simulator DOF names. `observe(...)` expects simulator-ordered joint positions/velocities, the latest **undelayed** simulator action from controller history, and named tensor fields documented in its signature.

| Observation slice | Values |
|---|---|
| 0:2 | Signed body roll/pitch supplied by physics caller |
| 2:5 | Body angular velocity, scale 1 |
| 5:25 | 20 joint offsets in policy order |
| 25:45 | 20 joint velocities, scale 0.05 |
| 45:63 | Latest undelayed 18 actions in policy order |
| 63:67 | Foot sensor norm >1.5, order FR/FL/RR/RL |
| 67:70 | Velocity commands, scale 1 |
| 70:73 | Current interpolated EE goal `(length,pitch,yaw)` |
| 73:76 | EE orientation delta Euler angles (zero in public default) |
| 76:100 | Added base mass, COM xyz, added gripper mass, friction, 18 motor-strength offsets |
| 100:860 | Ten older 76-value proprioceptive frames |

The observation is assembled **before** shifting history. For episode length ≤1 the next history is filled with the current frame; the already returned observation still contains the previous/reset history. Reset explicitly clears history. The foot contact predicate uses the norm of all six force-sensor wrench channels in the original source. Replacing this with three-axis net contact force needs a separate signal-fidelity evaluation through supported public Isaac Lab/PhysX APIs. This core does not claim that equivalence.

Policy DOF order is FR, FL, RR, RL, each hip/thigh/calf, then waist/shoulder/elbow/forearm-roll/wrist-angle/wrist-rotate, followed by two fingers in the observation. The runtime must verify actual named DOFs and map their simulator order. Controller construction requires the two fingers last, as the original path does.

`DeepWBCController(..., torque_limits, ...)` receives original simulator effort limits. Call `push(policy_actions)` once per 20 ms policy step, then `torques(q20,qd20,motor18)` on each of the four 5 ms physics steps. The default delay is two policy steps. Torque is `kp*(action*motor_strength*scale + q_default - q_wrapped) - kd*qd`, clipped to simulator effort limits. Motor strength scales target displacement, not the complete torque. Two zero finger torques are appended. Last three zero arm action scales still leave PD restoring their defaults.

A source quirk is intentionally preserved: observation wraps index `-8` of **20** DOFs (waist under original ordering), but torque wraps index `-8` of **18** DOFs (index 10). The test exercises both angles beyond pi so this difference cannot be silently fixed while claiming source parity. Any correction is a method change requiring separate evidence.

## Target-frame limitation and integration status

The public source references `sphere2cart`, `cart2sphere`, `euler_from_quat`, and `torch_wrap_to_pi_minuspi` through `isaacgym.torch_utils`, but their definitions are absent from the complete pinned repository. The CPU layer therefore exposes its reconstruction explicitly: `x=l*cos(p)*cos(y)`, `y=l*cos(p)*sin(y)`, `z=-l*sin(p)`, and signed principal-angle wrapping retaining +pi. LPY round-trip tests validate this chosen convention only, **not** numerical equivalence with an unavailable original helper. The fixed yaw-only world target origin `[base_x,base_y,0.53]` is directly source-grounded. Physics callers provide signed roll/pitch rather than concealing a guessed quaternion helper.

The initial CPU delivery excluded task, algorithm and simulator integration. These components are now connected: the task supplies dual rewards and physical state/randomization to the original PPO/DAgger runner, using the original full terrain and official unmodified Isaac Lab/PhysX. After the first replay exposed zero contacts from a terrain quaternion-order bug, the corrected bounded 16-environment run completed 21 updates and exported its model. The fresh 500-step replay restores contact throughout, with one signed-roll termination and weak tracking. This establishes the bounded integrated execution path, not learned control or paper reproduction; the invalid first run is preserved separately. Current conversion, material, wrench and physical-run evidence is recorded in [deepwbc-runtime.md](deepwbc-runtime.md), with algorithm contracts in [training/README.md](../pawcerto/methods/deepwbc/training/README.md). Missing angular helpers and the measured sensor differences remain explicit limitations; no custom physics-engine build is required.

## Observed evidence

All six core tests passed using the existing `pawcerto-mujoco` Python environment and one CPU thread. They execute the fixed original actor module and extracted original observation/control methods against nonconstant tensors; compare strict state exchange, both latent paths, critic, grouped probabilities, gradient-modified reverse state loading, reset/history timing, delayed actions, torque limits/fingers, and actual URDF mesh closure. Missing original angular helpers are injected explicitly for the component tests, so their parity is excluded from the evidence. These six component tests used no GPU, physical episode, training run or trained checkpoint. Subsequent integrated physical evidence is recorded separately in the runtime document.
