# Learning Force Control CPU implementation

Official source: [Improbable-AI/learning-compliance](https://github.com/Improbable-AI/learning-compliance/tree/c760e1d74ad165d3c069d4f57ab5d066f6a41eb6), fixed at `c760e1d74ad165d3c069d4f57ab5d066f6a41eb6`.

This delivery provides the original feedforward actor/critic, concurrent history adaptation, PPO-CSE and nonrecurrent rollout storage, CPU checkpoint consumption, and original two-file TorchScript export. It does not provide a simulator environment, training result, force-tracking acceptance, or hardware deployment. No Isaac Gym/Lab import, simulator launch, GPU execution, or dependency installation was needed.

## Verified source and robot boundary

```bash
python3 scripts/fetch_learning_force_control.py
```

The fetcher preserves all 181 official files and verifies each against its pinned Git blob, including root LICENSE and B1/Z1 assets. Existing modified files are rejected, and partial files are only promoted after verification. The B1+Z1 URDF inventory has 19 movable joints and 14 unique referenced meshes, all present. The exact joint order is recorded in `artifacts/learning-force-control-cpu/source-fetch.json`.

The original URDF has unbound `xacro:include` elements (first at line 968). Strict XML parsing failed with `ParseError: unbound prefix`. The fetcher supplies a namespace **only to its in-memory inventory parser**; source bytes remain unchanged and hash verified. This inventory is not proof that the original URDF loads in another simulator. Conversion/simulator support remains outside this CPU delivery.

The earlier research report describes 17 effective paper controls. That must not be treated as the checkpoint ABI of this revision:

| Item | Fixed executable source |
|---|---|
| Network output / `Cfg.env.num_actions` | 19, inherited from `b1_plus_z1_config.py:103`; `scripts/train.py` does not replace it |
| Movable joints | 12 B1 leg joints + Z1 joint1–joint6 + jointGripper |
| Observation | 87 = gravity 3 + commands 23 + positions 19 + velocities 19 + actions 19 + clocks 4 |
| History | 10 frames, frame skip 1; input width 870 |
| Privileged information | 16 = base velocity 3 + joint dynamics 3 + applied gripper force 3 + friction 1 + gripper position 3 + target position 3 |
| Action index 18 | `legged_robot.py:105–106` overwrites it with -0.1 on every control substep |
| Joint target index 17 | Overwritten by teleop input only when `teleop_occulus=True`, at `legged_robot.py:1548–1550`; released training config sets False |

Therefore this port preserves 19 outputs. It neither slices to 17 nor assumes UMI's 18-action layout. The source does not establish a default 17-output policy. The repository's Go1 actuator-net files are actuator models, not these policy weights.

## Deployment force inputs

`RCSensor` observes scaled force **commands** in command indices 12:15 and the hybrid force/position indicator at 22. It masks spherical position commands 15:18 for force-controlled environments. Command gait entries 5, 6, 7 must already reflect the resolved fixed gait ranges, as the upstream sensor sets them before observation construction. `actor_observation` consumes those resolved commands, sensor tensors and command scales; it does not sample commands or model physics.

Actual force is privileged supervision (`EeGripperForceSensor`), obtained from `env.forces[:, gripper_stator_index, :3]` and rotated by inverse base yaw. This is the source's applied-force tensor, not an independently verified contact/wrist force sensor. Student inference estimates its 16-dimensional latent from history and concatenates it to history for the actor. It does not require a measured-force tensor as deployed actor input. The 0.05 force adaptation weight multiplies prediction and target **before** MSE, as in the original algorithm. Force reconstruction or contact equivalence in official unmodified Isaac Lab/PhysX has not been evaluated here.

## Local checkpoint and original export

The upstream runner saves a **raw state dictionary**, not a dictionary containing `model_state_dict`. It also saves `body_latest.jit` and `adaptation_module_latest.jit`. Raw weights do not contain configuration, so `load_checkpoint(..., config=...)` requires the caller's resolved configuration and performs `strict=True` loading. No dimension fallback or dropped keys are used. The supplied `training_config()` records this fixed release's training network settings; it is not evidence of a particular author's W&B configuration.

```python
import torch
from pawcerto.methods.learning_force_control.export import load_checkpoint, JitPolicy
from pawcerto.methods.learning_force_control.config import training_config
model = load_checkpoint('trusted/ac_weights_latest.pt', config=training_config())
actions = model.act_student(torch.zeros(1, 870))
# For the author's original local pair (or the export below):
policy = JitPolicy('trusted/export')
actions = policy(torch.zeros(1, 870))
```

```bash
python -m pawcerto.methods.learning_force_control.export \
  artifacts/learning-force-control-cpu/ac_weights_latest.pt \
  --config artifacts/learning-force-control-cpu/config.json \
  --output artifacts/learning-force-control-cpu/export
```

Only load trusted local checkpoint/JIT files. Both files of an original author export must belong to the same run/configuration.

The default public `scripts/play.py` references `robot-locomotion/b1-loco-z1-manip/runs/voq40aun`. On 2026-09-13, an unauthenticated GraphQL request to `https://api.wandb.ai/graphql` for this project/run and its three weight files returned HTTP 200 with `{"data":{"project":null}}`. The exact request and response are in `artifacts/learning-force-control-cpu/public-checkpoint-request.json`. This proves only that this anonymous request did not obtain the project; it does not establish why or that the project/weights do not exist. No login, credentials, private downloads or account messages were used.

The durable CPU checkpoint was initialized from the **actual fixed original ActorCritic source**, seed 21. It is not author-trained weights. `result.json` records hashes of the raw checkpoint, configuration, JIT pair and saved input/output tensors. Strict loading and original-vs-export CPU inference produced exactly equal outputs (`max_abs_error=0`, batch 3 × 19 actions). No simulator modules were imported.

## Port changes and licensing

The method directory retains the original MIT LICENSE verbatim, including its references to additional license files. Actor, PPO and storage retain their code and tensor/optimizer formulas; the original files have no additional license header. Changes remove CLI/logging imports, use instance configuration rather than mutable process-wide PPO settings, copy adaptation weights before padding, and replace the unusable `test_mode()` call to nonexistent `nn.Module.test()` with `eval()`. The original `Normal.set_default_validate_args = False` assignment overwrote a PyTorch method rather than disabling validation; the port omits that global mutation. Network outputs and PPO parameter changes match the original source in the CPU checks.

The original history wrapper's rolling/strided tensor operations are extracted into `ObservationHistory`, without a Gym wrapper. Callers explicitly append observations and reset rows; this is not a substitute environment. `actor_observation` and `force_in_base_yaw` extract the corresponding sensor tensor semantics.

Upstream `utils/utils.py` and `legged_robot.py` explicitly reference additional RSL/legged-gym licenses, but those files are absent from this fixed tree. Those files are not vendored into the method. The unused recurrent storage generator depended on that helper and raises an explicit `NotImplementedError`; original `ActorCritic.is_recurrent=False`, and active PPO uses the nonrecurrent generator. No license from another method/revision was substituted. Simulator/environment porting and recurrent helper redistribution remain blocked on their actual licensing requirements. The root package configuration registers this method's retained LICENSE as package data.

## Validation and limitations

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
  /home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python -m pytest -q \
  tests/test_learning_force_control.py
```

Result: 3 passed. Tests compare original fixed source against the port for student/teacher/value outputs, raw strict checkpoint loading, two-JIT consumption, timeout bootstrapping, storage returns, PPO and adaptation losses, and every parameter after identical CPU updates. Additional tensor checks exercise history stride/reset, force-command masking and inverse-yaw force rotation. Original-source tests skip explicitly if source has not been fetched; the delivered run used the fetched source.

Initial pytest execution discovered an unrelated ROS plugin and failed because its `yaml` dependency was absent. Disabling unrelated plugin autoload resolved that environment issue without installing anything. A first parity run exposed the original Normal API overwrite when testing a proposed setter call; removing the unnecessary global mutation resolved it. PyTorch 2.11.0+cpu emits JIT deprecation warnings; JIT is deliberately retained to consume the author's existing format.

These are source-parity and CPU-consumption results. A synthetic tensor PPO update proves neither learning nor environment fidelity. Author-policy performance, B1/Z1 simulation, force/contact equivalence, training, and real-robot control remain untested.
