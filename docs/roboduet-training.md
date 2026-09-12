# RoboDuet automatic training

The dual PPO and rollout order follow the pinned training source
`a7e1528215c048199f90cb69ceb7749a1d745f28`. Default architecture, 24 physical control
steps per update, 5 PPO epochs, 4 minibatches, and the zero-based switch after
iteration 10000 are retained. The current validated artifacts are in
`outputs/roboduet-training-corrected-20260912/`; the earlier directory below is
historical evidence from erroneous runtime conditions. Training resources require a separately chosen
budget; the CLI requires an explicit number of additional iterations.

The runner reads dog observations 25 times per 24-step rollout. In Stage 2 it
samples the arm and writes body guidance before each dog observation, including
the final bootstrap-only call. It updates arm observations after each physical
step. In Stage 1 the initial arm observation is retained until cooperation
starts. Reset histories are cleared in place at the original point in the loop.
Dog PPO updates in both stages; arm PPO updates only during Stage 2. Each side
has a PPO Adam and an adaptation Adam, both registered over all actor-critic
parameters, as in the source. The arm training distribution applies tanh only
to the last two means before Normal sampling; the official playback graph uses
raw means.

`RoboDuetIsaacTrainingEnv` uses the actual task rewards, command curricula and
termination conditions in `training/task.py`, the method observer/controller,
and `Go1Arx5Isaac` public physics APIs. The default plane/no-vision/M-control
path is supported. Motor parameters are applied in the controller; friction,
restitution, payload mass and gravity must be applied by the physical runtime,
not merely inserted into privileged observations. Upstream mixed M-controller
arm values are not physical torques for reward interpretation: the original
reward tensor nevertheless contains those mixed values, which this port keeps.
See the runtime evidence for how Gym's effort-mode drive interprets them.

A short integration configuration explicitly shortens the switch to iteration 0:

```bash
OMNI_KIT_ACCEPT_EULA=yes PYTHONNOUSERSITE=1 \
/home/lyb/miniconda3/envs/pawcerto-lab-sim610/bin/python scripts/train_roboduet.py \
  --visualizer none --device cuda:0 --num-envs 16 --iterations 2 \
  --stage-switch-iteration 0 --output runs/roboduet_short
OMNI_KIT_ACCEPT_EULA=yes PYTHONNOUSERSITE=1 \
/home/lyb/miniconda3/envs/pawcerto-lab-sim610/bin/python scripts/train_roboduet.py \
  --visualizer none --device cuda:0 --iterations 1 \
  --resume runs/roboduet_short/model_2.pt --output runs/roboduet_short_resumed
```

These commands cover one Stage 1 update, one Stage 2 update, and a resumed
Stage 2 update. They are not a training-performance benchmark. The original
10000 switch boundary remains unchanged unless the test flag is supplied.

Checkpoints have format `pawcerto.roboduet.training.v1` and include complete
configuration, both network states, all four optimizer states, both adaptive
learning rates, next zero-based iteration, next rollout stage, completed arm
update count, transition count, task and
curriculum state, physical state and random parameters, observation/history
caches, and Python/NumPy/CPU/CUDA RNG states. CPU RNG is always loaded as a CPU
uint8 tensor. Save is atomic and allowed only at a completed iteration boundary.
This is a PawCerto extension: upstream resume loads only network weights.

Explicit articulation state restore does not restore PhysX internal contact
caches. Consequently a real simulator restart is not claimed to be a bitwise
continuation of the physical trajectory. The saved history/command/optimizer/
RNG/iteration state is restored without replaying or skipping a PPO update.

CPU validation in `tests/test_roboduet_training.py` executes the actual fixed
upstream PPO source with only unavailable framework imports replaced. Under
identical inputs and RNG, both sides have exactly equal GAE/advantages, eight
reported loss values, learning rates and updated weights. A clearly labeled
protocol test double checks rollout order, Stage 1/2 parameter updates and exact
next-update CPU checkpoint recovery. It provides no physical or learning claim.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/usr/lib/python3/dist-packages \
.venvs/isaaclab-sim610/bin/python -m pytest tests/test_roboduet_training.py -q
```

The system pytest path is a local test-runner workaround; it is not a runtime
dependency. The installed simulation environment has no pytest package.

The original timeout information is sticky: `reset_idx` refreshes
`extras['time_outs']` only when at least one environment resets. A following step
without resets reuses that mask. PawCerto explicitly fixes this defect by returning the current step's timeout
mask on every step. PPO timeout bootstrapping therefore applies only to actual
timeouts. This is a documented semantic correction relative to the pinned source.

The source random payload code addresses rigid body index 0, which the actual
Gym import identifies as the 0.440377 kg mounting base. Its [-2, 2] kg payload
range can produce invalid mass/inertia; Gym reported success while reading back
zero mass and negative inertia. PawCerto explicitly repairs the index defect:
`Cfg.domain_rand.base_mass_body = "trunk"` applies the same random payload range
to the named trunk. The default range and other randomization remain unchanged.
This is a documented source correction, not bitwise physics equivalence.

## Historical integration under erroneous runtime conditions, 2026-09-12

This historical run is superseded and does not validate the corrected runtime.
A later audit found that the runtime mislabeled native xyzw quaternions as wxyz,
so resets and derived observations used incorrect rotations. Its contact force
also omitted friction. Do not resume these historical checkpoints with the
corrected runtime.

The two-update run and independent resumed process both exited 0. They used 16
real Lab environments, original 24-step rollouts and network/PPO sizes, with the
explicit test switch boundary 0 and the trunk mass correction. All source
default randomization switches remained enabled. Artifacts are under
`outputs/roboduet-training-20260912/`: `short-config.json`, `train.log`,
`resume.log`, `initial/model_0.pt` through `model_2.pt`, `resumed/model_2.pt` and
`model_3.pt`, both `metrics.jsonl`, and `checkpoint-audit.json`.

| Zero-based update | Rollout stage | Dog weights max absolute change | Arm weights max absolute change | Terminations |
| --- | --- | --- | --- | --- |
| 0 | 1 | 0.0148121 | 0 | 37 |
| 1 | 2 | 0.0116843 | 0.0124173 | 41 |
| 2, resumed | 2 | 0.0128750 | 0.0118184 | 33 |

The final next iteration is 3, with 1152 environment transitions. All network
weights and reported losses are finite. The checkpoint before the resumed
update has exactly matching networks, four Adam states, both learning rates,
Python/NumPy/CPU/CUDA RNG, stage/iteration, controller buffers, task/curriculum
state, and cached observations/histories. Actual readback of root/joint state,
mass, materials, COM and gravity matches; restored inertia differs by at most
2.98e-8. CPU RNG remains a CPU uint8 tensor. This validates restart continuation
at an update boundary; no bitwise future PhysX trajectory equality is claimed.

Lab emits an explicit warning that `net_forces_w` currently returns only normal
contact force, not total contact force. Thus contact-shaped rewards/collision
signals in this integration retain a documented difference from Gym total
contact force. Terminations are frequent and rewards remain small. Finite
updates, successful restart and changed weights establish the training path,
not successful locomotion/manipulation learning or original-method acceptance.

The local AppLauncher uses `--visualizer none`, not the older `--headless`
flag. Its close method can exit before Python displays a pending exception;
the CLI now prints the traceback before closing, retaining execution failures.

## Corrected integration and five-module replay, 2026-09-12

`outputs/roboduet-training-corrected-20260912/` is the current evidence. The
physical runtime explicitly converts native xyzw and public wxyz quaternion
layouts, and sums normal force with signed ground/self-contact friction using
supported APIs. The earlier three-update artifacts are preserved separately
and are not evidence for these corrected conditions.

Both the fresh two-update process and the independent resumed one-update
process exited 0. The run retained the original full networks, 24 steps, 5 PPO
epochs, 4 minibatches, all default randomization switches and the documented
trunk-mass/timeout fixes. The bounded test uses 16 environments and switch
iteration 0.

| Zero-based update | Rollout stage | Dog weights max absolute change | Arm weights max absolute change | Terminations |
| --- | --- | --- | --- | --- |
| 0 | 1 | 0.0118477 | 0 | 5 |
| 1 | 2 | 0.0122394 | 0.0123664 | 23 |
| 2, resumed | 2 | 0.0150631 | 0.0120768 | 28 |

The final next iteration is 3, with 1152 transitions and two completed arm PPO
updates. All weights and losses are finite. Before the resumed update, the
saved networks, four optimizer states, learning rates, all RNG states,
controller buffers, task/curricula and observation histories match exactly.
Root/joint state, mass, material, COM, gravity and simulation time read back
exactly; inertia differs by at most 2.98e-8. This remains explicit-state
continuation rather than restoration of hidden PhysX contact caches.

`resumed/model_3_with_training_identity.pt` is the final full checkpoint. It is a
copy of the raw live-process `resumed/model_3.pt`, adding only
`arm_update_count=2`; all existing fields were checked equal. The processes had
loaded the earlier runner before that field was introduced. The migration is
restricted to full `pawcerto.roboduet.training.v1` runs, which start at iteration
zero and update the arm every Stage 2 iteration: the count is
`max(0, next_iteration - pretrained_to_hybrid_start - 1)`. This rule does not
apply to upstream weights-only checkpoints or arbitrary partial training. New
runner checkpoints save the actual successful arm-update count directly.

The last Stage 1 update opens the next rollout stage but leaves the arm update
count at zero. Export metadata therefore uses `next_rollout_stage` and
`arm_update_count` separately; stage 2 alone is not evidence the arm was trained.
The boundary and first arm update are exercised in the CPU runner/export tests.

Both `export_official_play/` and `export_training_mean/` contain all five JIT
modules. Using the final checkpoint's captured real physical/task caches and
histories, the CPU diagnostic restores two copies of the actual observer/plan
path and performs arm inference → body guidance → dog observation → dog
inference. Each of the five modules and both complete action outputs matches
the original checkpoint network exactly (observed max absolute error 0, checked
at float32 rtol/atol 1e-6) for all 16 captured environments in both modes.
The post-guidance dog observations and commands are also identical. The saved
inputs are `captured_inputs_official_play.pt` and
`captured_inputs_training_mean.pt`. This is CPU consumption of saved real
inputs, not a second physical rollout of the exported policies.

`checkpoint-and-export-audit.json` records the comparisons and checkpoint/runtime
hashes; `audit_and_export.py` contains the output-side diagnostic. Together the
training and export tests pass 7 cases. These checks demonstrate finite updates,
restart and five-module execution; they do not establish learned locomotion,
EE tracking, cooperation, or a successful original-method reproduction.
