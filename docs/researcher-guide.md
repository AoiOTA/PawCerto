# Using and extending PawCerto

PawCerto's goal is a research framework for reinforcement learning of whole-body control on quadrupedal manipulators. Today the implemented path is **UMI-on-Legs on Go2 + ARX5**, with Isaac Lab/PhysX training, independent MuJoCo execution, and actor export. This guide describes that working path and its concrete extension points. RoboDuet on Go1 + ARX5 now also has method-specific entrypoints; see [its training and export guide](roboduet-training.md). Its short Stage 1/2/resume validation proves integration. [Actual fixed-policy evaluation](roboduet-evaluation-result.md) completes 2/9 full cases at checkpoint 0 and 0/9 at 1600, with worse same-prefix leg tracking at 1600; [Checkpoint 10001](roboduet-model10001-evaluation.md) also completes zero of nine cases, with low-height termination after 0.10–0.40 seconds. Subsequent training has entered Stage 2; effective learning remains unverified. DeepWBC on Go1 + WidowX 250s completed corrected-terrain 21-update/500-step physical execution and independent actual-input export consumption; learning remains unproved. AS2 EDU + Piper H completed its 4000-update UMI candidate: Lab tracking improved but 7/16 cases inverted, final MuJoCo transfer failed in all 16 cases, and export consumption failed. Stable WBC and sim2sim were not established. There is no method registry, robot plugin API or universal training command.

The existing three training seeds establish learning and sim2sim results on the supplied trajectory pool, with remaining head-contact and stability problems. They do not establish held-out generalization, reliable whole-body control or real-hardware readiness. Read [current results](umi-current-result.md) and [public reproduction status](release-reproduction.md) before comparing a new experiment. KISS My Agent is optional developer tooling, not a user or runtime dependency.

## Robot coverage and planned transfer

Robot support means a measured method/robot/runtime combination, not just a loadable asset. The current priority is:

| Combination | Method or purpose | Evidence reached |
|---|---|---|
| Go2 + ARX5 | UMI-on-Legs | Actual training, Isaac Lab/MuJoCo evaluations and policy export; retain the reported stability and generalization limits |
| Go1 + ARX5 | RoboDuet | Actual Stage 1/2/resume, export and fixed physical evaluation; early 0/1600 comparison shows no learning improvement; 10001 fixed evaluation completed with zero of nine full cases; Stage 2 training is ongoing and the 50000 endpoint remains pending |
| Go1 + WidowX 250s | DeepWBC | Corrected full-terrain 21 updates and 500 policy steps in physical simulation; actual-input CPU export consumption; EE RMSE 0.47936 m and one signed-roll failure; learning unproved |
| AS2 EDU + Piper H with stock gripper | UMI target-body adaptation | Completed 4096 × 24 × 4000 candidate; [final result](as2-umi-learning-result.md): Lab tracking improves with 7/16 inversions and degraded support; MuJoCo 16/16 BADQACC, zero complete cases; export consumption fails |
| B2 + Z1 | UniFP | Official Lab 20+1 updates, separate strict force-stage probe, 500 exported-policy-driven steps and real-input CPU consumption; one roll reset, EE error 0.458129 m; training/evaluation rigid-property difference retained; learning unproved |
| B1 + Z1 | Learning Force Control | Official Lab 20+1 updates, actual 19-output/native-position-drive semantics and zero-update probe recovery; 500×16 finite JIT transitions and all-input CPU parity, four orientation resets and 0.821928 m position-mode EE error; author weights not obtained and learned force control unproved |

Go2 + Airbot Play (MLM) remains conditional on missing author training material.
Spot + arm (ReLIC) now has [bounded original PLAY runtime execution](relic-external-runtime.md)
and an [external-export CPU consumer](relic-external-export.md). The actual
19-joint order and source downstream arm/selected-leg target writes are recorded.
The 500-step run had eight ground-contact terminations, so correct target writes
and CPU parity do not establish stable performance. Upstream files remain
external under their noncommercial license; historical physics/contact equivalence
and the weights' training provenance remain unverified. No all-method/all-robot compatibility or physical
hardware execution is implied. See the [execution plan](project-plan.md) and
[AS2/Piper H asset assumptions](as2-piper-assets.md).

For DeepWBC, follow the method-specific [source preparation and training](deepwbc-training.md)
and [fixed-policy evaluation/export](deepwbc-evaluation.md). The [final report](experiments/deepwbc-reconstructed-runtime-20260912.md)
separates corrected supported-contact execution from the withdrawn zero-contact
run and states the approximate sensor and position-only task limits.

For AS2/Piper H, start with [source assets](as2-piper-assets.md) and [official Lab
conversion](as2-piper-isaac.md), then use the [learning/evaluation protocol](as2-umi-learning-plan.md).
That protocol specifies the actual run's model 0/500/4000 fixed16 comparisons,
final paired Lab evaluation and independent export consumer. The [completed result](as2-umi-learning-result.md)
records improved Lab tracking with inversions, failed MuJoCo transfer and failed
export consumption. That one-candidate run is finished. The subsequent
[target-aware pretraining family](as2-pretraining-family.md) now has generated
assets and actual Lab/MuJoCo consumption evidence; new learning has not started.
Its proposed learning budget is separate from the completed candidate. Run scripts from
the source checkout: a wheel does not bundle the scripts, vendor inputs or local
experiment outputs. Coordinate simulator work with the assigned GPU operator;
these examples do not authorize duplicate training services.

For UniFP, use [the original CPU components](unifp-implementation.md),
[official Lab conversion/training/resume](unifp-lab-runtime.md) and
[fixed evaluation/export](unifp-evaluation.md). The source installation extra
`python -m pip install -e '.[unifp]'` provides the tested pydelatin 0.3.0 terrain
dependency. The completed 21-update and 500-step check does not authorize another
run or full force-stage training. Its runtime report preserves the failed
post-checkpoint probe and training-time gripper-property gap.

For Learning Force Control, start with [the integration guide](learning-force-control.md),
then [official Lab training/resume](learning-force-control-lab.md) and
[full-checkpoint export/evaluation](learning-force-control-evaluation.md).
The historical [CPU raw-checkpoint entry](learning-force-control-implementation.md)
still requires the resolved config and matching two-file JIT pair; full training
checkpoints use the separate exported contract. Author weights were not obtained.
The actual task uses 19 network outputs and native position drives. The bounded
20+1 update execution and recovered post-save probe do not establish learned
force control. The completed fixed evaluation has four orientation resets and
poor tracking despite all-input export parity; its initial zero-consumption
reset failure remains preserved. The bounded assignment is closed. Preserve the documented source
measurement/force-body distinction and derived asset/license provenance.

For ReLIC, follow [external PT/ONNX consumption](relic-external-export.md).
Supply trusted local exports and packed observations; the adapter does not fetch
models, build robot observations or run a simulator. ONNX requires an already
available optional ONNX Runtime. The measured comparison uses three synthetic
inputs, not physical trajectories or learned-controller acceptance. To execute
the actual external task, use the separate [original PLAY runtime entry](relic-external-runtime.md)
with the designated simulator operator. Its documented two import migrations,
singleton-contact shape repair and local actuator constructor bridge preserve
the original compute path without an engine patch. The recorded 500-step budget
is an experiment boundary, not a permanent CLI limit or an authorization to
repeat simulation.

## 1. Run the existing path

Use the repository root as the working directory unless a command explicitly changes it. Keep source configuration, full checkpoints and derived exports together under distinct experiment names; do not reuse a directory for a different recipe. Generated `reference/`, `runs/` and `outputs/` content is not automatically supplied with a source checkout. Availability of historical weights and evidence archives is listed in [reproduction status](release-reproduction.md).

| Step | Existing entry and required inputs | What it produces |
|---|---|---|
| Fetch inputs | [scripts/fetch_umi.py](../scripts/fetch_umi.py), invoked as `python3 scripts/fetch_umi.py` | Pinned official source/robot assets, published trajectories and policy/config files under `third_party/` and `reference/` |
| Install independent evaluation | [root environment instructions](../README.md#inputs-and-environments), Conda `pawcerto-mujoco` | The selected MuJoCo 3.13 CPU environment; no Isaac installation is needed for this path |
| Build MuJoCo asset | [scripts/build_mujoco.py](../scripts/build_mujoco.py), after fetching | Generated MJCF with local mesh paths; rebuild after moving the checkout |
| Install training | [Isaac Lab installation guide](isaaclab-install.md), Conda `pawcerto-lab-sim610` | Fixed official Lab snapshot and Isaac Sim 6.1.0.0 dependencies; no PhysX rebuild |
| Convert Lab asset | [scripts/convert_umi_usd.py](../scripts/convert_umi_usd.py), using the Lab launcher environment below | Derived URDF/USD and the path recorded in `reference/isaac/go2_arx5/usd_path.txt` |
| Train or continue | [scripts/train_umi.py](../scripts/train_umi.py), config, trajectory file and measured joint order | `config.json`, `metrics.jsonl`, and full `model_N.pt` checkpoints |
| Evaluate | Nominal Lab, author-protocol Lab, or MuJoCo entries in section 3 | Per-condition metrics and recorded execution; protocols must be reported separately |
| Export | [scripts/export_umi.py](../scripts/export_umi.py) and the full checkpoint | `actor.ts`, execution `config.json`, `joint_names.json`, and source identities in `export.json` in a new directory |

The shortest published-policy CPU execution, after installing the documented CPU environment, is:

```bash
conda activate pawcerto-mujoco
python3 scripts/fetch_umi.py
python scripts/build_mujoco.py
python -m pawcerto.mujoco.evaluate \
  --checkpoint reference/checkpoints/tossing/ours \
  --trajectory reference/data/tossing.pkl --seconds 17 --seed 2027 \
  --output outputs/mujoco/researcher_official_example.json
```

For Lab, follow installation once rather than reinstalling an existing environment. Establish these variables from the repository root in each shell:

```bash
export PAWCERTO_ROOT="$PWD"
conda activate pawcerto-lab-sim610
export PAWCERTO_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=yes
cd "$PAWCERTO_ROOT/third_party/IsaacLab-develop-sim610"
"$PAWCERTO_PYTHON" "$PAWCERTO_ROOT/scripts/convert_umi_usd.py" --viz none --device cuda:0
```

`--viz none` is the current pinned development launcher's option. Historical `--headless` examples belong to the preserved release environment. Use the committed [Go2 + ARX5 joint order](../configs/umi_go2_arx5_joint_names.json); researchers do not need the old Isaac Gym runtime to re-query it. The [original Gym environment](reference-runtime.md) is a separate optional reference path.

## 2. Train, split and resume without changing the experiment identity

The [root training recipe](../README.md#train-the-current-physical-variant-in-isaac-lab) reproduces the existing physical variant, including the explicit `joint_velocity_limit_override_rad_s=1000.0`. Without that override the original config retains native URDF speed limits. Do not describe those two physical configurations as the same experiment. `--force-signal reconstructed-solver` is the training entry's default; `normal-contact` selects the earlier proxy. Their meanings and resume compatibility are documented in [force/runtime semantics](umi-reconstructed-force-training.md).

The following creates a **new researcher run** of the split-training recipe. The
project's own 71-ID, 4000-update run has completed; its results and known test
instability are in [the split-trained result](umi-source-inertia-result.md).
The example carries forward body-speed-v3 reconstructed forces and the explicit
1000 rad/s joint-speed override. Select the matching source-inertia asset from
the conversion step and preserve its identity when resuming.

Using the Lab shell established above, create the input in a new run directory. This changes only the already documented speed override in the released config; `exist_ok=False` preserves existing experiment directories:

```bash
cd "$PAWCERTO_ROOT"
"$PAWCERTO_PYTHON" - <<'PYCONFIG'
import json
from pathlib import Path
config = json.loads(Path('reference/checkpoints/tossing/ours/config.json').read_text())
config['joint_velocity_limit_override_rad_s'] = 1000.0
output = Path('runs/researcher_split_seed0')
output.mkdir(parents=True, exist_ok=False)
(output / 'input-config.json').write_text(json.dumps(config, indent=2) + '\n')
PYCONFIG
cd "$PAWCERTO_ROOT/third_party/IsaacLab-develop-sim610"
"$PAWCERTO_PYTHON" "$PAWCERTO_ROOT/scripts/train_umi.py" \
  --config "$PAWCERTO_ROOT/runs/researcher_split_seed0/input-config.json" \
  --trajectory "$PAWCERTO_ROOT/reference/data/tossing.pkl" \
  --split-manifest "$PAWCERTO_ROOT/configs/umi_tossing_split_seed2027.json" \
  --joint-names "$PAWCERTO_ROOT/configs/umi_go2_arx5_joint_names.json" \
  --num-envs 4096 --iterations 4000 --save-every 100 --seed 0 \
  --force-signal reconstructed-solver \
  --output "$PAWCERTO_ROOT/runs/researcher_split_seed0" --viz none --device cuda:0
```

No `--weights` or `--resume` means random initialization. The current rollout default is 24 transitions per environment; 4096 × 24 × 4000 yields 393,216,000 environment transitions. That budget is an experiment configuration, not an acceptance test. Use a new output directory and select your candidate recipe on validation results before starting final test evaluation.

The checked-in split selects **71 train / 15 validation / 15 test** from 101 source IDs, with seed 2027. Train always consumes `train` when the manifest is passed. Hashes, partition and actual IDs are embedded in the saved configuration and full checkpoint. Without a manifest, a legacy config keeps full-pool sampling. The source has no recording IDs, so unknown same-recording fragments cannot be certified independent. [Split details and evidence](umi-data-split.md) explain this limitation and the grouping procedure.

For an interrupted run saved at iteration 1500, this example performs **2500 additional updates**, ending at 4000, and writes continuation artifacts into a fresh directory:

```bash
"$PAWCERTO_PYTHON" "$PAWCERTO_ROOT/scripts/train_umi.py" \
  --config "$PAWCERTO_ROOT/runs/researcher_split_seed0/config.json" \
  --resume "$PAWCERTO_ROOT/runs/researcher_split_seed0/model_1500.pt" \
  --trajectory "$PAWCERTO_ROOT/reference/data/tossing.pkl" \
  --split-manifest "$PAWCERTO_ROOT/configs/umi_tossing_split_seed2027.json" \
  --joint-names "$PAWCERTO_ROOT/configs/umi_go2_arx5_joint_names.json" \
  --num-envs 4096 --iterations 2500 --save-every 100 --seed 0 \
  --force-signal reconstructed-solver \
  --output "$PAWCERTO_ROOT/runs/researcher_split_seed0_resume1500" --viz none --device cuda:0
```

Resume restores optimizer, iteration, learning rate, stored RNG and available curriculum state; physics resets, so it is not exact in-flight rollout continuation. Preserve the source runtime contract and split. The entry rejects cross-partition resume/weight initialization and requires a manifest with a saved split config. `--weights` initializes actor/critic only, without continuing optimizer or iteration, and is mutually exclusive with `--resume`; it does not turn previously full-pool weights into held-out weights. See the actual [runner save/load implementation](../pawcerto/methods/umi_on_legs/training/runner.py).

## 3. Choose the evaluation protocol and export

| Question | Entry | Coverage and interpretation |
|---|---|---|
| Nominal native Lab tracking | [scripts/run_umi_isaac.py](../scripts/run_umi_isaac.py), `--num-envs 16 --steps 850`, optionally `--compare-checkpoint` | Shared sampled trajectories and fresh resets; neither exhaustive source-ID coverage nor the author randomized protocol |
| Author-protocol native Lab performance | [scripts/eval_umi.py](../scripts/eval_umi.py), full checkpoint | Author configuration adjustments, trial/reset behavior and completed-episode statistics; `--split-manifest` / `--partition` select a pool but sampling remains with replacement |
| Independent sampled sim2sim | [pawcerto.mujoco.evaluate](../pawcerto/mujoco/evaluate.py), `--cases 16 --seed 2027 --seconds 17` | One batch draw sliced into 16 cases; repeated IDs are possible |
| Every validation or test trajectory | [scripts/eval_umi_split_mujoco.py](../scripts/eval_umi_split_mujoco.py) | Every ID in the explicitly selected manifest partition once; no replacement |

To develop against every validation ID using the split-training example, activate `pawcerto-mujoco`, return to the repository root, and run:

```bash
python scripts/eval_umi_split_mujoco.py \
  --checkpoint runs/researcher_split_seed0/model_4000.pt \
  --trajectory reference/data/tossing.pkl \
  --split-manifest configs/umi_tossing_split_seed2027.json \
  --partition validation --seconds 17 --seed 2027 \
  --output outputs/researcher_split_seed0/validation
```

Use the continuation directory's checkpoint if resuming the example. Freeze the recipe and checkpoint selection before running the same command with `--partition test` and a fresh output directory. Inspect every `trajectory_ID.json`, raw NPZ and `summary.json`; do not use test results to choose another candidate while retaining an untouched-test claim. Invalid/incomplete split evaluations save their reports and exit with code 2. An abrupt failure before a result exists propagates its exception; incomplete output is not a successful run.

Tracking means alone are insufficient: retain orientation error, inversion, numerical failures, terminations, contacts and support throughout the requested motion. Foot-force definitions differ between solver reconstruction, normal contacts and ground-filtered contacts. Calf or other intended support should be interpreted with the full motion, rather than rejected solely by contact counts. Nominal no-inversion and author timeout fraction do not measure the same thing; timeout is a survival indicator. See [contact diagnostics](umi-contact-next-step.md) and the [MuJoCo implementation](../pawcerto/mujoco/README.md).

For a saved actor package:

```bash
python scripts/export_umi.py \
  --checkpoint runs/researcher_split_seed0/model_4000.pt \
  --joint-names configs/umi_go2_arx5_joint_names.json \
  --output outputs/export/researcher_split_seed0
python -m pawcerto.mujoco.evaluate \
  --checkpoint outputs/export/researcher_split_seed0 \
  --trajectory reference/data/tossing.pkl --seconds 17 --seed 2027 \
  --output outputs/mujoco/researcher_export_example.json
```

The export verifies CPU actor reload parity before writing. `actor.ts` emits raw actions; it does **not** include observations, history, delay, scaling, PD or physics. Consumers must retain those from the execution config and adapters. The author Lab evaluator also needs critic/policy-standard-deviation data from a full checkpoint, so an actor-only directory is insufficient there.

New UMI exports also contain `export.json`, recording hashes of the actual source
weights, source execution config and exported actor, plus available embedded
training metadata. This record supports provenance inspection; the actor-only
loader still does not qualify a training partition from adjacent JSON.

Training-partition qualification comes from the actual full checkpoint's embedded config, not its adjacent execution config. Missing metadata and actor-only exports are conservatively unqualified for held-out reporting. Keep the full checkpoint for validation/test identity even after exporting. A legacy full-pool policy evaluated on these IDs remains a full-pool policy.

## 4. Where the data and control flow live

These are concrete implementation boundaries, not a stable generic backend API:

Three shared components now have existing method/simulator consumers:
[`pawcerto.artifacts.file_identity`](../pawcerto/artifacts.py) records input bytes
without imposing a checkpoint schema;
[`pawcerto.robots.go1_arx5`](../pawcerto/robots/go1_arx5.py) supplies the same source
URDF, 20 simulation-joint ordering and default pose to RoboDuet's Lab and MuJoCo
asset paths. The robot module imports no simulator or tensor library. Its 20
simulation joints do not redefine the method's 18 policy actions; and
[`pawcerto.robots.urdf.read_joint_limits`](../pawcerto/robots/urdf.py) reads named
lower/upper/effort/velocity limits in the caller's order for the UMI and RoboDuet
Lab adapters. The methods retain their own controllers, rewards, histories and
checkpoint loaders.

UMI's `train_umi.py`, `eval_umi.py` and `run_umi_isaac.py` accept `--urdf-path`
alongside `--usd-path`. Supply a matching merged URDF when changing the USD:
the USD provides simulator geometry, while the URDF provides UMI's source
velocity limits, contact-link topology and physical-randomization source data.
Each path uses the explicit argument first, the saved asset path second, then
the original Go2/ARX5 workspace default. Existing records without a URDF path
therefore retain the original URDF default. Resolved input paths and byte hashes
are recorded in `pawcerto_asset` (training/evaluation config) or `asset_identity`
(rollout report). Recording both files does not establish that they match.
UMI still requires positive source velocity limits and retains its existing
explicit velocity override. The saved UMI robot binding supplies the root, TCP and named-body/joint
references. Go2/ARX5 remains the default; AS2/Piper H has an explicit nominal
binding. Supplying a URDF alone does not adapt arbitrary robots or policies.

`tests/test_robot_inputs.py` checks both real source URDFs, alternate UMI limits
and contact topology, and all three CLI path/recording/runtime-call blocks on
CPU without starting Kit. These input checks alone do not establish GPU execution
or learning with an arbitrary replacement asset. AS2 has separate measured
[physical execution and completed candidate evidence](as2-umi-learning-result.md).

| Owner | Inputs → outputs and responsibilities |
|---|---|
| [UMI execution module](../pawcerto/methods/umi_on_legs/__init__.py) | `PoseSequence` loads target arrays; `UmiObservation` turns `RobotState`, target previews and previous actions into actor observations; `UmiController` handles clipped actions, delay and explicit PD; `UmiPolicy` loads the actor/config |
| [Isaac runtime](../pawcerto/isaac/runtime.py) | `Go2Arx5Isaac` owns the asset, joints, actual randomized physical properties, reset/state and `step_torque`; [solver_force.py](../pawcerto/isaac/solver_force.py) reconstructs supported solver force signals |
| [Training environment](../pawcerto/methods/umi_on_legs/training/isaac_env.py) | `UmiIsaacTrainingEnv` owns episode targets, observation/control cadence, rewards, termination, reset and curriculum lifecycle |
| [Method semantics](../pawcerto/methods/umi_on_legs/training/semantics.py) | Original reward/critic equations, force-input selection and runtime compatibility; keep algorithm changes distinct from physical adaptation |
| [Runner and PPO](../pawcerto/methods/umi_on_legs/training/runner.py) | `UmiTrainer` gathers rollouts, updates the vendored PPO and saves full state; `reset()` returns actor/critic observations and `step()` returns actor/critic observations, reward, done and info |
| [MuJoCo runtime](../pawcerto/mujoco/runtime.py) and [asset conversion](../pawcerto/mujoco/asset.py) | Independent CPU robot state, contacts and delayed explicit PD; the evaluator reuses UMI observations and actor inference rather than replaying recorded joint positions |

The current actor/action path has 18 joint actions, ordered by the measured original asset mapping. Policy cadence is four 5 ms physics steps. Reset includes one zero-action policy step; history sentinels and pose-update cadence affect the policy. Training supports sparse and dense reward/task stepping; the author evaluator changes that configuration intentionally. Do not move updates between these clocks as an unnoticed “cleanup.”

`RobotState` contains root angular velocity and gravity in the root frame, joint positions/velocities in policy order, a 4×4 end-effector pose in the environment-local world frame, and episode time. Lab removes each environment's world origin. The Go2 end-effector frame uses `link6` and its original fixed transform; AS2 uses the saved Piper gripper TCP binding. Using a body origin in place of either TCP changes the task. MuJoCo exposes xyzw quaternions at the adapter boundary and corrects link-origin linear velocity relative to the inertial center.

## 5. Introduce new target trajectories

For the existing robot/task, a new trajectory file is already supported by `--trajectory`; no new training platform is required. `PoseSequence` currently loads a pickle list of episode dictionaries. Prepare files from a trusted source because this reader uses Python pickle.

- Required arrays are `ee_pos[T,3]` in metres and `ee_axis_angle[T,3]` as rotation vectors in radians, **not Euler angles**. Episodes must be stackable to the same time length. Targets share the environment-local world frame expected by the task.
- Sampling uses `sequence_sampler.dt`, not the dictionary's `t` timestamps. Resample irregular data offline to that grid; the current reader does not interpolate it. The requested episode length truncates long arrays and pads short ones with their final pose. Review whether a held final target is suitable for the new task.
- `planar_center` subtracts the mean xy position of frames 1, 2 and 3; this requires at least four available target frames. `add_random_height_range` applies a sampled vertical offset. Preserve or deliberately change these config semantics rather than pre-centering twice.
- Fields such as `gripper_width`, recorded velocity or acceleration are not consumed by this target reader. A new manipulation task requiring gripper control or force interaction therefore needs actual action/task implementation; adding fields alone does not provide those capabilities.
- Preserve recording identifiers when available. [scripts/split_umi_trajectories.py](../scripts/split_umi_trajectories.py) accepts a new source and output manifest, grouping exact target content and explicit `recording_id`, `source_recording_id` or `source_id` links. Pass both the new `--trajectory` and its new `--split-manifest` to consumers; old manifests reject a changed source hash.

Before robot execution, inspect all selected IDs, finite target values, shapes, coordinate transforms, durations, padding and reachable workspace. Run [data-split tests](../tests/test_umi_data_split.py) and check any reader changes against [policy/trajectory parity tests](../tests/test_umi_policy.py). Then test the actual full target motion, including approach, contact and exit where required. A pushing or simplified trajectory is a different task, not evidence of held-out tossing performance.

## 6. Adapt another robot or method

A different URDF is **not** sufficient to support a new quadrupedal manipulator. UMI now has two concrete [robot bindings](../pawcerto/methods/umi_on_legs/robot_binding.py): the validated Go2 + ARX5 default and the nominal [AS2/Piper H adaptation](as2-umi-adaptation.md). The latter has CPU/MuJoCo basic-control evidence and a corrected 3-second Lab observation with four ground-supported feet throughout its last second. Its first Lab update started inverted because of an AS2-only quaternion conversion error; that evidence is retained as an unsuccessful adaptation record. After repair, a fresh 24-transition update had zero termination/collision metrics and 2.458 mean normal-contact supported feet, but 0.705 m position error. These earlier results qualify the bounded runtime path. The later [shared-geometry repair and full-size learning run](as2-umi-learning-plan.md) preserve the nominal control arrays exactly. The [completed 4000-update result](as2-umi-learning-result.md) improves Lab tracking while worsening support and inversion; all final MuJoCo cases and export consumption fail numerically. This does not establish stable WBC or successful transfer. Keep the working baseline intact and implement further adaptations in the actual owners:

| Adaptation | Changes that are presently required | Evidence before training conclusions |
|---|---|---|
| Asset and dynamics | Lab conversion and runtime asset paths, fixed-link collapse rules, collision shapes, inertias, actuators and limits; MuJoCo conversion's URDF-relative mesh resolution and base/freejoint assumptions | Independent FK, frame and inertia comparisons; no invented mass from collapsed marker links; correct collision geometry |
| Joint/action mapping | Lab joint-name mapping; MuJoCo `JOINT_NAMES`, state indices and current 18-action buffers; controller offsets/scales/gains/torque/delay arrays | Per-joint mapping and sign checks, bounded commanded torque and real action-delay timing |
| End-effector and observations | Saved binding's TCP offset/rotation, root and foot/body names, `RobotState`, observation scales/history and actor/critic input sizes | Matching target/state frames and observation values; old weights load only when dimensions **and semantics** agree |
| Training semantics | The environment's 18-action buffers and `actor[:,42:-18]` task slice, body/shape ordering in the privileged critic, mass/COM/friction randomization, body-name constraints and reset/termination | Real randomized properties reach the critic; reward and timeout behavior reflect intended new motion; no placeholder sensor values |
| Force reconstruction and contacts | Actual articulation topology, inertias, solved motion and foot indices in Lab; ground filters and contact classifications in both engines | Supported public-API signal validation and failure propagation; document differences from the original force sensor rather than requiring patched PhysX |
| Evaluation and export | Robot-specific evaluator state/frame/contact assumptions, actor dimensions and execution bundle joint names | Frozen-policy independent execution, full motion and failure reporting, exported actor parity and a fresh consumer |

A joint-count change propagates through all these consumers; it cannot be solved solely by changing a JSON array. Even a same-size arm replacement changes inertia, kinematics and attainable targets. Treat it as a robot adaptation requiring validation, not an unchanged original-method reproduction.

For another RL method, retain its defining policy inputs, objective, training stages, rollout semantics and deployment dependencies within its own method implementation. The current runner expects its concrete five-result environment contract and supports the supplied feedforward `Sequential` actor/critic construction; it is not proof that arbitrary methods fit unchanged. Extract shared utilities only when a second real consumer demonstrates matching semantics. No registry/plugin layer is required to begin that work.

Report experiments as original released-policy reference, Isaac Lab port, explicitly changed method/physical variant, or new-robot adaptation as appropriate. Name modifications, initialization, data partition, simulator versions and evaluation protocol. Keep original configurations and negative results available; do not infer success from a finite PPO update or a single upright trajectory.

## 7. Validate the changed path, not every historical experiment

Use tests that match the change. Existing [tests](../tests) cover UMI equations/optimization/resume, trajectory selection, FK/inertia, velocity frames, contacts, numerical failure propagation, substep trace non-interference and force reconstruction. The policy parity test requires the separate `pawcerto-umi-reference` environment with PyTorch3D; that environment is not necessary for normal CPU policy execution. Module-specific instructions are in the [training](../pawcerto/methods/umi_on_legs/training/README.md), [Isaac](../pawcerto/isaac/README.md) and [MuJoCo](../pawcerto/mujoco/README.md) READMEs.

Progress from relevant CPU checks to the actual affected simulator path, then to learning and full evaluation when the claim requires them. Preserve subprocess failures and invalid trajectory prefixes. Diagnostics must not insert extra physics solves to read a signal. Do not alter unrelated environments or rerun completed seeds simply to prove an unchanged utility. A scientific comparison needs unchanged conditions or an explicit account of what changed, and independent test data must remain separate from development choices.
