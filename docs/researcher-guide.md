# Using and extending PawCerto

PawCerto's goal is a research framework for reinforcement learning of whole-body control on quadrupedal manipulators. Today the implemented path is **UMI-on-Legs on Go2 + ARX5**, with Isaac Lab/PhysX training, independent MuJoCo execution, and actor export. This guide describes that working path and its concrete extension points. RoboDuet on Go1 + ARX5 now also has method-specific entrypoints; see [its training and export guide](roboduet-training.md). Its short Stage 1/2/resume validation proves integration, while effective learning remains unverified. DeepWBC, MLM and further robots remain planned. There is no method registry, robot plugin API or universal training command.

The existing three training seeds establish learning and sim2sim results on the supplied trajectory pool, with remaining head-contact and stability problems. They do not establish held-out generalization, reliable whole-body control or real-hardware readiness. Read [current results](umi-current-result.md) and [public reproduction status](release-reproduction.md) before comparing a new experiment. KISS My Agent is optional developer tooling, not a user or runtime dependency.

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
| Export | [scripts/export_umi.py](../scripts/export_umi.py) and the full checkpoint | `actor.ts`, execution `config.json`, and `joint_names.json` in a new directory |

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

The following is a **new split-training example**, not an executed result. It carries forward the existing body-speed-v3 reconstructed-force baseline and its explicit 1000 rad/s joint-speed override; it does not start a different native-limit recipe. The current project must finish the behavior-recipe decision before starting this long split run. If that decision changes the baseline, document the change and prepare a new input configuration first.

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

Training-partition qualification comes from the actual full checkpoint's embedded config, not its adjacent execution config. Missing metadata and actor-only exports are conservatively unqualified for held-out reporting. Keep the full checkpoint for validation/test identity even after exporting. A legacy full-pool policy evaluated on these IDs remains a full-pool policy.

## 4. Where the data and control flow live

These are concrete implementation boundaries, not a stable generic backend API:

| Owner | Inputs → outputs and responsibilities |
|---|---|
| [UMI execution module](../pawcerto/methods/umi_on_legs/__init__.py) | `PoseSequence` loads target arrays; `UmiObservation` turns `RobotState`, target previews and previous actions into actor observations; `UmiController` handles clipped actions, delay and explicit PD; `UmiPolicy` loads the actor/config |
| [Isaac runtime](../pawcerto/isaac/runtime.py) | `Go2Arx5Isaac` owns the asset, joints, actual randomized physical properties, reset/state and `step_torque`; [solver_force.py](../pawcerto/isaac/solver_force.py) reconstructs supported solver force signals |
| [Training environment](../pawcerto/methods/umi_on_legs/training/isaac_env.py) | `UmiIsaacTrainingEnv` owns episode targets, observation/control cadence, rewards, termination, reset and curriculum lifecycle |
| [Method semantics](../pawcerto/methods/umi_on_legs/training/semantics.py) | Original reward/critic equations, force-input selection and runtime compatibility; keep algorithm changes distinct from physical adaptation |
| [Runner and PPO](../pawcerto/methods/umi_on_legs/training/runner.py) | `UmiTrainer` gathers rollouts, updates the vendored PPO and saves full state; `reset()` returns actor/critic observations and `step()` returns actor/critic observations, reward, done and info |
| [MuJoCo runtime](../pawcerto/mujoco/runtime.py) and [asset conversion](../pawcerto/mujoco/asset.py) | Independent CPU robot state, contacts and delayed explicit PD; the evaluator reuses UMI observations and actor inference rather than replaying recorded joint positions |

The current actor/action path has 18 joint actions, ordered by the measured original asset mapping. Policy cadence is four 5 ms physics steps. Reset includes one zero-action policy step; history sentinels and pose-update cadence affect the policy. Training supports sparse and dense reward/task stepping; the author evaluator changes that configuration intentionally. Do not move updates between these clocks as an unnoticed “cleanup.”

`RobotState` contains root angular velocity and gravity in the root frame, joint positions/velocities in policy order, a 4×4 end-effector pose in the environment-local world frame, and episode time. Lab removes each environment's world origin. The current end-effector frame is reconstructed from `link6` and a fixed transform; using its body origin instead changes the task. MuJoCo exposes xyzw quaternions at the adapter boundary and corrects link-origin linear velocity relative to the inertial center.

## 5. Introduce new target trajectories

For the existing robot/task, a new trajectory file is already supported by `--trajectory`; no new training platform is required. `PoseSequence` currently loads a pickle list of episode dictionaries. Prepare files from a trusted source because this reader uses Python pickle.

- Required arrays are `ee_pos[T,3]` in metres and `ee_axis_angle[T,3]` as rotation vectors in radians, **not Euler angles**. Episodes must be stackable to the same time length. Targets share the environment-local world frame expected by the task.
- Sampling uses `sequence_sampler.dt`, not the dictionary's `t` timestamps. Resample irregular data offline to that grid; the current reader does not interpolate it. The requested episode length truncates long arrays and pads short ones with their final pose. Review whether a held final target is suitable for the new task.
- `planar_center` subtracts the mean xy position of frames 1, 2 and 3; this requires at least four available target frames. `add_random_height_range` applies a sampled vertical offset. Preserve or deliberately change these config semantics rather than pre-centering twice.
- Fields such as `gripper_width`, recorded velocity or acceleration are not consumed by this target reader. A new manipulation task requiring gripper control or force interaction therefore needs actual action/task implementation; adding fields alone does not provide those capabilities.
- Preserve recording identifiers when available. [scripts/split_umi_trajectories.py](../scripts/split_umi_trajectories.py) accepts a new source and output manifest, grouping exact target content and explicit `recording_id`, `source_recording_id` or `source_id` links. Pass both the new `--trajectory` and its new `--split-manifest` to consumers; old manifests reject a changed source hash.

Before robot execution, inspect all selected IDs, finite target values, shapes, coordinate transforms, durations, padding and reachable workspace. Run [data-split tests](../tests/test_umi_data_split.py) and check any reader changes against [policy/trajectory parity tests](../tests/test_umi_policy.py). Then test the actual full target motion, including approach, contact and exit where required. A pushing or simplified trajectory is a different task, not evidence of held-out tossing performance.

## 6. Adapt another robot or method

A different URDF is **not** sufficient to support a new quadrupedal manipulator. The current runtimes and several method tensors are specific to Go2 + ARX5. Keep the working baseline intact and implement the necessary changes in the actual owners:

| Adaptation | Changes that are presently required | Evidence before training conclusions |
|---|---|---|
| Asset and dynamics | Lab conversion and runtime asset paths, fixed-link collapse rules, collision shapes, inertias, actuators and limits; MuJoCo conversion's URDF-relative mesh resolution and base/freejoint assumptions | Independent FK, frame and inertia comparisons; no invented mass from collapsed marker links; correct collision geometry |
| Joint/action mapping | Lab joint-name mapping; MuJoCo `JOINT_NAMES`, state indices and current 18-action buffers; controller offsets/scales/gains/torque/delay arrays | Per-joint mapping and sign checks, bounded commanded torque and real action-delay timing |
| End-effector and observations | `link6` offset/rotation, base and foot/body names, `RobotState`, observation scales/history and actor/critic input sizes | Matching target/state frames and observation values; old weights load only when dimensions **and semantics** agree |
| Training semantics | The environment's 18-action buffers and `actor[:,42:-18]` task slice, body/shape ordering in the privileged critic, mass/COM/friction randomization, body-name constraints and reset/termination | Real randomized properties reach the critic; reward and timeout behavior reflect intended new motion; no placeholder sensor values |
| Force reconstruction and contacts | Actual articulation topology, inertias, solved motion and foot indices in Lab; ground filters and contact classifications in both engines | Supported public-API signal validation and failure propagation; document differences from the original force sensor rather than requiring patched PhysX |
| Evaluation and export | Robot-specific evaluator state/frame/contact assumptions, actor dimensions and execution bundle joint names | Frozen-policy independent execution, full motion and failure reporting, exported actor parity and a fresh consumer |

A joint-count change propagates through all these consumers; it cannot be solved solely by changing a JSON array. Even a same-size arm replacement changes inertia, kinematics and attainable targets. Treat it as a robot adaptation requiring validation, not an unchanged original-method reproduction.

For another RL method, retain its defining policy inputs, objective, training stages, rollout semantics and deployment dependencies within its own method implementation. The current runner expects its concrete five-result environment contract and supports the supplied feedforward `Sequential` actor/critic construction; it is not proof that arbitrary methods fit unchanged. Extract shared utilities only when a second real consumer demonstrates matching semantics. No registry/plugin layer is required to begin that work.

Report experiments as original released-policy reference, Isaac Lab port, explicitly changed method/physical variant, or new-robot adaptation as appropriate. Name modifications, initialization, data partition, simulator versions and evaluation protocol. Keep original configurations and negative results available; do not infer success from a finite PPO update or a single upright trajectory.

## 7. Validate the changed path, not every historical experiment

Use tests that match the change. Existing [tests](../tests) cover UMI equations/optimization/resume, trajectory selection, FK/inertia, velocity frames, contacts, numerical failure propagation, substep trace non-interference and force reconstruction. The policy parity test requires the separate `pawcerto-umi-reference` environment with PyTorch3D; that environment is not necessary for normal CPU policy execution. Module-specific instructions are in the [training](../pawcerto/methods/umi_on_legs/training/README.md), [Isaac](../pawcerto/isaac/README.md) and [MuJoCo](../pawcerto/mujoco/README.md) READMEs.

Progress from relevant CPU checks to the actual affected simulator path, then to learning and full evaluation when the claim requires them. Preserve subprocess failures and invalid trajectory prefixes. Diagnostics must not insert extra physics solves to read a signal. Do not alter unrelated environments or rerun completed seeds simply to prove an unchanged utility. A scientific comparison needs unchanged conditions or an explicit account of what changed, and independent test data must remain separate from development choices.
