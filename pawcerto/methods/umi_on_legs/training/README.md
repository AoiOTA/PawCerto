This is the released UMI-on-Legs PPO and a tensor/Isaac Lab adaptation of its
original tossing task. `ppo.py`, `actor_critic.py`, `storage.py`, and
`trajectory_utils.py` retain upstream source and license headers. Changes to
those files are import paths and the upstream `git.Optional` typo only.
The task equations in `semantics.py` derive from the MIT-licensed UMI-on-Legs
source; see `../LICENSE`.

`UmiTrainer` consumes the original five-result environment interface. The
actor and critic architectures and all PPO parameters come from the supplied
checkpoint config. Defaults are 24 transitions per environment, 64 learning
epochs, four minibatches, adaptive learning rate, and the original timeout
bootstrap using the pre-step value estimate. Clipping, rewards, trajectory
sampling, observation noise, randomized PD, global curriculum EMA, indexed
resets, and perturbation cadence are managed by `UmiIsaacTrainingEnv` plus the
physical runtime. Critic setup/state groups are each sorted by attribute name,
matching the real upstream environment, before the task and action blocks.

The Isaac runtime must provide actual applied randomized properties and body
contact observations. The current port reads normal-contact forces from
`ContactSensor.net_forces_w`, which may include self-contact normal forces.
A nominal-property placeholder is not an equivalent training
run. Original fixed-joint collapse and the 34 collision-shape identities matter
for the privileged critic. The physical backend owns those correspondences.

The current launcher uses Conda `pawcerto-lab-sim610` and the fixed official
Isaac Lab/Sim 6.1 installation in the [installation guide](../../../../docs/isaaclab-install.md).
The original measured Gym joint order is committed in
`configs/umi_go2_arx5_joint_names.json`; a new checkout does not need Gym to query it.
From the repository root, after converting the robot:

```bash
export PAWCERTO_ROOT="$PWD"
conda activate pawcerto-lab-sim610
export PAWCERTO_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=yes
cd "$PAWCERTO_ROOT/third_party/IsaacLab-develop-sim610"
"$PAWCERTO_PYTHON" "$PAWCERTO_ROOT/scripts/train_umi.py" --viz none --device cuda:0 \
  --joint-names "$PAWCERTO_ROOT/configs/umi_go2_arx5_joint_names.json" \
  --num-envs 4096 --output "$PAWCERTO_ROOT/runs/umi_training_example"
```

This command retains the source config's native joint-speed limits. To reproduce
the current trained physical variant, use the [root training recipe](../../../../README.md),
which explicitly sets `joint_velocity_limit_override_rad_s=1000.0`, 4000 updates,
and a fresh output directory. `--viz none` belongs to the current development
launcher; historical release examples used `--headless`.

`--weights "$PAWCERTO_ROOT/reference/checkpoints/tossing/ours/model.pt"` initializes the original
actor and critic without continuing their optimizer. `--resume` restores model,
optimizer, iteration, learning rate, CPU/CUDA RNG and the environment curriculum
when available. Physics resets on resume; these checkpoints do not restore an
identical in-flight simulator trajectory. Original upstream checkpoints have
no RNG or curriculum snapshot.

`metrics.jsonl` records optimization, rewards, EE position/orientation errors,
root height and support counts together. A finite update, a falling trajectory,
or a short no-fall run cannot establish learned whole-body control. The CPU
tests validate equations, source reuse, transport, optimization and resume,
without making a robot-control claim.

The current default EMD input is reconstructed solver-only foot world Fz through
supported public PhysX APIs (`--force-signal reconstructed-solver`); no physics
engine rebuild is required. `--force-signal normal-contact` explicitly selects
the earlier proxy. Historical runs passing only foot normal-contact Fz were not
equivalent to the original sensor, including contact-free airborne states.
See [current force semantics](../../../../docs/umi-reconstructed-force-training.md)
and the [historical consumer readout](../../../../docs/umi-force-sensor-consumers.md).
Actor/critic do not directly consume foot forces in the active saved config.
The reconstruction and three completed training seeds are current results;
exact Gym/Lab signal identity and reliable whole-body behavior are not established.

`supported_feet` counts this current port normal-contact Fz above 1 N;
`ground_supported_feet` separately counts ground-filtered normal Fz above 1 N
from `force_matrix_w`. Neither is the original Gym forceSensor signal. Old
summary metadata saying "original foot net solver Fz" is incorrect; existing
raw arrays, summary values, field names, and reward formulas remain unchanged.
