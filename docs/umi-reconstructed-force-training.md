# UMI training with reconstructed solver force

Training and author-style evaluation now default to `--force-signal reconstructed-solver`. The adapter feeds the current PhysX solver-only foot world-Z reconstruction into **EvenMassDistribution only**. It leaves `raw['feet_force_z']`, collision forces and ground-filtered forces as their existing normal-contact diagnostics. Actor and critic dimensions and observation order are unchanged.

For dense rewards the adapter selects the signal after every real physics substep; for sparse rewards it selects the final substep's signal. `feet_solver_force_z` must have shape `[num_envs, 4]` and finite floating-point values, and `feet_solver_force_valid` must be a fully true boolean vector of shape `[num_envs]`. The runtime invalidates old values after resets or state writes and publishes a fresh result only after a supported physics step. A missing or invalid reconstruction raises an error; it never falls back to contact forces. Observation-only reads after reset do not consume EMD and can precede the next valid reconstruction.

The original EMD calculation is unchanged: clamp Fz at zero, divide by total Fz plus `1e-8`, apply the original unbiased standard deviation and configured power, use the original flying branch when the normalized sum is too small, and multiply by the original weight. No force clipping, normalization, reward weight or observation term was added to make reconstruction pass.

## Saved semantics and initialization

The saved `config.json` and the existing checkpoint `config` field include:

```json
{
  "pawcerto_runtime": {
    "force_signal": "reconstructed-solver",
    "physics_revision": "umi-physics-mapping-v3",
    "reconstruction_revision": "stock-physx-body-speed-damping-v3",
    "joint_friction_model": "physx-legacy-friction-coefficient"
  },
  "training_initialization": {
    "mode": "random",
    "checkpoint": null,
    "source_pawcerto_runtime": null
  }
}
```

The physics revision identifies explicit application to every rigid body, conversion of the original angular limit to degree units, original URDF joint velocity limits and contact/rest offsets. The current reconstruction revision also includes the source-supported body-speed damping contribution, in addition to the distinct forward-dynamics/public-C DOF rules and bias/gyro correction. Complete native 6D numerical comparisons still have documented failures; this revision does not imply sensor equivalence. Historical completed model4000 retains its original velocity-clamps-v2 metadata. An explicit research override of joint limits to 1000 rad/s is separately recorded as `joint_velocity_limit_override_rad_s`; omission retains URDF limits. Joint friction uses the current verified legacy **coefficient** API. Earlier static-effort mapping results and earlier normal-contact proxy training retain their historical metadata and scope; they are not silently relabeled as this configuration.

`--resume` restores optimizer, iteration, RNG and task state only when the checkpoint's runtime contract exactly matches the current contract. Both the CLI and direct `UmiTrainer.load(..., load_optimizer=True)` check before loading model or optimizer state. A default `UmiTrainer(env)` takes the actual adapter config and binds its runtime contract; an explicit trainer config cannot override the environment contract. Missing legacy metadata, a different force signal or a different physics/friction revision is rejected. The environment also validates its saved training-state contract.

`--weights /path/model.pt` explicitly starts a new weight chain from actor/critic/std weights, with fresh optimizer and iteration zero. It does not resume the source run. The initialization mode, source path and any source runtime contract are saved separately. Without `--weights` or `--resume`, the model starts randomly; the completed 4000-iteration chain used this path. The separately documented [adaptation experiment](umi-relaxed-velocity-adaptation.md) uses weights-only initialization and an explicit physical variant.

Evaluation can load historical weights under an explicitly chosen current signal. Its resolved config and summary record the current contract and the checkpoint's training contract separately. A missing old training contract remains unknown; choosing a new evaluation signal does not prove that the checkpoint was trained with that signal.

## Commands for the designated GPU operator

The preselected experiment is described in [the corrected training plan](/home/lyb/PawCerto/docs/umi-corrected-training-plan.md). The selected fixed Lab/Sim 6.1 environment passed the separate integration check. Its local launcher setup is:

```bash
cd /home/lyb/PawCerto/third_party/IsaacLab-develop-sim610
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=yes
export UV_PROJECT_ENVIRONMENT=/home/lyb/PawCerto/.venvs/isaaclab-sim610
export UV_CACHE_DIR=/home/lyb/PawCerto/.venvs/isaaclab-sim610-cache
export UV_PYTHON_INSTALL_DIR=/home/lyb/PawCerto/.venvs/isaaclab-sim610-python
PAWCERTO_UV=/home/lyb/PawCerto/outputs/isaac/lab-develop-sim610-validation/uv-x86_64-unknown-linux-gnu/uv
```

The following records the earlier random-weight 4000 invocation and its existing output directory. The ongoing weights-only physical variant uses its separately recorded [execution plan](../runs/umi_reconstructed_solver_relaxed_body_speed_adapt_seed0_4096/execution-plan.json) and [experiment definition](umi-relaxed-velocity-adaptation.md). The GPU operator retains exact source identity and exception-capture wrappers alongside each run:

```bash
"$PAWCERTO_UV" run --no-sync --extra isaacsim python /home/lyb/PawCerto/scripts/train_umi.py \
  --config /home/lyb/PawCerto/runs/umi_reconstructed_solver_velocity_seed0_4096/input-config.json \
  --joint-names /home/lyb/PawCerto/reference/isaac/original_gym_joint_names.json \
  --num-envs 4096 --iterations 4000 --save-every 100 --seed 0 \
  --force-signal reconstructed-solver \
  --output /home/lyb/PawCerto/runs/umi_reconstructed_solver_velocity_seed0_4096 \
  --viz none --device cuda:0
```

The input config preserves the original recipe and adds engine provenance. `--viz none` replaces the removed `--headless` option on this development snapshot; the preserved release environment still uses `--headless`. The integration check's weights are not resumed into this random-initialization experiment. Saving every 100 iterations changes storage frequency only; evaluation remains at 0/500/4000. The earlier `umi_reconstructed_solver_seed0_4096` and `_resume` attempts retain their external SIGTERM and internal velocity-guard failures. Their checkpoints cannot resume an optimizer under this changed runtime contract. `--iterations` specifies additional iterations in an invocation, not a replacement final checkpoint number. Physics resets on resume, so recovery is not an uninterrupted trajectory replay.

Final author-style evaluation uses the preselected seed and preserves the trained policy's saved inputs:

```bash
"$PAWCERTO_UV" run --no-sync --extra isaacsim python /home/lyb/PawCerto/scripts/eval_umi.py \
  --checkpoint /home/lyb/PawCerto/runs/umi_reconstructed_solver_velocity_seed0_4096/model_4000.pt \
  --seed 2027 --force-signal reconstructed-solver \
  --output /home/lyb/PawCerto/outputs/isaac/umi_reconstructed_solver_velocity_seed0_4000_seed2027 \
  --viz none --device cuda:0
```

Use `--force-signal normal-contact` explicitly for the previous reward proxy. That switch changes only which foot signal EMD consumes; it does not undo the current body-property or legacy-friction fixes, so it is not a replay of the full old physics configuration. Missing reconstruction data is tolerated only in this explicitly selected proxy mode.

## Verification boundary

The initial signal integration passed **30 CPU tests**: the existing 25 checks plus five new signal/contract checks. The new adapter tests set contact forces to zero while reconstructed forces are nonzero, cover dense and sparse rewards, verify the original clipped-force variance numerically (`[-2,0,2,6]` gives penalty `1/8` before weight), retain the original all-zero flying penalty, and reject missing, stale/invalid and nonfinite reconstruction. Checkpoint tests verify metadata persistence, direct-API rejection before model mutation, and explicit weights-only initialization without optimizer restoration. The existing privileged critic ordering, actor/critic weights, PPO, reset and numerical-failure tests still pass.

Both CLI help paths, supplied with their required arguments, exited zero with GPU visibility disabled, and all changed modules compiled. The CPU suite used the existing training environment with GPU visibility disabled; its already installed Torch/NumPy were imported before appending the old reference environment solely for PyTorch3D's pure transform source. No packages were installed for these checks. See [CPU log](/home/lyb/PawCerto/outputs/analysis/umi-emd-reconstruction-cpu-tests.log) and [verification record](/home/lyb/PawCerto/outputs/analysis/umi-emd-reconstruction-verification.json).

These checks establish source selection and training interfaces, not learned control quality. Actual runtime integration, resource measurements and training are executed by the designated GPU operator. Current PhysX reconstruction evidence does not claim pointwise identity with the original Gym GPU force-sensor implementation. After the complete 48-case version regression, the plan explicitly selects the isolated MuJoCo 3.13.0 environment for the fixed 0/500/4000 checks. This decision precedes corrected long training; the earlier 3.8 environment and results remain preserved.

The later DOF/gyro correction passes **38 CPU tests** and all 450 original-tolerance native acceleration checks in CPU64, CPU32 and actual CUDA32. The [current report](../outputs/isaac/dof-clamp-free-diagnostic/cpu-production/README.md) retains the separate earlier high-gyro failures. Current solver-free CUDA residuals still change one of three original EMD flying decisions; the formula is not altered to hide that difference. The [latest main-code PPO check](../outputs/isaac/dof-clamp-one-iteration/validation.json) completed one full original iteration with finite checkpoint and metrics. Learned behavior remains a separate evaluation.

The later body-speed extension preserves the original EMD formula and adopts source multiplication order. Its analytical tests pass, while the two qualified actual-CUDA samples retain 10/150 and 1/150 out-of-tolerance components. Near-zero force residuals also change the EMD branch. See the [complete numerical readout](../outputs/diagnostics/umi-body-speed-damping/README.md); no tolerance was relaxed and the ongoing adaptation is explicitly a reconstructed-signal research variant. Exact historical commands and source identities are retained in each run’s execution records.
