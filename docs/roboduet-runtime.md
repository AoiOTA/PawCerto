# RoboDuet Go1 + ARX5 physics port

The source is `third_party/roboduet-reference` at
`a7e1528215c048199f90cb69ceb7749a1d745f28`. The default uses the existing official
Isaac Lab / Isaac Sim 6.1 installation and unmodified PhysX. No UMI runtime,
physics, reward or asset was changed.

## Implemented interface

`Go1Arx5Isaac(num_envs, device, usd_path)` provides `reset`, `set_stage`, `joints`,
`step_control`, `state`, and physical-property/state setters for the method's
training environment. `step_control(leg_torque, arm_target)` advances **one**
0.005 s physics substep. The controller recomputes the command four times per
policy step. Joint arrays have 20 columns in the actual Gym order stored in
`configs/roboduet_go1_arx5_joint_names.json`: **FL, FR, RL, RR** hip/thigh/calf,
then `zarx_j1` through `zarx_j8`. The URDF document order differs.

State positions are environment-local, linear/angular velocities are world-frame,
and quaternion order is wxyz. The EE is the original `zarx_body6` link frame;
no UMI gripper offset is added. Body arrays use `body_names`, and feet indices
follow Gym's FL, FR, RL, RR order. Material parameters apply to all robot shapes;
mass randomization selects the named body from `Cfg.domain_rand.base_mass_body`.
The PawCerto default is `trunk`, the explicit source defect repair described below.

Stage 1 writes all eight arm/gripper positions to their defaults and velocities
to zero immediately before each simulation call. Stage 2 writes only the two
gripper DOFs. Post-solve state is not overwritten: the same one-step drift allowed
by the original `_keep_arm_fixed` remains observable.

## The original M control mode does not drive arm position targets

The actual path writes the same combined tensor to effort and position-target
APIs, but every Gym DOF retains drive mode 3 (EFFORT). A fixed-root, no-gravity
original-asset probe measured the first arm joint starting at 0.3 rad:

| Input to joint 1 | Velocity after one 0.005 s step (rad/s) |
|---|---:|
| No input, stiffness 40 | -1.16899598 |
| Position target 0.6 only | -1.16899598 |
| Effort 0.6 only | -1.12934661 |
| Position and effort both 0.6 | -1.12934661 |
| Both 0.6, stiffness zero | 0.03793427 |

Thus the position-target write is ignored, while stiffness/damping remain active.
Lab preserves this with an implicit zero-rest spring/damper and the combined
command applied as effort. Treating the nominal arm target as an actual position
servo would change the reference method. This probe establishes actuator semantics;
it is not a matched Gym/Lab trajectory comparison.

## Asset and physical evidence

`outputs/roboduet-runtime-20260912/` contains the conversion, Gym readback, Lab
substep samples, and independent MuJoCo asset. The official conversion preserves
`dont_collapse` trunk/foot bodies and merges the fixed arm mounting into `base`.
Consequently `base` has mass 0.4403772056 kg, with no artificial 1 kg root body.
The mount is `(0, 0, 0.057)` with no rotation. Camera mass adds 0.1 kg to
`zarx_body6`; original Gym `recomputeInertia=True` readback confirms proportional
inertia scaling. The Lab and Gym body masses match, totaling **15.3086891994 kg**.
Their per-body inertia diagonal readbacks differ by at most 7.45058e-9 kg m²;
this comparison did not record the complete off-diagonal tensor.
The MuJoCo asset checks total mass, all 20 DOFs, fixed mount FK and all four feet.
It is an independent asset, not a RoboDuet MuJoCo policy evaluator.

The initial 16-environment Lab run completed 12 substeps in each stage without
nonfinite state. Maximum six-arm displacement was 0.00953515 rad in Stage 1 and
0.17677873 rad after release in Stage 2. The final Stage 2 gripper deviation was
5.99151e-7 rad. These are action-to-next-state and stage-mechanism results, not
control or learning success. The probe records post-solve values; the pre-solve
writes are code-level evidence, not separately sampled native readbacks.

After that run, the runtime was corrected to apply the original unconditional
`base.com = com_displacements` (zero when disabled), rather than retaining the
merged URDF COM. Randomization, nonzero reset velocity and checkpoint state APIs
were added for the training consumer. Their physical validation belongs to the
subsequent training integration output; the earlier stage numbers do not validate
these later additions. Root link velocity reset explicitly converts to COM
velocity before calling the public COM setter, avoiding the current official
link setter's COM/frame mismatch.

The original port read `ContactSensor.net_forces_w`, which this backend silently
aliases to **net normal** force with a warning. The corrected runtime enables
`track_friction_forces` and configures each contact body against the static ground
and every other collision body in its own environment. The official contact view
matches each environment's sensor to the corresponding filter body; it does not
construct all cross-environment pairs. The runtime reads `net_normal_forces_w`
and sums `friction_force_matrix_w` over those partners, preserving signed world
vectors before the reward takes a norm. Both official fields convert the latest
physics-step impulse using 0.005 s. The training environment samples the last
substep after decimation 4; it does not average or accumulate four force samples.

`--contact-only` runs a 16-environment sliding/intentional arm-self-contact probe.
The corrected capture in `contact-forces.pt/.json` measured:

- Filtered normal-force sums exactly matched unfiltered net normal force, residual 0 N.
- Ground friction reached 397.843 N and the intentionally overlapping arm pose
  produced 3568.005 N self-contact friction. These are diagnostic stress loads.
- Across both partners, self-contact normal and friction vectors each summed to
  zero, checking signs and avoiding double-counting the total on a body.
- Recombining saved normal, ground friction and self friction reproduced runtime
  totals to 0.0004883 N absolute error at the highest stress load (float32 sums).
- The real `Rewards._reward_tracking_contacts_shaped_force` differed by at most
  1.49e-8 on recorded policy boundaries, where most contact rewards were saturated.
  Applying the same function to all captured substep forces, with a fixed 0.5
  desired-contact input, showed a 0.0363728 difference at sliding substep 30.
  This latter calculation is a sampled-input diagnostic, not an extra policy reward.

The first capture exposed a runtime quaternion-contract error: official Lab 3
ProxyArray quaternions and pose setters use xyzw, while this runtime promised
wxyz to its training consumer. Public state output and explicit reset input now
convert at that boundary. Native default poses and the native COM velocity
calculation remain xyzw. The first failed-condition capture is retained as
`contact-forces-quaternion-mismatch.*`. Prior three-update training output used
the earlier normal-only signal and incorrect quaternion boundary; it cannot
validate the corrected interface. A new training integration record is required.
The corrected force capture proves this runtime's public contact-data composition
and exercised partner coverage, not cross-engine equality of entire trajectories.

## Observed source random-mass problem

`auto_train` samples body-0 added mass from `[-2, 2]` kg, although the actual body-0
mass is only 0.440377 kg. The original Gym floating-base setter probe returned:

| Requested mass | Setter return | Actual mass | Actual diagonal inertia |
|---|---|---:|---|
| 0.8 | True | 0.80000001 | positive |
| -0.5 | True | 0 | negative |
| Restore 0.44037721 | True | 0.44037721 | remained negative |

Each subsequent isolated single step remained finite, which does not make the
mass/inertia valid. The raw record is `gym-drive.json`, and original process output
is `gym-mass.log`. The authorized PawCerto repair explicitly sets
`Cfg.domain_rand.base_mass_body = "trunk"` and applies the unchanged `[-2, 2]` kg
range to the measured 5.2049999237 kg torso. Runtime resolves that name, retains its
original valid mass/inertia, and scales inertia proportionally from that baseline.
This avoids the invalid mount-body randomization without disabling randomization
or silently clamping samples. It is a declared source-method difference.

## Commands and observed recovery

From the project root, using the existing Conda interpreters:

```bash
OMNI_KIT_ACCEPT_EULA=yes PYTHONNOUSERSITE=1 \
 /home/lyb/miniconda3/envs/pawcerto-lab-sim610/bin/python \
 scripts/convert_roboduet_usd.py --viz none --device cuda:0

LD_LIBRARY_PATH=/home/lyb/miniconda3/envs/pawcerto-umi-reference/lib \
 /home/lyb/miniconda3/envs/pawcerto-umi-reference/bin/python \
 scripts/probe_roboduet_runtime.py --engine gym

OMNI_KIT_ACCEPT_EULA=yes PYTHONNOUSERSITE=1 \
 /home/lyb/miniconda3/envs/pawcerto-lab-sim610/bin/python \
 scripts/probe_roboduet_runtime.py --engine lab --viz none --device cuda:0

# Full normal plus friction capture and real reward-consumer diagnostic:
OMNI_KIT_ACCEPT_EULA=yes PYTHONNOUSERSITE=1 \
 /home/lyb/miniconda3/envs/pawcerto-lab-sim610/bin/python \
 scripts/probe_roboduet_runtime.py --engine lab --contact-only --viz none --device cuda:0

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
 /home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python \
 -m pytest tests/test_roboduet_asset.py -q
```

Conversion, Gym drive/mass probes, the initial Lab run and the asset test exited
zero. The asset test passed once. Earlier failed Gym imports and the incorrect
URDF-order assumption are preserved in the output logs. The Gym environment lacks
MuJoCo, while the existing `pawcerto.mujoco` initializer eagerly imports it; the
standalone asset helper is loaded directly by Isaac/Gym entrypoints to avoid
changing that unrelated initializer. Pytest's automatic ROS plugin loading failed
on an unrelated missing `yaml`; disabling external plugin autoload let the exact
asset test execute. No installation or physics-engine rebuild was needed.
