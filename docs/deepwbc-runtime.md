# DeepWBC official Isaac Lab runtime preparation

`pawcerto/isaac/deepwbc_runtime.py` implements the Go1 + WidowX runtime adapter against the installed official Isaac Lab / PhysX APIs. `scripts/convert_deepwbc_usd.py` prepares the original URDF and provides its conversion entry point. Initial validation used CPU preparation and standalone USD tests. Subsequent device-owned conversion and bounded real wrench/material/geometry probes are recorded below. The corrected-terrain fresh run completed 21 updates and a fixed 500-step evaluation on the original robot and terrain. All 500 evaluation samples have physical contact and lie inside the terrain. Exported policy actions match the checkpoint exactly on every recorded observation. Tracking remains poor and one signed-roll failure occurred; these results establish the bounded execution path, not learned task success. The earlier zero-contact run is preserved as an invalid placement attempt. See [the final integration report](experiments/deepwbc-reconstructed-runtime-20260912.md).

## Preparation and source preservation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/convert_deepwbc_usd.py --prepare-only
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest tests.test_deepwbc_runtime -v
```

The second command was run with the existing `pawcerto-lab-sim610` Python, which provides standalone USD bindings. It starts no Isaac application. Ten tests pass: physical URDF preservation, named mount/tree, six-axis frame math, import isolation, shared USD geometry authoring, full inertia tensor restoration, missing-source-origin COM restoration, startup finger-pose adaptation, restoration of the measured unbounded waist semantics, and safe material-payload validation.

Preparation writes ignored `reference/isaac/go1_widowx/source.urdf` and `source-manifest.json`. Source revision remains `8159e4ed8695b2d3f62a40d2ab8d88205ac5021a`. The source has total declared mass **14.150879 kg**, **25 collision elements**, and **20 moving joints**. The Go1/WidowX mount remains `[0.03,0,0.057]` with zero rotation. Mesh filenames become absolute local references; link names replace `/` with `_` for USD identifiers. Joint names, axes, inertia tensors, limits, collision geometry, and the right-finger mimic relation remain unchanged.

The original `widow_waist` is declared revolute but has no lower/upper limits. An initial preparation attempt incorrectly required these fields and failed; the repair retains their absence as JSON `null`. No angle bounds or joint type are invented. The runtime reads actual imported position limits. The actual importer/Gym limit discrepancy and its verified correction are recorded below.

For conversion, original `dont_collapse` joints protect the trunk, four feet, and massive EE marker while the official fixed-joint merger runs. Other fixed links merge through the official importer. The converter preserves explicit effort control, requests source density, and retains self-collision. It restores full source COM inertia tensors through USD principal axes, checking mass/COM and the reconstructed tensor rather than assuming the importer uses a particular quaternion orientation.

Collision offsets (`0.01`, `0`) and source cylinder-to-capsule semantics are authored in the **owning shared geometry layer**, before environments are cloned. There is no de-instancing fallback. Verification traverses instance proxies to confirm composed properties and requires collision-bearing instance geometry. A real standalone USD test with two references verifies that both still share one prototype, their collision primitives become capsules, and the contact offsets compose correctly. This fixture validates shared USD authoring; the successful original-asset conversion is recorded below. Memory capacity at thousands of environments remains unverified.

The full conversion entry point starts AppLauncher and is operated only by the designated device owner:

```bash
python scripts/convert_deepwbc_usd.py --output reference/isaac/go1_widowx --visualizer none --device cuda:0
```

The first device execution failed during COM validation; see below. Generated USD paths are recorded only after collision and inertia checks complete. Upstream assets remain ignored local inputs and are not redistributed.

## Runtime and environment interface

Construct `Go1WidowXIsaac` after AppLauncher with explicit `terrain_cfg` and `terrain_contact_paths`. The terrain owner supplies the original Perlin mesh; the runtime supplies no plane replacement. `env_origins` defaults to zero, so the task's source-sampled common-world positions remain common-world positions even though Lab initially creates clone namespaces on a grid. Collision groups still separate environments. An explicit offset tensor is supported, but source task integration uses zeros and samples its original world origins itself.

The runtime uses 5 ms physics steps, four per policy step, explicit position PD, 18 policy actions, and two zero finger torques. Names are mapped from actual imported joint names to original simulator order: FL, FR, RL, RR (hip/thigh/calf), waist, shoulder, elbow, forearm-roll, wrist-angle, wrist-rotate, left/right fingers. It does not reduce this to a 15-DOF robot; only 15 target scales are nonzero in the public configuration. No complete 6D manipulation capability is claimed.

| Method | Contract |
|---|---|
| `push_actions(policy_actions)` | `[N,18]` policy order; append the undelayed action and select the original two-step delay |
| `step_control()` | Compute explicit PD from current state, apply 20 efforts, and step once; returns state |
| `joints()` | Source-ordered joint position and velocity, both `[N,20]` |
| `reset(ids, root_states, joint_pos, joint_vel)` | Task-provided position, **xyzw** quaternion, world link linear/angular velocity, and source-ordered joint state; clears controller history |
| `apply_creation_randomization(...)` | `mass_params[N,5]`, `friction[N,1]`, `motor_strength[N,18]`, box added mass; public setters |
| `reset_box(ids, positions)` | Write XY only; retain actual box Z, orientation and velocity as original episodic reset does |
| `push_base_velocity_xy(velocity)` | Overwrite world base-link XY velocity while preserving the other velocity components |

`state()` returns `root_states[N,13]`, `dof_pos`, `dof_vel`, `torques[N,20]`, delayed `actions[N,18]`, `latest_sim_actions[N,18]`, `ee_pos`, `ee_orn` (xyzw), EE velocities, complete normal+tangential `contact_forces[N,B,3]`, `foot_wrenches[N,4,6]`, `force_signal`, and time. Foot order is FL, FR, RL, RR. The environment owner handles reset ordering, observation history, goals, two reward channels, termination, and the original initial zero-action step.

Root link velocities are converted to native COM velocities using the current body COM offset when written through the public setter. Creation mass changes scale source inertia with mass at fixed geometry and apply source COM deltas to the original root mount body rather than silently choosing the torso. This implements the original request to recompute inertia on mass changes, but equality to Gym's internal recomputation remains unverified. The original sampler can produce negative friction values. The runtime preserves those samples and represents their measured contact behavior with external/self material channels and single-membership collision groups; the actual validation and limitations are recorded below.

## Explicit six-axis force modes

The runtime supports `force_signal='reconstructed_sensor_wrench'` and
`force_signal='native_joint_wrench'`. The constructor retains the native default;
fresh training requires an explicit choice, checkpoints record that choice, and
resume rejects switching modes. The state exposes the selected identity. The
original consumers require all six channels: the contact observation tests their
combined norm against 1.5, and the foot-force reward squares the third force
component. Neither mode fills torque slots with zero or drops force channels.

Original DeepWBC constructs each Gym sensor at zero relative pose with default
properties: forward-dynamics and solver forces both enabled, local body axes.
The simultaneous original-Gym probe confirmed default equals explicit Both,
Both equals FD plus constraint (maximum force difference 3.8e-6 N), and FD equals
foot-body gravity in the PRE frame (maximum component difference 2.257e-7 N).
The actual four foot COM offsets are zero; nonzero-lever physics is untested.
This empirical result, rather than an assumed generic net-contact-force
interpretation, motivates the reconstructed mode.

`reconstructed_sensor_wrench` executes the original PD every 5 ms, refreshes PRE
COM kinematics, reads public M/G/C/J and then native COM/DOF velocities and
actuation, and predicts free acceleration with `StockSolverForce.predict`.
The native parent order, all 27 physical bodies and all 20 DOFs including fingers
are retained. Public C is read before velocity because its getter affects that
readback's velocity preprocessing. After the physical step, the runtime calls
`StockSolverForce.approximate_sensor_wrench`:

```
F_world = m * (a_reported - a_free) + m * g
T_origin_world = I_PRE * (alpha_reported - alpha_free) + (R_PRE * r_COM) cross F_world
wrench_body = [R_PRE.T * F_world, R_PRE.T * T_origin_world]
```

The angular residual uses the full PRE inertia at COM in rigid-body-prim axes.
It does not add a second gyroscopic term after subtracting free acceleration.
The four feet are selected after full-body reconstruction. At each 20 ms policy
step, the unchanged task consumes the fourth 5 ms sample. Model initialization
reads actual mass/inertia/COM after creation randomization; subsequent creation
writes invalidate that cache. Reset clears selected sensor rows. Unsupported
external wrenches, model properties or native/user ordering mismatches fail.

The real three-scene comparison and full arrays are in
`outputs/isaac/deepwbc-solver-wrench-lab-20260912-attempt3`. All 120 short-probe
contact bits and consumer-last-sample bits matched original Gym, while
contact-scene Fz² MAE remained approximately **10–13% of the original mean**.
Moving-contact maximum single-step force/torque errors were 15.658 N/.01235 Nm;
consumer-sample maxima were 2.028 N/.001502 Nm. PRE frame alignment did not
explain the principal moving-contact peak. Different contact solutions coexist
with these errors, so this separate-trajectory probe does not isolate all
reconstruction error or establish exact sensor equivalence. These error figures
belong to that specific short-probe material setup, not a measured error bound
for randomized production training. See
[the complete comparison and interface](experiments/deepwbc-solver-wrench-20260912.md).

`native_joint_wrench` remains a diagnostic alternative. Official Lab exposes
incoming reaction in the child-side joint frame at its anchor. The adapter reads
actual `localPos1`/`localRot1`, rotates force and torque and shifts the moment:
`F_link=R*F_joint`, `T_link=R*T_joint+r_link_to_anchor cross F_link`.
Independent CPU tests cover nonzero force, torque, rotation and lever arms.
The measured original-Gym comparison rejected this native reaction as an
equivalent sensor signal, despite the correct transform.

The original UMI `residual_force` and `predict` numerical paths are unchanged;
only the explicitly approximate six-axis method is added and consumed by
DeepWBC. Nine solver/math CPU tests pass, including non-diagonal inertia and
nonzero COM lever. Final-method CPU replay of all 120 recorded rows reproduces
the CUDA candidate within 3.815e-6 N and 1.863e-9 Nm. Additional CPU runtime tests
cover PRE/POST order, full-body-to-foot selection, randomized cache refresh and
resume-mode identity. These are bounded implementation checks, not training or
sensor-equivalence proofs. Official unmodified Isaac Lab/PhysX remains the
dependency; no engine rebuild is required.

## Remaining evidence boundary

Actual conversion, original-asset short trajectories, six-axis approximation
and their failures are recorded below and in the linked signal report. Exact
sensor parity, PhysX mimic behavior, randomized inertia/material equivalence,
physical stability, sustained training memory and learning remain distinct
qualification limits. The explicit reconstructed-sensor 16-env/21-update run,
model-21 export and fixed 500-step evaluation completed after terrain transform
repair. Physical contact is exercised, but EE RMSE .47936 m, forward-velocity
MAE .49878 m/s and one signed-roll failure do not demonstrate learned task
success; the remaining 455-sample episode is unfinished. See
[the integration record](experiments/deepwbc-reconstructed-runtime-20260912.md).
CPU checks or completed isolated probes cannot establish these remaining
outcomes. Subsequent device-owned execution preserves causes and reports only
the evidence reached.


## Actual conversion follow-up

The first device-owned conversion exited **1** with a preserved traceback in `outputs/deepwbc-native-wrench-execution-20260912/convert.log`. The importer generated `reference/isaac/go1_widowx/merged/merged.usda`, then COM validation found `[-inf,-inf,-inf]` instead of the source origin `[0,0,0]`.

CPU inspection identified exactly five affected bodies: `wx250s_ee_gripper_link` and all four feet. Their URDF inertial blocks omit `origin`, meaning zero in URDF, while USD leaves `centerOfMass` unauthored, meaning automatic computation through the schema's `-inf` sentinel. The converter now authors the actual source COM only when the imported COM has no authored value. An authored mismatch still fails and identifies its body.

The regression test reproduces the real unauthored USD attribute. All seven CPU tests pass, and an in-memory repair of the actual failed converted stage verified all 27 rigid bodies, with maximum reconstructed inertia-tensor error `3.06e-9`. The failed disk artifact was preserved. This repair does not itself establish a successful full conversion or physical run; the device owner continues the authorized execution.


The device-owned conversion subsequently completed and wrote `usd_path.txt` pointing to `reference/isaac/go1_widowx/merged_1/merged.usda`. The first 16-environment run then exited **1** at Lab's default-pose validation: the source finger defaults are zero, outside their original URDF ranges ±[0.015,0.037]. The runtime now uses the nearest valid finger endpoints **only for Lab construction**. The task/controller defaults and the original task's explicit zero-position reset writes remain unchanged before the first policy physics step. The corresponding source-backed CPU regression passes.

That run also reported unsupported GPU contact filters for the dynamic Box's collider mesh. Installed official Lab examples address dynamic contact targets by their **rigid-body root**, so the filter was corrected from `.../Box/geometry/mesh` to `.../Box`. This preserves the intended normal+tangential readout; the next real run must confirm backend acceptance. The complete first-run traceback and warnings remain in `outputs/deepwbc-native-wrench-execution-20260912/train-attempt1.log`.

Standalone inspection of the actual converted USD found all 20 moving joints, but the importer authors waist position limits `[0,0]` when the source revolute joint omits limits. This is recorded rather than silently changing the asset to continuous. Original Gym asset behavior must be measured before claiming physical action-channel equivalence. The original mimic XML is retained; the generated USD contains Newton mimic metadata, while actual PhysX mimic behavior is also unverified.


A second original Gym probe resolved the waist ambiguity with the missing **`hasLimits` flag**. In `outputs/isaac/deepwbc-wrench-gym-20260912-originalprops/dof-properties.json`, both the asset and actor report `hasLimits=False`; the asset stores ±FLTMAX and the actor stores ignored numeric zeros. The converter therefore now restores source-omitted revolute limits to USD `[-inf,+inf]`. This changes only the generated waist limit attributes and preserves all explicitly bounded joints and the original URDF. The new regression and the in-memory operation on the actual converted USD pass; nine CPU tests pass in total. The device owner must regenerate the validated USD before continuing, so the known locked import is not used for the short execution.


The corrected conversion completed at `merged_2/merged.usda`. The next 16-environment runtime passed joint initialization and initialized a 27-body joint-wrench sensor; its actual PhysX joint table reports waist limits approximately `[-3.4e38,+3.4e38]`, following the authored unbounded USD limits. The earlier dynamic Box contact-filter warning is absent after the rigid-body-path repair.

This second run still failed before any update/checkpoint. Its preserved log (`train-attempt2.log`) contains **two distinct failures**: the original single 11,978,802-triangle terrain exceeded PhysX BV4 child-node limits, and the material setter received a negative dynamic friction coefficient and then segfaulted. The terrain owner is preserving all source triangles in supported static mesh tiles. The runtime material layout was checked against the official tensor API: `[N,max_shapes,3]` ordered static friction, dynamic friction, restitution. There was no axis-layout mismatch.

The original public sampler deliberately includes negative values in `[-0.5,3]`. The runtime now rejects nonfinite/negative friction with a descriptive Python exception **before native mutation**, preserving the original values and preventing the observed backend crash. It does not silently clamp or resample. The original Gym handling is being probed before selecting any physical mapping. Ten CPU tests pass, including material layout, coefficient validation, and unchanged restitution. Neither this guard nor the geometry repair constitutes learning or force-sensor equivalence evidence.


Device-owned box-plane sliding probes measured the original Gym treatment rather than relying on property readback. In `outputs/isaac/deepwbc-gym-slide-ground1-20260912/analysis.json`, raw shape friction `[-0.5,-0.25,0,0.25,0.5,1]` produces effective dynamic coefficients approximately `[0.2500,0.3750,0.4997,0.6251,0.7487,1.0087]` against ground friction 1. Against ground friction 0 (`deepwbc-gym-slide-ground0-20260912/analysis.json`), they are `[0,0,0,0.1250,0.2500,0.4997]`. These conditions support averaging the pair before truncating negative effective friction, rather than clamping, taking absolute values, or replacing negative shape values before combination. This is bounded box-plane motion evidence, not proof of all robot contacts or static-friction behavior. A single nonnegative robot material cannot in general preserve these measured pairs with different opposing materials using the same averaging rule. The runtime therefore retains its explicit rejection while the integration owner resolves the supported-API fidelity boundary; no sampler distribution change is made.


### Independent two-channel material probe

The original task uses terrain friction 1 and leaves the box's original Gym material at its measured default 1. The integration candidate retains raw sampled `f`: robot external colliders use `(f+1)/2` with multiply combination, robot self colliders use `max(0,f)` with average combination, and ordinary terrain/box retain 1. Body/collider `UsdPhysics.FilteredPairsAPI` relationships target shared collider instance proxies to select the channel. This is a contact-pair representation, not a change to the latent sample; it is not yet installed in the training runtime.

`scripts/probe_deepwbc_material_pairs.py` uses `pawcerto/isaac/deepwbc_material_probe.py` to test raw `[-0.5,-0.25,0.25,1]` against ground, box, and a second robot body, plus box-ground and complete pair isolation. The first device execution (`outputs/isaac/deepwbc-material-pairs-20260912/analysis.json`) exited 1 after preserving all motion. All eight robot-box/robot-robot cases passed; measured external coefficients were approximately `[0.250001,0.375004,0.624912,1.002280]` and self coefficients `[0,0,0.250001,1.002281]`. However, all static-ground cases fell through, including ordinary box-ground with no special channel filter. This invalidated the fixture's static instanced support, not the dynamic-pair measurements.

The revised fixture uses ordinary noninstanced static colliders, as the actual terrain does, and authors their filters on the actual collider instead of an untyped parent Xform. Its 39 robot/dynamic-support collision instances still share one prototype in a standalone USD check. A second device execution remains required; no training default was changed.


The revised physical fixture completed with exit 0 (`deepwbc-material-pairs-v2-20260912`): all 14 pairs passed. The subsequent isolation attempt v3 failed at an unavailable legacy `isaacsim.core.cloner` import before motion; that failure is retained. V4 uses the same current `isaaclab.cloner.filter_collisions` call as `InteractiveScene`, and also completed with all 14 cases passing. Two environments begin at the identical pose `[0,0,0.055]`; their distinct raw friction values `-0.5` and `-0.25` retain measured coefficients `0.250001` and `0.375004`, respectively. This establishes the bounded shared-collider pair/filter and inverted-group isolation probe, not whole-robot learning.

`pawcerto/isaac/deepwbc_material_channels.py` now supplies the production pair representation. Before physics parsing it adds only collision geometry, retaining source shared mesh references and copying the originally noninstanced primitives. It preserves existing body exclusions, mass, COM and inertia. No public tensor API shape-path ordering was available, so the startup coefficients representing `f=0` (`external=0.5`, `self=0`) identify the actual native shape masks. The runtime checks both role counts, restores source default `f=1` before the first task physics step, and then applies the caller's original samples through the supported material setter. The two regression tests cover permuted native shape ordering with negative/large positive samples, and inherited mixed-geometry clones with unchanged mass and shared prototypes. Both pass. The inherited-clone test initially caught a redundant reference override; already-inherited self instances are now reused.

A CPU audit of the actual converted asset with 16 references (`outputs/deepwbc-material-channels/cpu-asset-audit.json`) finds 50 collision shapes per environment (22 mesh, 8 sphere, 20 cube), just 10 shared mesh prototypes across all environments, and unchanged mass/COM/principal-inertia attributes. These are composed-USD observations, not native cooking measurements.

The first reconstructed-sensor production attempt still failed before any update (`outputs/deepwbc-reconstructed-execution-20260912/train-attempt1.log`). Its **first causal error** is `PxgGeometryManager` attempting to allocate **579,166,759,168 bytes** of pinned host geometry-upload memory. PhysX then declares the scene corrupted. The later material readback mismatch and shutdown exit 11 occur after that corruption and are not treated as the initial material failure. Terrain and robot geometry owners are investigating the geometry upload; simply raising the process memory cap does not explain this request. The 21-update outcome remains pending.


Subsequent device runs separated the factors. Full terrain-only reset plus one step passed. The paired real robot on flat ground passed with both one and 16 environments. Actual readback gives 50 shapes and total mass `14.15087890625 kg` per environment; one-robot `static.npz` comparisons against the earlier single-channel run give zero difference for every body's mass, COM and inertia (`outputs/isaac/deepwbc-paired-robot-flat-20260912/inertia-comparison.json`).

One robot plus the complete terrain nevertheless reproduced a **36,875,815,168-byte** pinned upload request. Combined with the 16-environment failure, requests exactly follow `723,085,568 + 50*N*723,054,592` bytes. The local official PhysX source shows geometry registration accumulating pending upload bytes while removal of a pending geometry does not subtract those bytes. This supports repeated geometry registration during individual relationship edits as the mechanism, rather than the raw terrain or robot shape size alone.

The adapter now collects all filter targets and writes each final relationship once inside `Sdf.ChangeBlock`. A CPU comparison against the preserved failed-run implementation verifies identical final composed targets: **467 relationships / 18,800 targets** for 16 actual robot USD references and 19 stand-in terrain colliders (`outputs/deepwbc-material-channels/filter-batch-equivalence.json`). The same physical one-robot/full-terrain probe is being repeated to determine whether batching resolves the upload failure; no source geometry or collision-pair semantics were changed.


The batched physical repeat still requests exactly `36,875,815,168` bytes and fails (`combined-batched.log`). Therefore batching relationship authoring is **not a sufficient explanation or fix**; the same final terrain-to-shape relationship expansion remains a candidate. A diagnostic-only copy (`combined-no-terrain-pairs.py`) omits terrain-to-self filtering while retaining all geometry and other runtime behavior. Its result must only discriminate the failure cause: omitting that filter is not an acceptable implementation of the original contact semantics.


The `terrain=[]` diagnostic completed successfully, but review identified a second omitted USD operation: it also skipped rebinding terrain to the ordinary material with the same existing coefficient 1. Consequently that contrast isolates the terrain-related authoring collectively, not pair filters alone. A smaller output-only diagnostic (`combined-no-terrain-bind.py`, using `diagnostic-no-terrain-bind.py`) now omits only the repeated terrain material binding while retaining every terrain-to-self filter. Its source manifest records the single removed line; production code remains unchanged while this distinction is tested.


The strict single-operation `no-terrain-bind` diagnostic still failed with the same 36.8758 GB first request. Removing redundant binding alone therefore does not repair the issue. The accepted implementation replaces this DeepWBC scene's previous environment grouping with **3N+1 single-membership inverted collision groups**: External/Self/Box per environment and global Terrain. It allows External–Terrain, External–same-environment Box, Self–same-environment Self, and Box–Terrain. Cross-environment contact stays disabled and existing body exclusions are retained. The old group scope is inactive, avoiding ambiguous membership in overlapping group systems. Large terrain meshes no longer carry per-shape self-exclusion relationships. Original terrain material binding remains present.

CPU validation of 16 actual robot USD references plus 19 stand-in terrain colliders checks **49 groups, 835 single-membership shapes, all group-pair decisions, 10 shared mesh prototypes**, and inactive old groups (`outputs/deepwbc-material-channels/group-asset-audit.json`). The standard USD table is explicitly interpreted with the current PhysX scene-wide inversion flag; that logical check is not itself a physical test.

The production group helper then passed all **14 physical material/isolation cases** (`outputs/isaac/deepwbc-material-groups-20260913/analysis.json`), including coincident environments with distinct negative raw friction. The normal one-robot/full-original-terrain composition also completed with exit 0, one reconstructed-wrench control step, **4 groups / 70 single-membership shapes**, 50 native robot shapes, mass `14.15087890625 kg`, and maximum RSS `6,194,008 KiB` (`outputs/deepwbc-reconstructed-execution-20260912/combined-one-groups/result.json`). No diagnostic omission remains. The device owner has returned to the original 16-environment, 21-update attempt; successful initialization and these probes do not establish learning.


The original **16-environment / 21-update / seed-1** run subsequently completed with **exit 0**. Its actual log records DAgger at iterations 0 and 20 and PPO at iterations 1–19. The checkpoint is `outputs/isaac/deepwbc-reconstructed-21updates-20260912/model_21.pt`; adjacent `evidence.json` records `iterations_completed: 21` and explicitly does not infer learning success. Full original terrain, raw friction sampling, production groups, physical randomization and reconstructed-wrench stepping were exercised. Policy export and the fixed 500-step evaluation are the remaining bounded checks.


### Physical qualification withdrawn after replay audit

The first exported fixed 500-step trajectory is finite and matches its exported policy, but `outputs/isaac/deepwbc-reconstructed-model21-fixed500-seed2027-20260912/trajectory-audit.json` finds **zero contact forces for every body at every step**, 33 base-height terminations, and episodes lasting only 14–16 steps. These observations invalidate physical task qualification for this run and the preceding 21-update checkpoint; computational completion remains the only established integrated result. Terrain spawning is under repair for a suspected current-Lab xyzw versus USD real-first quaternion conversion mismatch. The failed checkpoint and trajectory are preserved. A fresh original 16-env/21-update run, export, and fixed 500-step evaluation are required after the actual transform correction. The independent material-pair evidence remains separate from this failed terrain-placement integration.


The terrain owner confirmed and repaired the transform: current Lab supplies quaternion **xyzw**, while the old spawner passed it to `Gf.Quatf` as real-first. The corrected constructor uses scalar `quat[3]` and vector `quat[:3]`. A source-backed regression failed before the repair and all five terrain tests pass afterward. The full terrain world-coordinate audit (`outputs/deepwbc-terrain/quaternion-repair-20260913/world-result.json`) places every prior evaluation XY inside the corrected bounds and outside the old bounds. Original vertices, triangles and source heightfield overflow behavior are unchanged. Fresh device execution is underway; this CPU transform evidence does not yet establish restored physical contacts.


### Final corrected-terrain integration

The repaired original **16-env / seed-1 / 21-update** run completed with exit 0 at `outputs/isaac/deepwbc-reconstructed-21updates-terrainfixed-20260913`, exercising DAgger at iterations 0 and 20 and PPO at the other 19 iterations. Its model-21 export completed into `outputs/deepwbc-reconstructed-execution-20260912/model21-terrainfixed-export`. The unchanged seed-2027 fixed 500-step evaluation completed at `outputs/isaac/deepwbc-terrainfixed-model21-fixed500-seed2027-20260913`. All 500 samples are inside the corrected terrain and have body contact above 1 N; root height is .316588–.417994 m and median summed vertical contact force is 149.973 N. Checkpoint, fused export, split export and recorded GPU actions agree exactly on all measured observations. This closes the assigned material/runtime integration and physical execution path without further contact repairs.

The task result remains limited: EE RMSE .47936082 m, forward-velocity MAE .49878025 m/s, and one signed-roll termination; the final 455 samples are an unfinished episode. These are valid bounded negative performance results, not learned locomotion, EE tracking, exact sensor parity or paper reproduction. All previous failures and invalid zero-contact runs remain preserved. Full commands, audits and qualifications are recorded in [the final integration report](experiments/deepwbc-reconstructed-runtime-20260912.md#final-corrected-terrain-execution-2026-09-13).
