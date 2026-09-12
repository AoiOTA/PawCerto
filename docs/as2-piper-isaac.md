# AS2 EDU + Piper-H nominal Isaac Lab asset

This converts the pinned nominal research model documented in
[as2-piper-assets.md](as2-piper-assets.md) with official, unmodified Isaac Lab
and PhysX. It does not require a physics-engine rebuild, import PawWeaver
policies or hardware estimates, or connect hardware.

## Reproduce

Run from the source checkout after [fetching and building the pinned AS2
inputs](as2-piper-assets.md#reproduce-on-cpu). The Python wheel does not include
`scripts/`, vendor inputs or generated USD. Use `python` from the existing
official [Lab environment](isaaclab-install.md) for USD operations; CPU
preparation also works in the documented CPU environment. Generated assets
and the local experiment records are not included in this source delivery.

CPU preparation (no AppLauncher or GPU):

```bash
python scripts/convert_as2_piper_usd.py --prepare-only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_as2_piper_isaac_asset.py
```

Actual conversion requires the installed official Isaac Lab runtime. Schedule
this with the current GPU operator; do not launch a second competing service:

```bash
python scripts/convert_as2_piper_usd.py --visualizer none
```

Current Lab checkout is `third_party/IsaacLab-develop-sim610`. `--output`
selects an output directory; default is `reference/isaac/as2_piper`.
The output contains `source.urdf`, `merged.urdf`, `preparation.json`,
the actual USD path named by `usd_path.txt` (current importer may produce
`merged/merged.usda` despite the requested `robot.usd`) with shared layers,
`conversion-validation.json` and `usd_path.txt`. Copy the entire USD directory
when relocating the generated asset; reprepare URDFs to resolve mesh paths
on a different checkout.

To update an already converted asset without AppLauncher or GPU, with its
`usd_path.txt` pointing at that asset and all shared layers inside `--output`:

```bash
python scripts/convert_as2_piper_usd.py --prepare-collision-offsets-only \
  --output reference/isaac/as2_piper --contact-offset 0.01 --rest-offset 0
```

This writes generated USD layers and `collision-offset-validation.json`;
it is not a read-only check. Do not apply it to assets held by a running
simulator. Use a separate copy for validation. See the
[AS2 learning protocol](as2-umi-learning-plan.md) for the preserved OOM,
shared-geometry control comparison and formal model0 baseline.

## Source and consumer contract

`pawcerto.robots.as2_piper.prepare_urdf(output)` reads the canonical config
and verifies vendor source hashes through the existing independent assembler.
The source mount is `(0, 0, 0.12)` m and the original stock gripper is fixed at
40 mm opening. Hardware mass, mount, actuator dynamics and TCP calibration
remain unverified.

- Floating physical root body: `base_link`.
- Four retained foot bodies: `FL_foot`, `FR_foot`, `RL_foot`, `RR_foot`.
- TCP: `piper_gripper_base` plus `(0, 0, 0.138)` m with identity local rotation.
- Controlled names and order: `configs/as2_piper.json:controlled_joint_order`,
  12 leg joints followed by `piper_joint1` through `piper_joint6`.
- 28 positive-mass bodies, 18 revolute joints, source total mass 22.347 kg.
- All positive-mass fixed links remain separate, including foot links, arm
  mounting base, flange, gripper housing and both fixed fingers.

`merged.urdf` removes only the empty `world` floating-base placeholder and
empty fixed leaf `piper_gripper_link`. It does not merge positive-mass links,
change collision transforms, aggregate inertia, or move the physical TCP.
The virtual stroke frame is restored in USD as a kinematic Xform under
`piper_gripper_base`, at its source `(0, 0, 0.04)` m; the `tcp` sibling Xform
uses the nominal TCP offset. Neither carries MassAPI or RigidBodyAPI.

Read actual `default_prim`, `articulation_root_path`, `body_paths`, `joints`
and `tcp_path` from `conversion-validation.json`. Do not assume importer
joint traversal order equals the configured action order. Consumers should
resolve names, and use the gripper-body pose plus the TCP offset for dynamic
state instead of treating the TCP Xform as another rigid body.

## Physics provenance

Source primitive collision shapes remain unchanged (cylinders are not
silently changed into capsules). Source collision meshes use the official
importer's convex-hull approximation, which does not preserve concave
contact geometry. Visuals are copies of source collision geometry. The
layered USD uses shared instanceable geometry; the converter never disables
instancing or makes clones unique.

Mass, COM, full rotated inertia tensors, joint frames, axes and source
position/effort/velocity bounds come from the selected URDFs. The current
importer emitted velocity only as `newton:velocityLimit` and defaulted to
its generic `physics` variant. The converter explicitly selects `physx`,
authors `physxJoint:maxJointVelocity` from source rad/s converted to deg/s
(as documented by the installed official PhysxSchema), and explicitly
authors the declared self-collision choice. All 18 drive effort bounds and
zero drive gains are verified against the source/direct-torque contract. COM and inertia
are explicitly authored in USD principal-moment form: this avoids an
unauthored COM becoming PhysX's automatic `-inf` sentinel and avoids known
importer principal-axis representation errors. These are source-preserving
corrections, not estimated robot parameters.

Joint drive target is `none` (zero importer stiffness and damping) for direct
torque control. No added armature, identified motor gains or measured friction
is supplied. Self collision is explicitly enabled. The delivered collision
geometry declares contact offset 0.01 m and rest offset 0 m, matching the UMI
runtime configuration. These attributes are authored in the shared geometry
layers, including all 11 mesh instance proxies, so the runtime can retain
instancing when cloning 4096 environments. Materials and other settings retain
the importer defaults until declared runtime overrides apply. A standalone
`--prepare-collision-offsets-only` converter option applies and validates these
attributes without launching a simulator; a normal conversion also authors them. Source effort and velocity limits recorded in the report
are a consumer contract, not hardware certification or proof that every
runtime controller enforces them.

## Source-delivery verification (2026-09-13)

The source candidate was checked in an isolated checkout snapshot containing
only the four assigned AS2 runtime/converter/document changes. Existing CPU
dependencies were reused; no simulator, training or dependency installation
was started for this delivery.

- Selected source-preparation, robot-binding and solver-force CPU tests:
  **10 passed, 13 subtests passed**. The broader upstream policy-reference
  test could not collect because the CPU environment lacks `pytorch3d`;
  the Lab interpreter lacks `pytest`. These are not counted as passes.
- The public `--prepare-only` and AS2 configuration-preparation commands
  completed. The public `--prepare-collision-offsets-only` command repaired
  deliberately different offsets on a disposable copy of the generated USD:
  **42 collisions / 11 instance-proxy collisions**, with all compared
  non-offset composed attributes, relations, types and instancing unchanged.
  Standalone USD source inertia/FK/joint validation passed without AppLauncher.
- A wheel was built offline with `pip wheel --no-deps --no-build-isolation
  --no-index --no-cache-dir`. Its **82 package files** matched the isolated
  source bytes. It contains Python modules and licenses, not the checkout
  scripts, assets, experiment outputs or engine patches.
- A process outside the repository imported the extracted wheel's UMI
  modules, loaded the actual 4096-environment `model_0.pt`, exported it and
  consumed that export. Checkpoint/export actions were exactly equal on
  **64 deterministic 132-dimensional CPU inputs**. A separate Torch/NumPy-only
  process, importing neither PawCerto nor a simulator, consumed the same
  TorchScript actor with zero action difference. This is serialization and
  package-consumer evidence, not a new closed-loop rollout or fresh-install
  validation; the wheel was extracted for import, not installed.
- The formal model0 saved fixed16 arrays were independently read back:
  **16 complete cases, 849 saved endpoints each**, finite numeric arrays,
  exact frozen targets, and position/orientation means matching the published
  baseline. The target-packing command in the linked learning protocol also
  ran on copies of those inputs. No active experiment files were rewritten.

Local validation commands, scripts, logs and consumer result JSONs are retained
under `/tmp/pawcerto-as2-delivery-20260913/`; they are not shipped artifacts.
The recorded physical startup/control results remain attributed to the GPU
operator. Model500/4000 performance, final Lab comparison and final learned
policy export are outside this source-delivery result and remain pending.

## Evidence boundary

CPU preparation checks source inertials, collisions and every retained joint
byte-for-byte, then compares all retained body frames and TCP by independent
URDF FK at two poses. The actual conversion audit checks all rigid-body names
and masses, corrected COM/full inertia tensors, zero-pose body FK, all joint
endpoints/axes/anchors/position limits, fixed-joint constraints, and collision
count with instance proxies. All 31 primitive collision centers, orientations
and dimensions are checked using shape matrices; the 11 source meshes remain
convex-hull approximations.
The optional USD pytest uses a session layer and does not mutate the delivered
asset. With the installed environments (Lab has USD but no pytest; the CPU
MuJoCo environment has pytest but no USD), run the standalone read-only audit:

```bash
python scripts/convert_as2_piper_usd.py --validate-only
```

This opens USD using `pxr` only and never starts AppLauncher.

Observed 2026-09-12 results:

- CPU pytest: 1 passed, 1 skipped. The already-installed MuJoCo Python
  environment has pytest but no `pxr`; the standalone audit below exercises
  the actual USD with Lab Python instead. No package was installed.
- Official conversion completed with exit 0 in the GPU operator's bounded
  service. Log: `outputs/as2-piper-lab-execution-20260912/convert.log`.
- Standalone USD audit completed with exit 0 after the source-preserving
  PhysX velocity-field/variant correction. Report:
  `outputs/as2-piper-lab-execution-20260912/standalone-usd-validation.json`.
- Actual asset: `reference/isaac/as2_piper/merged/merged.usda`.
  Default prim `/as2_piper_nominal`; articulation/root body
  `/as2_piper_nominal/Geometry/base_link`.
- 28 physical bodies, 18 revolute joints, 9 retained fixed joints,
  22.347 kg total mass; maximum source inertia-tensor entry error
  `6.85e-9 kg m²` on independent reload.
- 42 collisions: 5 spheres, 17 cylinders, 9 boxes and 11 meshes. All 11
  mesh collisions remain instance proxies; primitive shapes also retain
  source dimensions and transforms.
- Nominal TCP path:
  `/as2_piper_nominal/Geometry/base_link/piper_base_link/piper_link1/piper_link2/piper_link3/piper_link4/piper_link5/piper_link6/piper_flange_link/piper_gripper_base/tcp`.

These are actual conversion and static USD evidence.
No asset check establishes learned WBC, UMI transfer success, sustained
contact stability, force-tracking accuracy or readiness for physical robots.
