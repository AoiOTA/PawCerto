# AS2 + Piper-H nominal vendor asset

This independent CPU asset prepares later PawCerto robot transfer. It uses the
vendor **AS2 URDF, Piper-H URDF and stock-gripper Xacro/meshes**, with a declared
nominal rigid mount. It does not import PawWeaver code, diagnostic actuator
settings, policies, or physics overrides. No method registry or training
integration is added.

The default includes the stock gripper's vendor flange, housing and two fingers,
fixed at **40 mm opening** to retain the 18-joint WBC action interface. The
vendor prismatic/mimic transforms are evaluated at that opening; no gripper
actuation or grasp controller is claimed. Camera, mounting plate and cables
remain excluded. AS2-to-arm mount and the grasp TCP remain nominal, not measured
physical calibration. The bare-arm control is retained in
`configs/as2_piper_bare.json` and `reference/as2_piper/bare/`.

## Reproduce on CPU

From the PawCerto checkout, a separate CPU environment can be prepared without
installing a training runtime:

```bash
python3 -m venv .venvs/as2-piper-cpu
source .venvs/as2-piper-cpu/bin/activate
python -m pip install numpy scipy "mujoco>=3.8" pytest
```

Then fetch, build and check:

```bash
python scripts/fetch_as2_piper.py
python scripts/build_as2_piper.py
python scripts/build_as2_piper.py --config configs/as2_piper_bare.json --output reference/as2_piper/bare
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_as2_piper_asset.py
```

An optional `--cache /path/to/existing/upstream` reuses local source files only
when each matches the blob SHA in the official pinned Git tree. It does not
trust another project's manifest. Fetches preserve vendor license files and
record repository/revision, Git blob SHA-1, SHA-256, size and source URL in
`third_party/as2_piper/provenance.json`. Existing changed files fail explicitly.
The build rechecks source hashes. No source code or mesh is redistributed in
the tracked changes; `third_party/` and `reference/` are already gitignored.

`--sources`, `--config` and `--output` select alternate input/output paths for
the build. Default outputs are `reference/as2_piper/robot.urdf`, `robot.xml`,
and `validation.json`. Generated XML contains absolute mesh paths; regenerate
on the receiving checkout rather than copying it alone. Loading requires only
MuJoCo, with no Isaac, PhysX patch, ROS, GPU or hardware connection.

The local verification used `/home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python`,
MuJoCo 3.13.0. The first plain pytest invocation failed while importing the
ambient ROS `launch_testing` plugin because `yaml` was missing. Disabling
unrelated automatic pytest plugins as above ran the asset tests; no ROS
installation or environment mutation was needed.

## Source model selection

| Component | Official fixed source | Use |
| --- | --- | --- |
| AS2 | [unitree_ros 7d6075f](https://github.com/unitreerobotics/unitree_ros/tree/7d6075f7f58588b189b940130e3edab3c839b2df/robots/as2_description) | URDF inertials, collisions, joints and limits |
| Piper-H | [agx_arm_urdf f6642ce](https://github.com/agilexrobotics/agx_arm_urdf/tree/f6642ce0d7872c686f29c99e9e10cd23d1d49313/piper_h) | Six-joint URDF, stock-gripper Xacro and collision STL meshes |
| Alternate AS2 | [unitree_mujoco 1eb6642](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/as2/as2.xml) | Mass discrepancy record only; not mixed into selected URDF |

The selected AS2 URDF sums to **17.64 kg**, the bare Piper-H to **4.167 kg**,
plus **0.54 kg** of stock gripper and flange, for **22.347 kg** combined.
The retained bare-arm control totals **21.807 kg**. The alternate official AS2 MJCF sums to **20.7 kg**.
These source models disagree; no guessed battery, ballast or proportional
inertia correction reconciles them. The physical robot's actual mass and
inertia remain unverified. This source-based research model can support
reversible simulation, but it cannot establish hardware equivalence.

All vendor inertial origins/tensors and arm/leg joint transforms are retained.
Stock-gripper prismatic/mimic joints become fixed joints at the configured
opening; their original limits and mimic expressions are retained in the
validation report. The pinned Xacro only includes the bare arm and plain links
and joints, so no ROS/Xacro runtime is needed to compose those source elements. Piper
names receive a `piper_` prefix to avoid collisions. Rendering-only visuals
are omitted; source collision geometry also supplies the visible robot shape.
MuJoCo represents each mesh collision by its convex hull; contact surfaces
are therefore not a fidelity claim for arbitrary detailed mesh concavities.
The converter retains fixed bodies (`fusestatic=false`) and uses source
inertia (`inertiafromgeom=false`), adds a ground plane, 18 direct torque motors,
a TCP site, and a `default` keyframe. The simulation timestep is 5 ms (200 Hz);
this is not a verified hardware or policy control frequency. MuJoCo's unmodified default contact
parameters apply; no contact or actuator system identification was performed.

## Explicit research configuration

`configs/as2_piper.json` owns the mount, TCP and initial state:

- The rigid massless mount places `piper_base_link` at `(0, 0, 0.12)` m in AS2
  `base_link`, with zero roll/pitch/yaw. This is a nominal research choice, not
  measured mechanical assembly or the prior project's mount height.
- `tcp` is the nominal midpoint of the opposing vendor finger reference
  origins: `piper_gripper_base` plus `(0, 0, 0.138)` m, with the gripper-base
  orientation. This is a source-frame reference, not a calibrated contact
  center or a verified SDK TCP. The bare control retains `piper_link6` TCP.
- Stock-gripper source masses are flange 0.04 kg, housing 0.45 kg and
  fingers 0.025 kg each. Source 100 mm stroke maps through mimic multipliers
  +0.5 and -0.5 to the two fingers. At the selected 40 mm stroke these joints
  are fixed at +0.02 and -0.02 m respectively, after applying source joint
  frame rotations. The virtual massless gripper joint is also fixed; it adds
  no fabricated inertia. This is a fixed-open WBC asset, not a full grasp model.
- Base initial pose is `(0, 0, 0.55)` m and identity quaternion `(w,x,y,z)`.
  Each leg is initialized at `(0, 0.8, -1.5)` rad; the arm at
  `(0, 1, -1.2, 0, 0.2, 0)` rad. This is a collision-free elevated initial pose,
  not an equilibrium or learned standing pose.
- Motors accept joint torque in N·m, with unit gear and the source URDF effort
  bounds. Those source values are not measured current/torque limits. No gains,
  motor dynamics, added armature, friction compensation or diagnostic policy
  is introduced. Source velocity bounds are reported but not enforced by
  these direct MuJoCo torque motors. Published/manual limits can differ from
  URDF limits; this asset preserves the selected source rather than merging
  them silently.

The explicit controlled order (independent of consumers' existing robot order)
is:

```text
FL_hip_joint, FL_thigh_joint, FL_calf_joint,
FR_hip_joint, FR_thigh_joint, FR_calf_joint,
RL_hip_joint, RL_thigh_joint, RL_calf_joint,
RR_hip_joint, RR_thigh_joint, RR_calf_joint,
piper_joint1, piper_joint2, piper_joint3,
piper_joint4, piper_joint5, piper_joint6
```

Current compiled model: `nq=25`, `nv=24`, `nu=18`; joint qpos indices are
7–24, joint velocity indices 6–23. `validation.json` records actual named
mapping and source lower/upper/effort/velocity limits. Consumers should use
names or that explicit mapping, not assume another robot's ordering. Reset
with `mujoco.mj_resetDataKeyframe(model, data, 0)` to select the default pose;
plain `MjData(model)` alone does not select that keyframe.

## Observed validation

The 2026-09-12 CPU build verified 36 pinned source files and passed:

- Per-body mass, inertial center and full rotated inertia-tensor comparisons
  against source URDF; maximum tensor entry error `3.32e-7 kg m²` after MJCF
  serialization; total mass preserved.
- Every controlled joint's limits, actuator target/order and effort bounds,
  with all default joint positions inside the source limits.
- Independent URDF forward kinematics against MuJoCo for all bodies and TCP
  at the default and two interior-limit poses. Maximum position error
  `2.58e-7 m`; maximum rotation-matrix entry error `5.45e-7`.
- Default forward dynamics finite, generalized inertia positive definite
  (minimum eigenvalue reported in `validation.json` in mixed generalized units),
  no default contacts, TCP world position approximately `(0.356816, 0, 1.165838)` m.
- Direct torque inputs map with unit gain to the named generalized joint
  forces and saturate at source effort bounds. The 50 ms unactuated no-contact
  probe gives whole-robot COM velocity `(0, 0, -0.4905)` m/s, matching gravity.
- Five tests passed: actual build/reload plus ten unactuated CPU integration
  steps (finite state, no MuJoCo warnings), deliberate wrong-axis detection
  by independent FK, rejection of modified source bytes, frozen-gripper FK
  against the original vendor prismatic/mimic chain, and retained bare-arm
  control build.

These are asset and brief numerical execution evidence. There is no learned
policy, WBC acceptance, balance result, sustained contact test, sim-to-sim
policy transfer, real force tracking, or hardware certification here. The ten
steps are 50 ms of unactuated dynamics, not a stability evaluation.

## Hardware force feedback: evidence and unresolved options

A bounded read-only official-source check on 2026-09-12 does **not establish
that the user's purchased AS2 EDU + Piper-H + stock gripper includes an
independent wrist six-axis force/torque sensor**. Plan provisionally without
one unless the purchased option or physical unit establishes otherwise.
Absence of a URDF sensor tag, an unverified metadata flag, or a missing product
specification is not proof that a physical sensor is absent.

The official [AS2 specification](https://www.unitree.com/As2/) lists dual joint
encoders and variant-dependent capabilities. It does not resolve, in the
material checked, AS2's foot-force sensing implementation or whether a
particular purchased option has dedicated force/torque sensing. A generic
Unitree `foot_force`/`tau_est` field from another robot is insufficient to claim
AS2-specific measurement hardware, validity or units. AS2 foot-force signal
origin, hardware availability and interface behavior remain unverified here.

For the arm, the official
[Piper SDK high-speed feedback implementation at c9e8a28](https://github.com/agilexrobotics/piper_sdk/blob/c9e8a28174e71eeaac448593cb65f8ab258a92fe/piper_sdk/piper_msgs/msg_v2/feedback/arm_feedback_high_spd.py)
exposes motor current (CAN `0x251`–`0x256`, raw unit `0.001 A`) and calculates
`effort` by multiplying current by fixed coefficients (1.18125 for joints
1–3, 0.95844 for joints 4–6). This specific effort field is a current-derived
torque estimate, not evidence of an independent joint torque sensor. The
code calls its torque unit `0.001 N/m` (a dimensional notation inconsistency
with torque); calibration and physical meaning should be checked for the
actual Piper-H firmware before treating it as accurate load torque.

For the stock gripper, the official
[Piper SDK gripper feedback at c9e8a28](https://github.com/agilexrobotics/piper_sdk/blob/c9e8a28174e71eeaac448593cb65f8ab258a92fe/piper_sdk/piper_msgs/msg_v2/feedback/arm_feedback_gripper.py)
defines CAN `0x2A8`, stroke `grippers_angle` in `0.001 mm` and
`grippers_effort` as gripper torque in `0.001 N·m` in its English documentation.
It does not establish whether that gripper torque comes from an independent
transducer or an internal estimate. The FOC `sensor_status` flag is also not
evidence of a six-axis force sensor. This scalar gripper quantity cannot be
substituted for a 3D endpoint external force or six-dimensional wrist wrench.

There is an SDK-version distinction: the newer official
[pyAgxArm AGX gripper driver at e7aef17](https://github.com/agilexrobotics/pyAgxArm/blob/e7aef17d54cac80cbaeb1b4110ab3d8f1337a95b/pyAgxArm/protocols/can_protocol/drivers/effector/agx_gripper/default/driver.py)
defines `get_gripper_status().force` in N. Do not transfer scaling or physical
interpretation across those APIs without matching actual firmware and gripper
version. No live telemetry, current-to-torque calibration, bandwidth, bias,
contact-force accuracy or hardware configuration was measured in this work.

The original UniFP / Learning Force Control methods' ability to operate
without external force sensors is a separate method-level question. Lack of
an established wrist sensor is not a reason to stop CPU asset simulation or
conclude those methods cannot be applied. Their reproduction and the real
AS2/Piper-H assembly's actuator/interface identification must be evaluated
separately; the asset checks above establish neither.
