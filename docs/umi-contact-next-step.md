# UMI contact diagnosis: findings and current correction

The original bounded contact and warmup diagnosis remains valid: seed1 saturates and loses support before its MuJoCo head impact, and replacing only the Lab warmup physical state does not remove that collision. **The initialization-only direction is closed.** No controller delay/PD defect was found.

The subsequently authorized nonzero-state comparison identifies a new actionable issue: the current imported USD's principal inertia axes disagree with the source URDF. At three frozen states before the head event, nonzero-torque free responses differ materially even when both engines report no contact. Native Lab responses agree with its own measured mass/bias matrices; the matrix mismatch is explained by the imported inertia orientation. A source-tensor correction is implemented for an isolated new USD and has passed both CPU reopen checks and actual native inertia/mass-matrix/single-step validation. The excited free-response discrepancy drops below4.11e-5 rad/s across all three points and their repeats. Contact response differences remain; this is not yet a full-policy behavioral fix or release acceptance.

Original defaults, weights, source assets and all previous recordings remain unchanged. No reward tuning, parameter sweep or initialization change follows from these results. The subsequently authorized corrected-asset seed0 run completed4000 updates on the fixed71/15/15 split. Its final test contains a trajectory1 fall/head-ground event followed by BADQACC at10.525 s; the candidate does not pass full-policy acceptance. Old full-pool policies retain their original identities and do not become held-out policies retroactively. The [full result](umi-source-inertia-result.md) and [machine evidence](../outputs/umi-source-inertia-train71-20260912/result.json) separate component fidelity, training completion and policy failure.

## Historical starting evidence: existing 20 ms records

Two independent read-only investigations examined saved events and the original method's active consumers. The fixed target positions and rotations match Official at the compared times.

- **Seed 1, case 6:** in the saved 5.84–6.24 s window, support is already abnormal before the 6.04 s head contact. At 6.00 s no foot exceeds the existing ground-support threshold and FR calf velocity reaches +66.58 rad/s; Official's largest joint speed at that clock is 0.408 rad/s. At 6.02 s the recorded raw FR calf action is -13.717. At 6.04 s the head-contact norm reaches 779.108 N, body tilt is 24.98 degrees and EE error is 38.0 mm, versus Official's 6.08 degrees and 9.0 mm. The base is slightly higher than Official, so low base height alone does not explain the event. Raw action is not the delayed, clipped executed joint target.
- **Seed 2, case 6:** at 7.96 s all four feet exceed the support threshold, but front/rear positive foot loads are about 22.41/132.31 N, versus Official's 139.28/61.53 N. RR calf also contacts the ground, with 179.66 N force norm and 145.96 N upward force. This is a different pattern from seed 1's large joint-speed excursion. Counting supporting feet alone misses the load distribution and nonfoot contact.

Sources are the [seed-1 saved case](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed1_seed2027/model_4000/case_06.json), [seed-2 saved case](../outputs/mujoco/umi_reconstructed_solver_relaxed_body_speed_scratch_seed2_seed2027/model_4000/case_06.json), their adjacent NPZ arrays, and the [Official paired case](../outputs/mujoco/version-3.13-regression/official_ours/case_06.json). The analyzed zero-based row windows are 290–310 and 386–406 respectively. Existing 20 ms sampling cannot establish the order of every 5 ms torque/contact event or exclude an earlier transient contact.

## The method boundary

The active current and original paths both apply sparse rewards and termination after four 5 ms physical steps. Head/base/hip/thigh/link1–5 contact above the configured threshold triggers termination; calf contact is penalized but is not in that termination set. Collision cost counts affected links, rather than scaling with the magnitude of a large impact. EMD normalizes the four foot forces; it does not directly measure their fraction of all body-ground load. Thus every nonfoot contact is not automatically the same kind of original-method violation, and a large nonfoot load share does not imply a proportionate EMD penalty.

The comparison found no omitted active original constraint that currently explains the head event. Named residual config entries without constructors are also inactive upstream. The current collision implementation omits multiplying its threshold by `violation_force_norm`, but that value is 1 in all three runs, so it does not affect their results. Relevant consumers: [current semantics](../pawcerto/methods/umi_on_legs/training/semantics.py), [current training step](../pawcerto/methods/umi_on_legs/training/isaac_env.py), [original constraints](../third_party/umi-on-legs/mani-centric-wbc/legged_gym/env/isaacgym/constraints.py), [original step/termination](../third_party/umi-on-legs/mani-centric-wbc/legged_gym/env/isaacgym/env.py).

## Historical proposal that led to the completed diagnosis

Freeze seed 1's final policy and retain the original case-6 target, initial condition, controller, delay, physics settings and time horizon. Read out actual executed PD torque, joint position/velocity and contact at 5 ms resolution around the earliest visible change near 5.98–6.00 s, through the head event. Mark the existing 20 ms action/reward/termination boundaries. Preserve the already exercised nominal setup rather than silently changing its initialization or batching. Use the new Conda environments.

Start with the missing MuJoCo execution/contact ordering. Reuse the [existing Lab nominal16](../outputs/isaac/umi-relaxed-body-speed-scratch-seed1-nominal16-seed2027/rollout.json) for its saved EE, root and foot-support fields. If the next decision requires head-contact timing in Lab, supplement only that missing signal under the same nominal setup: its existing ten-field report cannot establish absence of head contact. This is a fixed-policy diagnostic, not a new training seed or another demonstration that the simulator starts.

The result should decide whether the first actionable difference is an execution/physics mismatch, a transient contact missed by the original sampling window, or behavior already present in Lab. Do not infer one of those causes from current 20 ms records alone. Only then choose a single implementation or training change. Compare its full fixed-case tracking, orientation, nonfoot contact, inversion and early-termination results together; retaining a low mean EE error is not enough. No extra seeds, reward additions, relaxed thresholds, synthetic target simplification or public release are part of this proposal.


## Executed 2026-09-12: 5 ms diagnosis

The three MuJoCo conditions (seed1 final4000, Official, seed2 final4000) each ran the original case 6 from **one shared sample(16, 2027)** for the complete 17 seconds, twice with tracing off/on. Each trace has 3400 physical steps, including the original four-step zero-action warmup. All seven NPZ arrays and the full result JSON are **exactly equal** between tracing off/on for each condition. No extra substep `mj_forward` was introduced. Existing endpoint results remain unchanged.

`Go2Arx5Mujoco.step(..., trace=callback)` and `evaluate(..., substep_trace=callback)` default to no tracing. The new [trace runner](../scripts/trace_umi_contacts.py) saves raw/clipped/previous/executed actions, target, requested/applied torque, pre/post generalized position and velocity, existing contact solve, root/EE state and target clock. Callback/output failures propagate. It refuses to reuse an existing output directory.

The actual integrator is **mjINT_IMPLICITFAST**. All times below are `time_before_s`, the start of the step whose existing solve is read after `mj_step`; `time_after_s` is 5 ms later. Contact records internally carry the latter engine time. Cartesian body/contact fields belong to the existing solve, while `qpos_after/qvel_after` have already been integrated. The original 20 ms endpoint `mj_forward` is a **separate recomputed solution**, not the force that advanced the preceding step.

### Seed1: saturation and lost support precede head contact

| Solve start → end | Actual observation |
|---|---|
| 5.980 → 5.985 s | Executed FR calf action +3.962796926; requested PD 55.663894 Nm, applied 44.4 Nm. Maximum post-step joint speed 21.59 rad/s; two feet carry >1 N. |
| 5.995 → 6.000 s | No foot carries >1 N; maximum post-step joint speed 66.58 rad/s. No active nonfoot contact in this solve. |
| 6.020 → 6.025 s | The delayed -13.717 action now executes. FR calf requests -175.6 Nm and applies -44.4 Nm. This cannot explain the earlier initial support loss. |
| 6.030 → 6.035 s | First active nonfoot contact **in this event window**: FL_calflower1-ground. Maximum post-step joint speed 118.05 rad/s. Earlier startup calf contacts exist and remain recorded. |
| 6.040 → 6.045 s | First Head_lower-ground solve, peak force norm 552.385994 N. Head solves remain active through start 6.065 s. |

Head-contact sampled occupancy is 0.030 s; summed upward solved force × 0.005 s is 6.9442 Ns. The offline reconstructed contact-point horizontal speed reaches 1.1598 m/s. These are sampled diagnostics, not certified continuous impact duration. The previously saved **779.108 N** is the endpoint-forward recomputation at 6.04 s, not a contradiction or an integrated impulse estimate.

In 5.84–6.24 s, seed1 reaches 118.05 rad/s and has 133 saturated joint-substep pairs; Official reaches 0.4981 rad/s and has zero. Recomputing delay, target, PD and clipping from all three traces reproduces the implemented values. There is no observed controller mapping/delay defect to repair on this evidence. Saturation is part of the bad sequence, not proof that a saturation penalty is the right fix.

### Lab: the same policy and nominal batch do not show that event

The existing `run_umi_isaac.py` now optionally reads already configured all-body ContactSensors after each existing simulation/scene update, with `--contact-trace` and `--trace-case 6`. It does not alter sensors, batch, initialization, delay or physics. Native execution in Conda `pawcerto-lab-sim610`, seed1 final4000, nominal16, seed2027 and 850 policy periods exited **0**, recording 3400 substeps for case6. All eight fields emitted by the base evaluation entry match all 849 historical nominal16 rows exactly. The historical probe additionally emitted task-time and target-index fields; those two fields were not part of this endpoint comparison.

Neither Head_upper nor Head_lower has a net-normal contact >1 N in any recorded Lab substep. The 5.84–6.24 s window retains four supported feet, near-upright root and joint speeds around 0.6 rad/s or less. This fills the prior missing head signal for **this nominal case**; it does not invalidate the two previously observed author-protocol inversions or claim all Lab conditions are safe. Lab's public sensor returns body net-normal force, not MuJoCo's per-contact normal-plus-friction vector. Force magnitudes across these APIs are not treated as equivalent.

The full traces also place the earliest physics divergence before actor feedback: in the first zero-torque warmup step 0–0.005 s, joint q/qd agree initially to floating-point precision, but RL calf post-step speed is MuJoCo -1.41260 versus Lab -7.01590 rad/s. Foot ground Fz is respectively [17.41, 17.41, 13.09, 13.09] and [4.26, 4.16, 46.91, 46.70] N (FR, FL, RR, RL). Executed actions first diverge at 0.040 s. This shows a physical response difference before the feedback trajectories separate; it does not attribute the later head event to that first-step difference alone.

### Seed2: a separate calf-support pattern

Seed2 has no head event in this complete MuJoCo case. RR calf contact starts at solve 7.960 s and recurs to episode end, with 5.645 s sampled occupancy, 185.3166 N peak per-contact norm, 36.2635 Ns summed upward impulse and 0.3027 m/s maximum horizontal contact-point speed. Its 5.84–6.24 s maximum joint speed is 1.4983 rad/s with no torque saturation. This is distinct from seed1's violent loss-of-support sequence. It is not automatically a failure based on calf contact, but these traces alone do not establish requested low-posture entry/exit controllability.

### Decision and remaining uncertainty

The diagnostic resolves the event order and confirms a **cross-engine closed-loop behavior difference**, with no observed delay/PD implementation defect. It does not yet identify a single physics parameter or training constraint that should be changed. Preserve the current model/reward and stop before speculative tuning or another undifferentiated rollout.

If further causal attribution is required, the missing discriminating information is the complete constraint generalized force/impulse and free-dynamics response at a **shared frozen state and applied torque**, before feedback diverges. The Lab record currently lacks tangential contact forces, contact moments and pairwise solution; neither the current traces nor already completed geometry/controller checks fill that gap. A bounded common-state constrained/free response comparison was subsequently authorized and executed below. Repeating full nominal rollouts or inferring a reward fix from saturation would not answer this distinction.

### Reproduction and artifacts

```bash
/home/lyb/miniconda3/envs/pawcerto-mujoco/bin/python scripts/trace_umi_contacts.py \
  --checkpoint runs/umi_reconstructed_solver_relaxed_body_speed_scratch_seed1_4096/model_4000.pt \
  --output outputs/diagnostics/umi-contact-5ms-seed1-new
```

The runner defaults to `reference/data/tossing.pkl`, `--cases 16 --case 6 --seed 2027 --seconds 17`. Official and seed2 use their existing checkpoint paths. Local evidence: [seed1 validation](../outputs/diagnostics/umi-contact-5ms-seed1/validation.json), [Official validation](../outputs/diagnostics/umi-contact-5ms-official/validation.json), [seed2 validation](../outputs/diagnostics/umi-contact-5ms-seed2/validation.json), [Lab validation](../outputs/diagnostics/umi-contact-5ms-lab-seed1/validation.json), and [contact/impulse/slip summary](../outputs/diagnostics/umi-contact-5ms-analysis/summary.json). Adjacent JSONL files preserve all physical steps; the analysis script reconstructs contact-point speed offline in a separate MjData and never feeds it into a rollout.

Ten affected MuJoCo tests pass (trace, contact reads, numerical failure propagation, asset/FK). A broader optional `test_umi_policy` import failed because this inference Conda environment has no `pytorch3d`; this was not hidden or repaired by changing the environment. The real fixed-policy replay exercises the affected policy consumer. Python syntax checks pass for both runners and the modified runtime/evaluator. No training, reward/physics edits, new seed, commit or public release was performed by this diagnostic.


### Follow-up executed: initial ground/free response decomposition

The authorized follow-up is a component diagnostic, not a changed nominal evaluation. Two fresh stock Lab processes retained the original 16 environments, seed1 config, reset and robot. Only the free condition disabled collision on the **ground Cuboid** at construction; robot contacts, self-collision, joint properties, gravity and inertia were retained. MuJoCo disabled only the named ground geom's collision masks in its free condition. Each condition used exactly four zero-torque 5 ms steps. Only the first step matches the normal controller's warmup torque; later steps intentionally continue zero torque to observe the component response, rather than the normal post-first-step PD corrections.

Both Lab processes and the MuJoCo probe exited 0. Ground/free initial joint and root states are exactly equal within each engine. The fresh ground first-step q, qd, actual torque and ground forces match the prior trace exactly. Native velocity-limit readbacks are retained. Every applied torque is zero, and both free conditions have no contact at all over all four steps.

| Step end | Ground: max cross-engine joint speed difference | Free: max cross-engine joint speed difference |
|---|---:|---:|
| 0.005 s | 5.603301383 rad/s | 1.487967e-7 rad/s |
| 0.010 s | 4.823316783 rad/s | 2.975934e-7 rad/s |
| 0.015 s | 4.207180724 rad/s | 4.463901e-7 rad/s |
| 0.020 s | 3.783718442 rad/s | 5.951868e-7 rad/s |

First-step free root vertical velocity is -0.04904999584 m/s in Lab versus -0.04905 m/s in MuJoCo. Thus **contact response dominates the observed initial zero-torque discrepancy**; uniform-gravity free motion does not show that discrepancy. The original MuJoCo initial contact solution records 23.50 mm penetration at front feet and 13.92 mm at rear feet, giving a concrete contact-recovery condition behind the first response difference.

This does **not** validate all free articulated dynamics: unforced free fall does not excite inertia differences as a nonzero torque would. Nor does it prove that initial contact recovery alone causes the later 6 s head event. Existing evidence still does not select one solver, friction, collision or training parameter to change. Keep the original-method initialization and physics unchanged. This ground/free subexperiment stops here; its result does not justify a reward patch or parameter sweep. The separately authorized warmup-state intervention follows below.

[Single-step validation](../outputs/diagnostics/umi-contact-5ms-single-step/validation.json) contains the numerical comparison. Adjacent `probe_lab.py`, `probe_mujoco.py`, four condition JSON files and both Lab logs preserve the exact operations, initial states, applied torques and force/state readings. MuJoCo additionally saves bias, passive, constraint generalized force and acceleration for each step. No stock source was patched or rebuilt. GPU ownership was returned immediately after the second process exited.

### Follow-up executed: replace only the genuine Lab PD-warmup physical state

The next authorized causal check asked whether the physical state left by initial contact recovery substantially explains the later head event. One stock Lab cold start saved the **original zero-action PD warmup**, four substeps in the original 16-environment nominal setup. This is different from the preceding zero-torque component probe. All four steps' q, qd, root position and applied torque, and the final EE pose, exactly match the original Lab substep trace. The process exited 0 and GPU ownership was released before the CPU intervention.

A separate CPU script intercepted the existing MuJoCo `mj_forward` at the end of its 20 ms warmup. Its no-intervention control matches the old seven NPZ arrays and complete result JSON exactly (source paths normalized only). The **one intervention** replaces qpos/qvel there with Lab's complete warmup physical state: root position, quaternion, world linear/angular velocity and policy-order joint q/qd. No other intervention was tried.

Lab's source API defines quaternion order xyzw; this is explicitly converted to MuJoCo wxyz and normalized. Free-joint angular velocity is converted from world to local frame. Native `mj_objectVelocity` independently verifies recovered root world linear and angular velocity to 1e-12; the transformed MuJoCo EE FK matches the saved Lab EE pose with maximum matrix-element error **1.7911e-7**. Existing observer construction/pose-history advancement then proceeds normally from the MuJoCo warmup pose. MuJoCo time, policy weights, targets, delay buffers and control clock remain unchanged. Its own qacc warmstart, applied-force buffer and solver state are retained; no incompatible PhysX internal state is copied. The existing forward recomputes the endpoint solution.

| Full 17 s case6 result | Original/no-op control | Lab warmup state intervention |
|---|---:|---:|
| First head solve start | 6.040 s | 5.945 s |
| Head force peak | 552.386 N | 514.705 N |
| Head-contact 5 ms samples | 6 | 4 |
| EE position mean | 13.9180 mm | 12.8454 mm |
| EE orientation mean | 0.0317361 rad | 0.0304645 rad |
| Minimum root up-dot | 0.881770 | 0.900416 |
| Saturated joint-substep pairs | 204 | 231 |
| Zero-ground-support physical steps | 19 | 28 |
| Completion / inversion | Complete / no | Complete / no |

**The intervention does not eliminate the head collision and introduces more saturation and unsupported physical steps. Stop the initialization-only direction.** Some average metrics improve and the event shifts earlier, showing sensitivity to the initial physical state, but the result does not support fixing warmup alone as the solution. It is not a nominal evaluation or a behavior-acceptance pass. No reward, physics, initialization default, extra variant or training change follows from this negative result.

[Intervention validation](../outputs/diagnostics/umi-contact-5ms-warmup-intervention/validation.json), [paired summary](../outputs/diagnostics/umi-contact-5ms-warmup-intervention/summary.json), and [Lab warmup validation](../outputs/diagnostics/umi-contact-5ms-warmup-intervention/lab-validation.json) retain the checks. Adjacent scripts, complete saved Lab state, hook before/after state, two 17 s JSON/NPZ results, per-substep records and Lab log preserve the intervention and its cause/limitations. Both MuJoCo episodes and the Lab collection exited 0.


### Follow-up executed: three nonzero state/torque comparisons

The next component experiment froze three seed1 case6 trace states **before execution**: 5.975 s (last step before first FR calf saturation), 5.980 s (first44.4 Nm saturation) and5.995 s (no-contact solve before the large speed jump). Every state retained its full generalized position/velocity and recorded applied torque. The [fixed plan](../outputs/diagnostics/umi-contact-nonzero-state-20260912/plan.json) records source hash, state arrays and assignments.

One successful stockLab startup used16 environments for paired ground/free replicas of these three states and exactly one simultaneous5 ms step. The free condition translated the entire robot upward2 m, preserving internal configuration, orientation, velocity, torque, self-collision and joint constraints; only external-ground proximity changed. It is a rigid-translation component intervention, not a new nominal rollout. All free body-normal sensors were zero and MuJoCo free cases had zero contact/constraint dimensions. Input readbacks matched with maximum root-position error2.38e-7 m, root-linear-velocity error1.49e-8 m/s, exact float32 joint q/qd and exact applied torque. Replicate velocity spread stayed below6.16e-5 rad/s.

| Frozen state | Ground max Lab–MuJoCo joint-speed difference | Free max Lab–MuJoCo joint-speed difference |
|---|---:|---:|
| 5.975 s | 6.0431 rad/s | 4.4507 rad/s |
| 5.980 s | 13.6998 rad/s | 13.1712 rad/s |
| 5.995 s | 14.2934 rad/s | 14.2934 rad/s |

The5.995 s ground case also has no contact in either engine. Therefore contact response cannot explain the whole nonzero-input discrepancy. In the free cases, Lab's measured velocity increment matches its own native`M^-1*(tau-C-G)` prediction within1.44e-5 rad/s. MuJoCo's cold ground steps, with explicitly zero`qacc_warmstart`, match the corresponding historical post-step velocities within3.56e-15 rad/s. PhysX contact caches are not directly transferable or fully observable; these cold checks qualify only the selected states, not all history-dependent contact response. [Response comparison](../outputs/diagnostics/umi-contact-nonzero-state-20260912/response-comparison.json) preserves full per-condition numbers and force API limits.

The first actual Lab startup stopped **before any physical step** because the current stockLab`write_root_link_velocity_to_sim_index` computed the COM buffer but supplied the link-velocity buffer to the native COM-velocity setter. The saved native COM velocity exactly equals the requested link velocity; the missing`omega cross R*com_offset` predicts the observed0.03430 m/s error. The probe was locally repaired by explicitly calculating COM velocity and using the public COM setter; no Lab source was patched. This API defect was exercised by the new probe, not by the original zero-velocity reset, and is not presented as the cause of the historical head collision. The failed inputs/logs remain in`lab-attempt1/`; [input-mapping diagnosis](../outputs/diagnostics/umi-contact-nonzero-state-20260912/input-mapping-diagnosis.json) records the source line and saved-data check. A preceding shell invocation failed before Kit bootstrap because an existing environment flag was not inherited; it is also retained. Successful collection retained the stock actuator CUDA-graph-capture error and its logged eager fallback rather than hiding it.

### Actionable finding: source inertia orientation and isolated correction

The comparison explicitly maps MuJoCo root-link world linear/local angular coordinates into PhysX root-COM world linear/world angular coordinates, with the native joint permutation. Source fixed bodies are aggregated about each of the25 native body COMs. Masses, COM locations, Jacobians and gravity agree to float precision, but native inertia orientations differ. Native tensors reconstruct as`Q D Q^T`, whereas the source URDF tensors match`Q^T D Q` within4.87e-8 kg m². Native`M` rebuilt from its own`J` and body inertias agrees within1.22e-7; the original cross-engine mappedM mismatch0.0204–0.0207 shrinks to5.01e-7–9.29e-7 when the source-consistent tensors replace those orientations in the CPU reconstruction. These matrix entries have mixed generalized-coordinate units. See [model mapping](../outputs/diagnostics/umi-contact-nonzero-state-20260912/model-input-comparison.json), [inertia diagnosis](../outputs/diagnostics/umi-contact-nonzero-state-20260912/inertia-orientation-diagnosis.json), and [actual USD/source properties](../outputs/diagnostics/umi-contact-nonzero-state-20260912/source-inertia-properties.json).

The maintained [converter](../scripts/convert_umi_usd.py) now derives principal moments and a proper right-handed eigenvector basis directly from the source mergedURDF COM tensor, including any inertial-origin rotation, then writes USD`MassAPI` attributes. It does not blindly invert every quaternion and does not change officialLab/PhysX. The actual helper was applied to an isolated copy at[the new USD](../outputs/diagnostics/umi-contact-nonzero-state-20260912/asset-source-inertia/go2_arx5_merged.usda). After reopening, all25 source tensors match within2.73e-9 kg m²; only19 principalAxes/diagonalInertia attribute values differ. Other prim types, schemas, attributes and relationships match, and the original default pointer is unchanged. [Asset validation](../outputs/diagnostics/umi-contact-nonzero-state-20260912/asset-validation.json) and [helper checks](../outputs/diagnostics/umi-contact-nonzero-state-20260912/converter-helper-checks.json) include reapplication to an already correct asset and a nonzero-origin-rotation tensor case.

The CPU result was followed by native validation below; it is not by itself a safe-policy claim. The prepared[candidate training configuration](../outputs/diagnostics/umi-contact-nonzero-state-20260912/candidate-train-input.json) retains the effective body-speed-v3/joint1000 recipe,4096×24×4000 and random seed0, adds only the71-training-trajectory selection, and removes copied historical execution metadata. CLI endpoint/checkpoint interval are made explicit in the config. The[training plan](../outputs/diagnostics/umi-contact-nonzero-state-20260912/candidate-training-plan.json) records exact differences, the new asset and the runtime-path consumption requirement. At that diagnostic stage no new training had started; the later fixed training/evaluation outcome is recorded below.


### Corrected USD: actual native dynamics verification

The isolated source-inertia asset then ran the **same three fixed inputs**,16-environment ground/free assignment and one simultaneous5 ms Lab step. No new selected states or parameter variants were added. The process exited0 and GPU ownership was released. The native25-body inertia tensors match aggregated source tensors within9.73e-8 kg m²; mapped generalizedM maximum difference is8.94e-7, with relativeFrobenius error below8.87e-8. This uses actual PhysX getter outputs after the new USD loads, not only a CPU counterfactual.

| Frozen state | Original USD free max joint-speed difference | Corrected USD free max joint-speed difference | Corrected USD ground max difference |
|---|---:|---:|---:|
| 5.975 s | 4.4507 rad/s | 2.05e-5 rad/s | 6.0606 rad/s |
| 5.980 s | 13.1712 rad/s | 4.11e-5 rad/s | 7.9545 rad/s |
| 5.995 s | 14.2934 rad/s | 2.25e-5 rad/s | 2.20e-5 rad/s |

The corrected free-error column bounds all preassigned replicas; raw JSON also retains the first-pair values. All corrected free cases remain contact-free. The5.995 s ground case also has no contact and now agrees similarly. The two states that actually contact the ground still differ, separating the repaired **inertia/free-dynamics mapping** from the remaining **contact/solver response**. PhysX and MuJoCo retain their original integrators, contacts, geometry and constraints; no friction/contact/solver tuning was performed. Neither the corrected USD single steps nor lower matrix error proves that the existing frozen policy no longer hits its head over17 seconds.

[Corrected native model comparison](../outputs/diagnostics/umi-contact-nonzero-state-20260912/corrected-lab/model-input-comparison.json), [corrected responses](../outputs/diagnostics/umi-contact-nonzero-state-20260912/corrected-lab/response-comparison.json), and [combined validation](../outputs/diagnostics/umi-contact-nonzero-state-20260912/validation.json) retain the native arrays, response errors, process statuses and unchanged-source hashes. The original failed input attempt, successful stock-asset attempt and corrected-asset attempt are separate records. SourceLab/PhysX, old USD, default USD pointer, old checkpoints and old rollout records were not changed.

### Fixed training and final evaluation outcome

The corrected-asset candidate completed random seed0,4096 environments×24 rollout steps×4000 updates:393,216,000transitions, process exit0, finite metrics/parameters and no resume or recipe change. [Training verification](../runs/umi_source_inertia_train71_seed0_4096/training-verification.json) binds the full checkpoint to the corrected USD and71-training-ID selection. The real training/evaluation consumers now prioritize explicit asset paths, then saved checkpoint asset identity; legacy models retain the workspace default. Actual checkpoint training asset/runtime fields remain distinct from adjacent execution configs and actor-only exports.

Final15validation and old/new corrected-USD author15validation completed before the preselected4000 candidate was recorded as frozen. Both policies had0 inverted episodes among all502 author completions (reported latest500 separately). The subsequent historical16-target benchmark completed in both engines; the new native5ms summary on/off result is exactly identical across849×16 endpoint rows and all9 fields. The [native comparison](../outputs/umi-source-inertia-train71-20260912/native-benchmark-paired-verification.json) records the shared asset and targets. The exported actor also ran the actual17s MuJoCo consumer on validation trajectory7, with all7 saved arrays exactly matching the existing full-checkpoint replay. [Export consumer](../outputs/umi-source-inertia-train71-20260912/export-consumer-17s/verification.json).

The once-only15test evaluation failed:14 trajectories completed17s, but trajectory1 fell and contacted the ground with its head before the numerical guard caught BADQACC(DOF21) at10.525s. Its invalid prefix remains preserved, and the evaluation process returned2 rather than allowing the engine restart to masquerade as a complete episode. The14-case mean excludes this failure and is not a15-case result. Validation trajectory40's separate finger/head self-contact while upright is not conflated with the test head-ground failure. [Test summary](../outputs/umi-source-inertia-train71-20260912/test-model4000/summary.json), [failure](../outputs/umi-source-inertia-train71-20260912/test-model4000/trajectory_001.json), [5ms evidence](../outputs/umi-source-inertia-train71-20260912/test-model4000/trajectory_001-substeps.json).

The source-inertia repair remains validated at its component evidence level. Full fixed-policy acceptance has failed; no test retry, tuning, extra training or publication follows. GPU work has finished and ownership was returned.
