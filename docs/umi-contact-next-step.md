# UMI contact diagnosis: completed findings and stop boundary

The bounded diagnosis is complete. The original three trained seeds, dual-engine evaluations and exports remain the baseline. **No control mapping/delay/PD defect was found.** Seed1 case6 loses support and saturates before its MuJoCo head impact, while the matching Lab case stays stable. Ground/free single-step comparison identifies contact response as the dominant initial difference under the tested zero-torque condition. Replacing the genuine Lab PD-warmup physical state in MuJoCo does **not** remove the later head collision. This negative intervention closes the initialization-only direction.

No single solver, friction, reward or physics correction is justified by these results. The plan's diagnostic stop boundary has been reached: no parameter sweep, retraining, acceptance relaxation or product-physics change. Default control, initialization, rewards, physics and policy weights remain unchanged. The delivered changes are optional readouts, their verified runners/tests and this report.

Evidence below covers three MuJoCo 17 s trace-on/off pairs, one matching Lab nominal16 trace, the bounded ground/free component probe, and one warmup-state causal intervention with a no-op control. Trace-on/off and no-op controls preserve the original results exactly; no result is promoted to behavioral acceptance. The final paired intervention results and artifacts are in the last section.

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
