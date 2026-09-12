# DeepWBC PRE solver-residual six-axis experiment

This experiment is motivated by original Gym simultaneous sensor modes: default equals explicit FD+constraint, FD equals body gravity in the previous/PRE foot frame, while raw net acceleration and native joint reaction failed equivalence. It tests one new candidate without changing DeepWBC production mode or UMI runtime.

For each body, in world coordinates, the candidate is

```
F = m (a_reported - a_free) + m g
T_origin = I_PRE (alpha_reported - alpha_free) + (R_PRE r_COM) cross F
wrench_body = [R_PRE.T F, R_PRE.T T_origin]
```

`a_free` and `alpha_free` come from existing `StockSolverForce.predict`, using all physical bodies and all 20 DOFs, PRE public M/C/G/J, native actuation, COM state and body-prim rotation. No extra gyro torque is added to the angular residual: subtracting free acceleration already removes its contribution. The nonzero lever is tested analytically; these physical foot COM offsets are zero.

Public NVIDIA PhysX source at SHA `4f2103c3a9052906296defb12166753450ef787c` supports the timing and flag interpretation, but is not established as the installed binary's exact source:

- [forwardDynamic2.cu](https://github.com/NVIDIA-Omniverse/PhysX/blob/4f2103c3a9052906296defb12166753450ef787c/physx/source/gpuarticulation/src/CUDA/forwardDynamic2.cu): articulation free dynamics contains unconditional gyro at line 1271; reported acceleration contains free and solver delta velocity/dt at line 3237.
- [inverseDynamic.cu](https://github.com/NVIDIA-Omniverse/PhysX/blob/4f2103c3a9052906296defb12166753450ef787c/physx/source/gpuarticulation/src/CUDA/inverseDynamic.cu): public C includes gyro at line 786 without rigid-body gyro-flag selection.

Actual composed properties are gyro=true and retain=false for all 27 bodies. No new flag branch is required, so the original helper constructor checks, `predict` arithmetic and existing force residual remain unchanged. A briefly relaxed gyro check during preparation was restored after the actual property readback; it did not change the sampled true-gyro arithmetic.

CPU verification: `pawcerto-lab-sim610/bin/python -m unittest tests.test_solver_force tests.test_deepwbc_solver_wrench` passed 9 tests. Coverage includes full non-diagonal inertia, PRE rotation, nonzero COM lever, all six residual axes, original Go2 velocity/gyro branches, and exact cancellation of equal reported/free acceleration leaving gravity only. This establishes analytic implementation behavior, not Gym equivalence.

The GPU operator executes `scripts/probe_deepwbc_solver_wrench.py` in the official Lab environment. Protocol is the same one-env static/free-fall/moving-contact protocol, 5 ms explicit PD, 40 recorded steps per scene, static warmup 200 and other warmup 1. The probe saves native properties, PRE dynamics, POST accelerations, candidate six-axis wrenches and solver-only wrenches. `scripts/analyze_deepwbc_solver_wrench.py` compares both all 5 ms samples and each four-step consumer's last sample against original Gym FD+constraint and constraint-only channels.

Real collection: attempt 3 exited 0 and saved 120 rows to `outputs/isaac/deepwbc-solver-wrench-lab-20260912-attempt3`. Native arrays contain M `(120,1,26,26)`, J `(120,1,27,6,26)`, full 20-DOF native torque and velocity, 27-body reported and free accelerations, and 4x6 candidates. Actual properties are body damping zero, gyro true, retain false, maximum linear speed 1000 m/s and angular speed approximately 1000 rad/s. `comparison.json` contains complete per-axis results.

| Scene | All-5ms force RMSE xyz (N) | All-5ms torque RMSE xyz (Nm) | Consumer Fz squared MAE (N²) | Contact-bit disagreement |
|---|---|---|---|---|
| Static contact | .215, .192, .295 | .0000811, .000132, .0000681 | 2.725 | 0% |
| Free fall | .00307, .0170, .000112 | .00000845, .00000930, .00000729 | .000102 | 0% |
| Moving contact | 1.448, .452, .593 | .000181, .001110, .000167 | 8.405 | 0% |

All-5ms and four-step-last-sample contact bits both agree in all three scenarios. This is a substantial improvement over the already rejected raw net-body candidate, but it does not establish amplitude equivalence. Moving-contact maximum single-step force difference is 15.658 N, torque difference .01235 Nm. Consumer last-sample maxima are 2.028 N and .001502 Nm. Lab foot positions are exactly equal to the prior Lab raw-net probe, showing the dynamics reads did not alter this observed trajectory. Gym/Lab position RMSE remains .138–1.695 mm.

The largest mismatch occurs at moving-contact recorded step 27, FL foot: Lab/Gym native contact forces are `[27.012,-27.435,51.335]` and `[0.0000013,0,9.861]` N, respectively; foot lateral positions differ by 3.63 mm. Therefore materially different contact solutions coexist with the signal mismatch. The present separate-trajectory comparison cannot isolate reconstruction error from contact physics mismatch. No production signal switch was made.

The next evidence-bearing step is to use the separately owned contact/friction physics findings to align the contact solution, then compare this unchanged candidate on matched trajectories; a source-level or same-engine sensor oracle would be needed to prove the exact six-axis solver convention. Do not tune this formula to fit the current cross-engine residual.

Preserved routine failures: attempt 1 rejected an incompatible `--headless` AppLauncher option; the operator used the verified `--visualizer none` entry. Attempt 2 initialized but failed at the first gravity expression because native static tensors were on CPU while the target device was CUDA. The probe now explicitly transfers native getter tensors to the requested device. Original logs and partial outputs remain under `outputs/deepwbc-native-wrench-execution-20260912` and the attempt-2 output directory. The previous-crash minidump upload warning in the successful log is not the current process failure.

Final CPU tests passed 9/9 after restoring the original constructor checks. No install, commit, production runtime edit, training service change, or additional GPU run was performed by this experiment owner.

## PRE frame, velocity and reward scale follow-up

`frame-reward-analysis.json` is reproduced by the same CPU analyzer. Gym PRE rotations/velocities are recovered from the previous recorded POST, so only this diagnostic omits each scenario's first row. Rotating the Lab candidate into the Gym PRE frame is a diagnosis, not a formula change. Moving force RMSE changes from `[1.467,.458,.601]` to `[1.446,.421,.643]` N and maximum force error from 15.658 to 15.663 N: frame mismatch does not explain the peak. Static RMSE changes from `[.216,.194,.299]` to `[.188,.170,.237]` N, so some static component error is frame-related. PRE linear/angular velocity RMSE is .0126 m/s/.0346 rad/s static, .00139/.00496 free fall, and .0700/.309 moving. These are observed trajectory differences, not a quantified causal explanation of all residual error.

| Consumer distribution | Gym Fz² mean | Candidate Fz² mean | MAE / Gym mean | Signed bias / Gym mean |
|---|---:|---:|---:|---:|
| Static | 20.842 | 19.198 | 13.07% | -7.89% |
| Free fall | .20056 | .20062 | .0508% | +.0311% |
| Moving | 83.644 | 78.579 | 10.05% | -6.06% |

Moving Gym Fz² median is .2088 N², 95th percentile 358.05 N², maximum 480.0 N². Hence MAE 8.405 N² is not negligible relative to the reward input; agreement of threshold bits cannot justify production equivalence. These numbers describe the raw per-foot squared-force input. A constant reward scale preserves the relative ratios; they are not complete episode reward or learning results.

## Core-owner interface

The explicit experimental interface is now named `StockSolverForce.approximate_sensor_wrench(reported_acceleration, free_acceleration, *, link_rotation, com_offset_b, applied_force_w)`. Inputs are tensors on the model device in physical-body order: reported/free acceleration `(N,B,6)` in world COM coordinates, PRE body-prim rotation `(N,B,3,3)`, fixed COM displacement in prim axes `(N,B,3)`, and world force applied at COM `(N,B,3)` (only gravity in this verified probe). Output is `(N,B,6)` with force then moment in PRE body axes at the body origin. Select FL/FR/RL/RR only after reconstructing free dynamics for all bodies/DOFs. The method does not provide an arbitrary external applied torque channel.

The existing UMI `residual_force` and `predict` remain unchanged, and there is no new production mode. For any integration, the core owner must preserve PRE snapshots across the physics step and call this explicitly approximate interface. Root owns the decision whether the measured error is acceptable; this report does not authorize or implement a default replacement.

The successful run used the prior name `residual_wrench`; the final rename to `approximate_sensor_wrench` and restoration of the original constructor guard do not alter arithmetic. Operator source snapshots preserve the exact executed script, and the final CPU tests pass under the explicit approximate name.

Final renamed-method CPU replay on all 120 captured reported/free acceleration rows reproduces the executed CUDA candidates with maximum force difference 3.815e-6 N and moment difference 1.863e-9 Nm (`final-method-cpu-replay.json`, float32 device arithmetic). This validates the final method against the captured run, not independent correctness of the free-dynamics prediction.
