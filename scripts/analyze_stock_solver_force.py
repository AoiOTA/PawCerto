"""Experimental offline stock-API model residual, never a reward backend.

The expanded capture exercises full floating-base M/C/G/J and recursive CoM
kinematic bias. Unverified body properties and unit/decomposition limits remain
explicit: finite residuals do not establish solver-only or Gym sensor fidelity.
The earlier incomplete capture supports only a zero-speed gravity response.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FOOT_ORDER = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
MISSING_CAPTURE_REQUIREMENTS = {
    "actual_body_masses": "Runtime masses in captured body order, not assumed URDF nominal masses.",
    "before_link_and_com_poses": "World link and CoM poses with quaternion convention for the exact pre-step dynamics state.",
    "joint_tree_and_frames": "Parent/child body indices, joint type, DOF index, local joint anchors/rotations and axis, or verified equivalent world anchors/axes.",
    "generalized_and_jacobian_contract": "Root coordinate order/reference point, Jacobian linear reference point, world-frame/angular convention and active DOF order.",
    "complete_free_force_inputs": "Applied same-step actuation plus external wrenches, gravity flags, damping, friction and armature settings; identify any source terms already included in M/C/G.",
    "step_refresh_contract": "Pre M/C/G/J/pose/velocity capture ordering and post total-acceleration refresh/averaging semantics for the same physics step.",
}


def require_array(capture, key):
    value = np.asarray(capture[key], dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite capture field: {key}")
    return value


def com_bias_acceleration(positions, velocities, parents):
    """World CoM velocity bias, linear then angular; root generalized qdd=0.

    PhysX forwardDynamic2.cu 1172--1207 and 2118--2121: each relative
    link velocity contributes a local Coriolis term, propagated down the tree.
    This uses no numerical differentiation or fitted coefficients.
    """
    bias = np.zeros_like(velocities)
    for child in range(1, positions.shape[1]):
        parent = parents[child]
        r = positions[:, child]-positions[:, parent]
        wp = velocities[:, parent, 3:]
        dw = velocities[:, child, 3:]-wp
        dv = velocities[:, child, :3]-velocities[:, parent, :3]-np.cross(wp, r)
        bias[:, child, 3:] = bias[:, parent, 3:]+np.cross(wp, dw)
        bias[:, child, :3] = (bias[:, parent, :3]+np.cross(bias[:, parent, 3:], r)
                             +np.cross(wp, np.cross(wp, r))+2*np.cross(wp, dv)+np.cross(dw, dv))
    return bias


def check_free_dynamics_step(capture_path, summary_path, metadata_path, output, step):
    """One saved, independently reset no-constraint step; never adjusts tolerance."""
    source = json.loads(summary_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if source["errors"] or not 0 <= step < source.get("steps", 0) <= 3:
        raise ValueError("No clean authorized independent sample at this index")
    exclusion = source["constraint_exclusion"]
    if (exclusion["fixed_base"] or exclusion["joint_count"] != 24
            or exclusion["max_fixed_tendons"] or exclusion["max_spatial_tendons"]
            or exclusion["external_constraints"]
            or not exclusion["mass_com_inertia_unchanged_vs_original_capture"]
            or len(exclusion["robot_collisions"]) != 34
            or any(r["enabled"] is not False for r in exclusion["robot_collisions"])):
        raise ValueError("Required full-articulation/constraint-exclusion evidence is incomplete")
    if source["simulation_config"]["physics"]["enable_external_forces_every_iteration"]:
        raise ValueError("External forces deferred to TGS require different classification")
    if (metadata["root_generalized_order"] != "world linear xyz, world angular xyz, internal DOF order"
            or metadata["jacobian_reference"] != "world COM linear then world angular; root included"):
        raise ValueError("Unsupported generalized/CoM frame contract")
    names = source["body_names"]
    body_properties = {
        "physxRigidBody:disableGravity":False,
        "physxRigidBody:linearDamping":0., "physxRigidBody:angularDamping":0.,
        "physxRigidBody:maxLinearVelocity":1000.,
        "physxRigidBody:maxAngularVelocity":float(np.float32(np.rad2deg(1000.))),
        "physxRigidBody:maxDepenetrationVelocity":1.,
        "physxRigidBody:enableGyroscopicForces":True, "physxRigidBody:retainAccelerations":False,
    }
    body_entries = {entry["path"]:entry for entry in source["usd_physics"]}
    for path in source["link_paths"][0]:
        entry = body_entries[path]
        if ("PhysxRigidBodyAPI" not in entry["schemas"]
                or any(entry["attributes"].get(k) != v for k,v in body_properties.items())):
            raise ValueError(f"Composed body damping/limits/gravity/gyro/retain property mismatch: {path}")
    parents = {j["body1_index"]:j["body0_index"] for j in metadata["joint_tree"]}
    if (len(names) != 25 or set(parents) != set(range(1, 25))
            or any(not 0 <= p < c for c, p in parents.items())
            or any(source["metatype"]["link_parent_indices"][names[c]] != p for c, p in parents.items())):
        raise ValueError("Full physical body tree is not the validated parent-before-child articulation")
    sample = source["independent_samples"][step]
    if sample["index"] != step or min(sample["joint_min_margin_pre_rad"], sample["joint_min_margin_post_rad"]) <= .1:
        raise ValueError("Joint-limit exclusion did not pass for this sample")
    prefix = f"step{step:02d}"
    with np.load(capture_path, allow_pickle=False) as capture:
        # Preserve every consumed source field in this result; the producer's
        # cumulative capture file may subsequently acquire additional samples.
        saved = {k:require_array(capture, k) for k in capture.files
                 if k.startswith("static__") or k.startswith(prefix+"_")}
    pre = lambda key:saved[prefix+"_pre__"+key]
    post = lambda key:saved[prefix+"_post__"+key]
    for field in ("get_dof_stiffnesses", "get_dof_dampings", "get_dof_friction_coefficients",
                  "get_dof_friction_properties", "get_dof_armatures", "get_disable_gravities"):
        if np.any(saved["static__"+field] != 0):
            raise ValueError(f"Unmodeled nonzero property {field}")
    for composer in ("instantaneous_wrench_composer", "permanent_wrench_composer"):
        if source["samples"][prefix+"_pre"][composer+"_active"]:
            raise ValueError("Active external-wrench composer")
        for field in ("global_force_w", "global_torque_w", "global_force_at_com_w", "local_force_b", "local_torque_b"):
            if np.any(pre(composer+"_"+field) != 0):
                raise ValueError("Nonzero external-wrench buffer")
    if any(np.any(v != 0) for k,v in saved.items() if k.startswith(prefix+"_post__contact_")):
        raise ValueError("Reported contact exists in the intended free-dynamics sample")
    M, C, G, J = [pre(k) for k in ("get_generalized_mass_matrices", "get_coriolis_and_centrifugal_compensation_forces",
                                  "get_gravity_compensation_forces", "get_jacobians")]
    if M.shape != (1,24,24) or J.shape != (1,25,6,24):
        raise ValueError("Full floating-base matrices are required")
    tau = pre("get_dof_actuation_forces")
    expected_tau = np.zeros_like(tau)
    expected_tau[:,source["gym_to_lab"]] = saved[prefix+"__command_torque_gym_order"]
    if not np.array_equal(tau, expected_tau) or not np.array_equal(tau, pre("data_applied_torque")):
        raise ValueError("Same-step submitted and actual torque differ")
    velocity = pre("get_link_velocities")
    qd = pre("get_dof_velocities")
    generalized_velocity = np.concatenate((pre("get_root_velocities"),qd),axis=-1)
    if (np.any(np.abs(qd) >= saved["static__get_dof_max_velocities"])
            or np.max(np.linalg.norm(velocity[...,3:],axis=-1)) >= 1/source["dt"]):
        raise ValueError("Captured pre state reaches a source nonlinear velocity preprocessing threshold")
    Q = np.concatenate((np.zeros((1,6)),tau),axis=-1)
    qdd = np.linalg.solve(M,(Q-C-G)[...,None])[...,0]
    bias = com_bias_acceleration(pre("data_body_com_pose_w")[...,:3],velocity,parents)
    predicted = np.einsum("elij,ej->eli",J,qdd)+bias
    reported = post("get_link_accelerations")
    error = np.abs(reported-predicted)
    tolerance = 1e-3+1e-4*np.abs(predicted)
    for value in (predicted, reported, error, tolerance):
        if value.shape != (1,25,6) or not np.isfinite(value).all():
            raise ValueError("All 25x6 prediction/report/error values must be finite")
    components = {}
    for label, offset, unit in (("linear",0,"m/s^2"),("angular",3,"rad/s^2")):
        err, tol = error[...,offset:offset+3], tolerance[...,offset:offset+3]
        idx = np.unravel_index(np.argmax(err/tol),err.shape)
        fullidx = (*idx[:-1],idx[-1]+offset)
        components[label] = {"unit":unit, "max_abs_error":float(err.max()),
            "max_error_over_tolerance":float((err/tol).max()), "failed_elements":int(np.sum(err>tol)),
            "worst_body":names[idx[1]], "worst_axis":"xyz"[idx[2]],
            "worst_prediction":float(predicted[fullidx]), "worst_reported":float(reported[fullidx]),
            "worst_abs_error":float(error[fullidx]), "worst_tolerance":float(tolerance[fullidx])}
    report = {"status":"passed" if np.all(error<=tolerance) else "failed", "passed":bool(np.all(error<=tolerance)),
        "step":step, "threshold":"elementwise abs_error <= 1e-3 + 1e-4*abs(prediction)",
        "elements_checked":150, "components":components,
        "jacobian_velocity_max_abs_error":float(np.max(np.abs(np.einsum("elij,ej->eli",J,generalized_velocity)-velocity))),
        "solve_residual_max_abs":float(np.max(np.abs(np.einsum("eij,ej->ei",M,qdd)-(Q-C-G)))),
        "capture_path_at_read":str(capture_path.resolve()), "capture_sha256_at_read":hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "summary_snapshot":source, "metadata_snapshot":metadata,
        "boundary":"One saved stock PhysX independent single-step free-dynamics numerical check; fixed full 25x6 threshold. No fitted coefficients, threshold changes, optimizer steps or original Gym sensor equivalence claim."}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    saved.update(predicted_link_acceleration=predicted, reported_link_acceleration=reported,
                 absolute_error=error, tolerance=tolerance, recursive_com_bias=bias, model_qdd=qdd)
    np.savez_compressed(output.with_suffix(".npz"),**saved)
    lines = [f"# Independent stock free-dynamics sample {step}", "", f"Result: **{report['status']}** over all 150 elements.",
             "", report["threshold"], "", "| Component | Units | Max absolute error | Max error / tolerance | Failed / 75 | Worst body / axis |",
             "|---|---|---:|---:|---:|---|"]
    for key, c in components.items():
        lines.append(f"| {key} | {c['unit']} | {c['max_abs_error']:.9g} | {c['max_error_over_tolerance']:.9g} | {c['failed_elements']} | {c['worst_body']} / {c['worst_axis']} |")
    lines.extend(["",report["boundary"],""])
    output.with_suffix(".md").write_text("\n".join(lines))
    return report


def analyze_expanded(capture_path, summary_path, metadata_path, output, expected_max_angular_velocity_deg_s=1000.):
    source = json.loads(summary_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if source["status"] != "capture_complete" or source["errors"]:
        raise ValueError("The expanded capture did not complete cleanly")
    if source["simulation_config"]["physics"]["enable_external_forces_every_iteration"]:
        raise ValueError("External-forces-every-TGS-iteration needs a different solver/FD classification")
    if metadata["root_generalized_order"] != "world linear xyz, world angular xyz, internal DOF order":
        raise ValueError("Unsupported or missing generalized-coordinate contract")
    if metadata["jacobian_reference"] != "world COM linear then world angular; root included":
        raise ValueError("Jacobian reference point/order is not the required CoM contract")
    names = source["body_names"]
    parents = {j["body1_index"]:j["body0_index"] for j in metadata["joint_tree"]}
    if set(parents) != set(range(1, len(names))):
        raise ValueError("Joint tree must cover each nonroot physical body exactly once")
    if any(parent < 0 or parent >= child for child, parent in parents.items()):
        raise ValueError("Captured physical body order must be parent-before-child")
    expected_parents = source["metatype"]["link_parent_indices"]
    if any(expected_parents[names[child]] != parent for child, parent in parents.items()):
        raise ValueError("USD joint tree does not match the engine articulation metatype")
    foot_ids = [names.index(name) for name in FOOT_ORDER]
    dt = float(source["dt"])
    # Inspect this capture's own composed properties. Never infer old capture
    # settings from a later runtime fix or from the intended scene config.
    if not np.isfinite(expected_max_angular_velocity_deg_s) or expected_max_angular_velocity_deg_s <= 0:
        raise ValueError("Expected angular velocity limit must be finite and positive")
    # USD stores this property as float32; require its exact rounded value.
    angular_limit_deg_s = float(np.float32(expected_max_angular_velocity_deg_s))
    angular_limit_rad_s = float(np.deg2rad(angular_limit_deg_s))
    angular_limit_matches_original = bool(np.isclose(angular_limit_rad_s, 1000., rtol=1e-7, atol=0.))
    angular_unit_note = (
        f"The checked composed angular limit is {angular_limit_deg_s:.10g} deg/s "
        f"({angular_limit_rad_s:.10g} rad/s). "
        + ("It matches the original Gym configured 1000 rad/s within USD float32 rounding. "
           if angular_limit_matches_original else
           "It differs from the original Gym configured 1000 rad/s. ")
        + "This does not establish full physical equivalence or change earlier captures."
    )
    expected_body_properties = {
        "physxRigidBody:disableGravity":False,
        "physxRigidBody:linearDamping":0., "physxRigidBody:angularDamping":0.,
        "physxRigidBody:maxLinearVelocity":1000., "physxRigidBody:maxAngularVelocity":angular_limit_deg_s,
        "physxRigidBody:maxDepenetrationVelocity":1.,
        "physxRigidBody:enableGyroscopicForces":True, "physxRigidBody:retainAccelerations":False,
    }
    stage_bodies = {r["path"]:r for r in source["usd_physics"]}
    incomplete_body_properties = []
    for path in source["link_paths"][0]:
        entry = stage_bodies[path]
        missing = [key for key, expected in expected_body_properties.items()
                   if key not in entry["attributes"] or entry["attributes"][key] != expected]
        if "PhysxRigidBodyAPI" not in entry["schemas"] or missing:
            incomplete_body_properties.append({"path":path, "missing_or_mismatched_fields":missing})
    all_body_properties_match = not incomplete_body_properties
    rows, arrays = [], {}
    tracked = {k:[] for k in ("generalized_velocity", "known_generalized_force", "free_qdd",
                             "recursive_com_bias", "model_free_link_acceleration",
                             "reported_total_link_acceleration", "model_residual_force",
                             "velocity_difference_link_acceleration", "finite_difference_force_sensitivity")}
    zero_joint_properties = ("get_dof_stiffnesses", "get_dof_dampings", "get_dof_friction_coefficients",
                             "get_dof_friction_properties", "get_dof_armatures")
    with np.load(capture_path, allow_pickle=False) as capture:
        masses = require_array(capture, "static__get_masses")
        if masses.shape != (1, len(names)) or np.any(masses <= 0):
            raise ValueError("This bounded capture requires one articulation with positive measured masses")
        for name in zero_joint_properties:
            if np.any(require_array(capture, "static__"+name) != 0):
                raise ValueError(f"Nonzero {name} requires explicit modeling; it is not silently omitted")
        if np.any(require_array(capture, "static__get_disable_gravities") != 0):
            raise ValueError("Per-body disabled gravity needs separate compensation semantics")
        for step in range(source["steps"]):
            pre, post = f"step{step:02d}_pre", f"step{step:02d}_post"
            get_pre = lambda key:require_array(capture, pre+"__"+key)
            for composer in ("instantaneous_wrench_composer", "permanent_wrench_composer"):
                if source["samples"][pre][composer+"_active"]:
                    raise ValueError("Active external-wrench composer is not modeled in this capture-specific diagnostic")
                for field in ("global_force_w", "global_torque_w", "global_force_at_com_w", "local_force_b", "local_torque_b"):
                    if np.any(get_pre(composer+"_"+field) != 0):
                        raise ValueError("Nonzero external-wrench buffer cannot be silently omitted")
            M, C, G, J = [get_pre(key) for key in ("get_generalized_mass_matrices",
                            "get_coriolis_and_centrifugal_compensation_forces", "get_gravity_compensation_forces", "get_jacobians")]
            qd, tau = get_pre("get_dof_velocities"), get_pre("get_dof_actuation_forces")
            velocity = get_pre("get_link_velocities")
            generalized_velocity = np.concatenate((get_pre("get_root_velocities"), qd), axis=-1)
            if M.shape != (1, qd.shape[1]+6, qd.shape[1]+6) or J.shape != (1, len(names), 6, qd.shape[1]+6):
                raise ValueError("M/J do not cover all floating-base generalized coordinates and CoMs")
            command = require_array(capture, f"step{step:02d}__command_torque_gym_order")
            internal_command = np.zeros_like(tau)
            internal_command[:, source["gym_to_lab"]] = command
            if not np.array_equal(tau, internal_command) or not np.array_equal(tau, get_pre("data_applied_torque")):
                raise ValueError("Submitted command and actual same-step actuation differ")
            positions = get_pre("data_body_com_pose_w")[..., :3]
            bias = com_bias_acceleration(positions, velocity, parents)
            # This known-input model excludes unverified child-link body damping
            # and clamps. Q is not labeled a fully verified simulator free force.
            Q = np.concatenate((np.zeros((1, 6)), tau), axis=-1)
            qdd = np.linalg.solve(M, (Q-C-G)[..., None])[..., 0]
            free_acc = np.einsum("elij,ej->eli", J, qdd)+bias
            total_acc = require_array(capture, post+"__get_link_accelerations")
            residual_force = masses[..., None]*(total_acc-free_acc)[..., :3]
            after_velocity = require_array(capture, post+"__get_link_velocities")
            dv_acc = (after_velocity-velocity)/dt
            sensitivity = masses[..., None]*(dv_acc-total_acc)[..., :3]
            values = (generalized_velocity, Q, qdd, bias, free_acc, total_acc,
                      residual_force, dv_acc, sensitivity)
            for key, value in zip(tracked, values):
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite derived field at step {step}: {key}")
                tracked[key].append(value)
            rows.append({
                "step":step, "pre_time_s":step*dt, "post_time_s":(step+1)*dt,
                "actuation_max_abs_Nm":float(np.max(np.abs(tau))),
                "matrix_condition_number":float(np.linalg.cond(M[0])),
                "matrix_symmetry_max_abs":float(np.max(np.abs(M-M.swapaxes(-1, -2)))),
                "solve_residual_max_abs":float(np.max(np.abs(np.einsum("eij,ej->ei", M, qdd)-(Q-C-G)))),
                "jacobian_velocity_max_abs_error":float(np.max(np.abs(np.einsum("elij,ej->eli", J, generalized_velocity)-velocity))),
                "raw_vs_lab_com_velocity_max_abs_error":float(np.max(np.abs(velocity-get_pre("data_body_com_vel_w")))),
                "recursive_bias_max_abs":float(np.max(np.abs(bias))),
                "candidate_foot_Fz_N":residual_force[0, foot_ids, 2].tolist(),
                "reported_acc_vs_delta_velocity_max_abs":float(np.max(np.abs(total_acc-dv_acc))),
                "finite_difference_foot_force_sensitivity_Fz_N":sensitivity[0, foot_ids, 2].tolist(),
            })
        arrays.update({key:np.stack(values) for key, values in tracked.items()})
        arrays["runtime_body_masses"] = masses
    foot_fz = arrays["model_residual_force"][:, 0, foot_ids, 2]
    body_property_limitation = (
        "All 25 bodies expose the checked composed zero-damping/gravity/limits/gyroscopic/retain-acceleration values after initialization; direct native per-link damping/limit getters are unavailable. "+angular_unit_note
        if all_body_properties_match else
        "Actual child-link body damping/max-velocity/retain-acceleration/gyroscopic settings are not fully captured; root USD attributes do not verify all child links. No unknown coefficient was fitted or presented as measured zero.")
    limitations = [
        body_property_limitation,
        "The model free acceleration uses captured M/C/G/J, actual joint actuation, zero captured joint damping/friction/armature and inactive/zero external-wrench composers. It is a known-input model, not a verified complete simulator FD baseline.",
        "Reported link acceleration is retained. Final velocities include a later momentum-conservation correction; delta-velocity/dt is recorded only as sensitivity, not a more accurate substitute for the solver increment.",
        "Pre instantaneous model free acceleration and post reported total acceleration do not prove an exact integrated solver-only decomposition.",
        "12 startup/contact steps (60 ms), no original Gym sensor oracle, no raw solver delta comparison, no behavior/learning or force-sensor equivalence result.",
    ]
    report = {
        "status":("stock_api_moving_state_model_residual_complete" if all_body_properties_match
                  else "conditional_moving_state_model_residual_complete"),
        "capture":str(capture_path.resolve()), "capture_sha256":hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "summary":str(summary_path.resolve()), "metadata":str(metadata_path.resolve()),
        "steps":len(rows), "physics_dt_s":dt, "foot_order":list(FOOT_ORDER),
        "actual_foot_masses_kg":masses[0, foot_ids].tolist(),
        "force_candidate_evaluated":True, "full_moving_state_model_evaluated":True,
        "complete_simulator_free_force_verified":False, "original_sensor_equivalence_verified":False,
        "all_body_composed_properties_match":all_body_properties_match,
        "expected_body_properties":expected_body_properties,
        "body_max_angular_velocity_units":angular_unit_note,
        "expected_max_angular_velocity_deg_s_before_USD_rounding":expected_max_angular_velocity_deg_s,
        "composed_angular_limit_matches_original_Gym_rad_s":all_body_properties_match and angular_limit_matches_original,
        "incomplete_body_property_records":incomplete_body_properties,
        "formula":"qdd_model = solve(M, [0_world_root_wrench, actual_tau]-C-G); a_free_model = J*qdd_model+b_CoM; candidate_Fz = measured_mass*(post_reported_total_acc-pre_model_free_acc).z",
        "recursive_bias_formula":"r=pc-pp; dw=wc-wp; dv=vc-vp-wp cross r; bc_w=bp_w+wp cross dw; bc_v=bp_v+bp_w cross r+wp cross(wp cross r)+2 wp cross dv+dw cross dv; root b=0",
        "source_reference":"PhysX commit 517a0073715120e114ee055b63b26c95e00d9039, forwardDynamic2.cu:1172-1207,2118-2121 (bias),2801-2912 (momentum correction),3189 (reported acceleration). Static semantic reference, not installed binary equivalence proof.",
        "external_forces_every_TGS_iteration":False,
        "per_step":rows,
        "candidate_foot_Fz_N_statistics":{
            "mean_by_foot":np.mean(foot_fz, axis=0).tolist(), "min_by_foot":np.min(foot_fz, axis=0).tolist(),
            "max_by_foot":np.max(foot_fz, axis=0).tolist(), "positive_counts_by_foot":np.sum(foot_fz>0, axis=0).tolist()},
        "jacobian_velocity_max_abs_error":max(r["jacobian_velocity_max_abs_error"] for r in rows),
        "solve_residual_max_abs":max(r["solve_residual_max_abs"] for r in rows),
        "limitations":limitations,
        "boundary":"CPU analysis of saved stock public API inputs only; no runtime/reward/engine edits, GPU execution, fitted coefficients or second-simulator FD clone.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    np.savez(output.with_suffix(".npz"), **arrays)
    lines = ["# Stock dynamics: moving-state model force residual", "",
             "The captured 12-step full floating-base calculation is complete; an exact simulator free-force baseline and original sensor equivalence remain unverified.", "",
             report["formula"], "", report["recursive_bias_formula"], "",
             f"J*v versus measured world CoM velocity: maximum absolute error {report['jacobian_velocity_max_abs_error']:.9g}. Full 24-coordinate solve residual: {report['solve_residual_max_abs']:.9g}. Actual foot mass: {masses[0, foot_ids].tolist()} kg.", "",
             "| Step | Applied torque max Nm | FR Fz N | FL Fz N | RR Fz N | RL Fz N |", "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['step']} | {row['actuation_max_abs_Nm']:.6f} | "+" | ".join(f"{x:.6f}" for x in row["candidate_foot_Fz_N"])+" |")
    lines.extend(["", "These are model residuals, not measured original Gym force-sensor outputs.", ""])
    lines.extend("- "+item for item in limitations)
    lines.extend(["", report["source_reference"], "", report["boundary"], ""])
    output.with_suffix(".md").write_text("\n".join(lines))
    return report


def analyze(capture_path, summary_path, output):
    source = json.loads(summary_path.read_text())
    arrays = {}
    report = {
        "capture": str(capture_path.resolve()),
        "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "summary": str(summary_path.resolve()),
        "status": "partial_zero_speed_diagnostic",
        "foot_order": list(FOOT_ORDER),
        "force_candidate_evaluated": False,
        "full_moving_state_reconstruction_evaluated": False,
        "missing_capture_requirements": MISSING_CAPTURE_REQUIREMENTS,
        "boundary": "Offline algebra on stock API capture only. No GPU, second simulator, runtime modification, fitted coefficients, sensor-equivalence result or reward integration.",
    }
    with np.load(capture_path, allow_pickle=False) as capture:
        M = require_array(capture, "before__get_generalized_mass_matrices")
        G = require_array(capture, "before__get_gravity_compensation_forces")
        C = require_array(capture, "before__get_coriolis_and_centrifugal_compensation_forces")
        J = require_array(capture, "before__get_jacobians")
        v = require_array(capture, "before__get_link_velocities")
        qd = require_array(capture, "before__get_dof_velocities")
        total_acc = require_array(capture, "after__get_link_accelerations")
        after_v = require_array(capture, "after__get_link_velocities")
        body_names = source["body_names"]
        num_envs, dofs = qd.shape
        generalized_size = dofs + 6
        if M.shape != (num_envs, generalized_size, generalized_size):
            raise ValueError("Capture is not a full floating-base generalized mass matrix")
        if J.shape != (num_envs, len(body_names), 6, generalized_size):
            raise ValueError("Jacobian shape does not cover all floating-base link rows")
        if G.shape != C.shape or G.shape != (num_envs, generalized_size):
            raise ValueError("Compensation force dimensions do not match M")
        if v.shape != total_acc.shape or v.shape != after_v.shape:
            raise ValueError("Link velocity/acceleration shapes differ")
        if np.any(v != 0) or np.any(qd != 0):
            raise ValueError("Moving-state capture needs recursive CoM bias; zero-bias diagnostic is not applicable")
        # This is an explicit hypothetical Q_free=0 response, not an implicit
        # claim that all external-force inputs were captured.
        rhs = -C-G
        qdd_free = np.linalg.solve(M, rhs[..., None])[..., 0]
        free_acc = np.einsum("elij,ej->eli", J, qdd_free)
        residual = total_acc-free_acc
        foot_ids = [body_names.index(name) for name in FOOT_ORDER]
        dt = float(source["dt"])
        vel_difference_acc = (after_v-v)/dt
        report.update(
            generalized_size=generalized_size,
            num_articulations=num_envs,
            physics_dt_s=dt,
            pre_velocities_exactly_zero=True,
            bias_scope="At zero generalized/link velocities, velocity-quadratic kinematic bias is zero; this does not exercise moving-state recursive bias.",
            assumed_generalized_force="Exactly zero for the reported algebraic response; actual complete free-force capture remains missing.",
            matrix_symmetry_max_abs=float(np.max(np.abs(M-M.swapaxes(-1, -2)))),
            matrix_condition_numbers=[float(np.linalg.cond(matrix)) for matrix in M],
            solve_residual_max_abs=float(np.max(np.abs(np.einsum("eij,ej->ei", M, qdd_free)-rhs))),
            root_jacobian_first_six_columns=J[:, 0, :, :6].tolist(),
            zero_generalized_force_free_qdd=qdd_free.tolist(),
            zero_speed_foot_free_linear_acceleration=free_acc[:, foot_ids, :3].tolist(),
            post_total_foot_linear_acceleration=total_acc[:, foot_ids, :3].tolist(),
            unscaled_foot_linear_residual_m_per_s2=residual[:, foot_ids, :3].tolist(),
            unscaled_foot_Fz_per_mass_m_per_s2=residual[:, foot_ids, 2].tolist(),
            reported_total_acc_vs_velocity_difference_max_abs=float(np.max(np.abs(total_acc-vel_difference_acc))),
            before_actuation_max_abs=float(np.max(np.abs(require_array(capture, "before__get_dof_actuation_forces")))),
            after_actuation_max_abs=float(np.max(np.abs(require_array(capture, "after__get_dof_actuation_forces")))),
            commanded_torque_max_abs=float(np.max(np.abs(require_array(capture, "command_torque_gym_order")))),
            units_note="Residual units are m/s^2; actual body masses are absent, so no force in newtons is reported. Post total minus pre instantaneous free acceleration is a diagnostic, not a proven solver-only measurement.",
        )
        arrays.update(zero_generalized_force_free_qdd=qdd_free,
                      zero_speed_link_free_acceleration=free_acc,
                      post_total_minus_pre_zero_speed_free_acceleration=residual,
                      post_total_acceleration=total_acc,
                      post_velocity_difference_acceleration=vel_difference_acc)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    np.savez(output.with_suffix(".npz"), **arrays)
    fz = report["unscaled_foot_Fz_per_mass_m_per_s2"][0]
    lines = ["# Stock dynamics: partial offline diagnostic", "",
             "Only the captured zero-speed, zero-generalized-force gravity response was evaluated. Moving-state recursive CoM bias and force reconstruction are not evaluated.", "",
             f"Full generalized dimension: {generalized_size}; matrix symmetry residual {report['matrix_symmetry_max_abs']:.3g}; linear solve residual {report['solve_residual_max_abs']:.3g}.", "",
             "| Foot | Post-total minus pre-free Z, m/s^2 |", "|---|---:|"]
    lines.extend(f"| {name} | {value:.9f} |" for name, value in zip(FOOT_ORDER, fz))
    lines.extend(["", report["units_note"], "", "Missing for the complete assigned reconstruction:", ""])
    lines.extend(f"- {key}: {value}" for key, value in MISSING_CAPTURE_REQUIREMENTS.items())
    lines.extend(["", report["boundary"], ""])
    output.with_suffix(".md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT/"outputs/isaac/stock-dynamics-capability-probe"
    parser.add_argument("--capture", type=Path, default=base/"capture.npz")
    parser.add_argument("--summary", type=Path, default=base/"summary.json")
    parser.add_argument("--metadata", type=Path, help="Expanded capture metadata enables the moving-state calculation")
    parser.add_argument("--expected-max-angular-velocity-deg-s", type=float, default=1000.,
                        help="Capture-specific expected Lab property; rounded to USD float32 before exact checking")
    parser.add_argument("--free-dynamics-step", type=int, help="Check one saved independent no-constraint sample at the fixed threshold")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/analysis/stock-solver-force-probe")
    args = parser.parse_args()
    if args.free_dynamics_step is not None:
        if args.metadata is None:
            parser.error("--free-dynamics-step requires --metadata")
        result = check_free_dynamics_step(args.capture,args.summary,args.metadata,args.output,args.free_dynamics_step)
        print(json.dumps({k:result[k] for k in ("status","passed","step","components")},indent=2))
        return
    result = (analyze_expanded(args.capture, args.summary, args.metadata, args.output,
                               args.expected_max_angular_velocity_deg_s) if args.metadata
              else analyze(args.capture, args.summary, args.output))
    print(json.dumps({key:result[key] for key in ("status", "force_candidate_evaluated", "solve_residual_max_abs")}, indent=2))


if __name__ == "__main__":
    main()
