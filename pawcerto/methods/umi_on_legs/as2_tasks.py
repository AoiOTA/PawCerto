"""Finite, CPU-only AS2/Piper-H EE task construction from an existing asset family.

The stored base/joint configurations witness FK reachability only. They are not
contact-consistent locomotion, policy inputs, supervision, or executed states.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from pawcerto.mujoco.as2_piper_asset import transform, urdf_fk
from . import PoseSequence


def _read_json(path):
    return json.loads(Path(path).read_text())


def _source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _relative_path(value, owner):
    path = Path(value)
    return path if path.is_absolute() else Path(owner).parent / path


def _finite(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return array


def _validate_recipe_binding(recipe, assembly, assembly_path, nominal, training_path):
    """The executing robot/default pose must match the geometry producing targets."""
    binding = recipe.get("pawcerto_robot", {})
    expected = {"name": "as2_piper", "root_body": assembly["mount"]["parent"],
                "joint_names": assembly["controlled_joint_order"],
                "tcp_body": assembly["tcp"]["body"], "tcp_xyz": assembly["tcp"]["xyz"]}
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("Training robot binding differs from assembly joints/root/TCP")
    rotation = _finite(binding.get("tcp_rotation"), (3, 3), "Binding TCP rotation")
    expected_rotation = transform(assembly["tcp"]["xyz"], assembly["tcp"]["rpy"])[:3, :3]
    if not np.allclose(rotation, expected_rotation, atol=1e-12, rtol=0):
        raise ValueError("Training robot binding differs from assembly TCP rotation")
    if binding.get("nominal_input", {}).get("sha256") != _source(assembly_path)["sha256"]:
        raise ValueError("Training robot binding differs from the selected assembly identity")
    bound_urdf = _relative_path(binding["urdf_path"], training_path)
    if (_source(bound_urdf)["sha256"] != nominal["urdf_sha256"]
            or binding.get("source_urdf", {}).get("sha256") != nominal["urdf_sha256"]):
        raise ValueError("Training robot binding URDF differs from the nominal family asset")
    env = recipe["env"]
    offsets = [env["controller"]["offset"]["data"],
               env["state_obs"]["dof_pos"]["offset"]["data"],
               env["privileged_state_obs"]["dof_pos"]["offset"]["data"]]
    if any(not np.array_equal(offset, assembly["default_joint_positions"]) for offset in offsets):
        raise ValueError("Training default joint/controller/observation offsets differ from assembly")
    if env["tasks"]["reaching"]["link_name"] != assembly["tcp"]["body"]:
        raise ValueError("Training reaching link differs from assembly TCP body")


def validate_task_config(task, assembly, robot, recipe):
    """Reject incompatible timing, starting states and waypoints before writing."""
    dt = float(task["dt"])
    duration = float(task["episode_length_s"])
    frames = task["frames"]
    if (not np.isfinite([dt, duration]).all() or dt <= 0 or duration <= 0
            or not isinstance(frames, int) or frames < 3
            or not np.isclose(frames * dt, duration, atol=1e-12, rtol=0)
            or int(duration / dt) != frames):
        raise ValueError("Episode contract requires frames == episode_length_s / dt")
    if (recipe["env"]["cfg"]["env"]["episode_length_s"] != duration
            or recipe["env"]["cfg"]["sim"]["dt"] != dt):
        raise ValueError("Episode contract differs from the supplied training configuration")
    names = assembly["controlled_joint_order"]
    if len(names) != 18 or len(set(names)) != 18:
        raise ValueError("Expected the existing 12 leg + 6 arm joint order")
    default = _finite(assembly["default_joint_positions"], (18,), "Default joints")
    limits = []
    for name, position in zip(names, default):
        joint = robot.find(f"joint[@name='{name}']")
        if joint is None or joint.get("type") != "revolute":
            raise ValueError(f"Missing source revolute joint: {name}")
        limit = joint.find("limit")
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        if not np.isfinite([lo, hi]).all() or lo > hi or not lo <= position <= hi:
            raise ValueError(f"Default configuration outside joint limits: {name}")
        limits.append([lo, hi])
    arm_limits = np.asarray(limits[12:])
    times = np.arange(frames) * dt
    knots = np.asarray(task["knot_times_s"], dtype=float)
    if (knots.ndim != 1 or len(knots) < 4 or not np.isfinite(knots).all()
            or knots[0] != 0 or knots[-1] != times[-1]
            or np.any(np.diff(knots) <= 0)):
        raise ValueError("Knots must increase from zero to the final sampled time")
    poses = {name: _finite(q, (6,), f"Pose {name}")
             for name, q in task["poses_rad"].items()}
    for name, q in poses.items():
        if np.any(q < arm_limits[:, 0]) or np.any(q > arm_limits[:, 1]):
            raise ValueError(f"Pose {name} exceeds source joint limits")
    trajectories = task["trajectories"]
    if len(trajectories) != 8 or len({t["name"] for t in trajectories}) != 8:
        raise ValueError("This finite task family requires eight uniquely named trajectories")
    for trajectory in trajectories:
        waypoints = trajectory["waypoints"]
        if len(waypoints) != len(knots) or any(p not in poses for p in waypoints):
            raise ValueError(f"Invalid waypoints for {trajectory['name']}")
        # Phase-zero reset must start and end in the supplied nominal arm pose.
        for index in [0, 1, -2, -1]:
            if not np.array_equal(poses[waypoints[index]], default[12:]):
                raise ValueError("Initial and terminal holds must match default arm joints")
        if "translation_direction" in trajectory:
            direction = _finite(trajectory["translation_direction"], (3,), "Translation direction")
            if direction[2] != 0 or not np.isclose(np.linalg.norm(direction), 1):
                raise ValueError("Translation direction must be a unit vector in the XY plane")
    _finite(task["root_xy_m"], (2,), "Root XY")
    clearance = float(task["default_foot_clearance_m"])
    translation = task["translation"]
    start, end = float(translation["start_s"]), float(translation["end_s"])
    distance = float(translation["distance_m"])
    required = float(translation["minimum_required_root_displacement_m"])
    if (not np.isfinite([clearance, start, end, distance, required]).all()
            or clearance < 0 or distance <= 0 or required < 0
            or not knots[1] <= start < end <= knots[-2]):
        raise ValueError("Invalid geometric clearance or translation interval/displacement")
    return times, knots, poses, arm_limits


def _smooth(u):
    u = np.clip(u, 0, 1)
    return u ** 3 * (10 + u * (-15 + 6 * u))


def _joint_path(times, knots, waypoints):
    result = np.zeros((len(times), 6))
    for i in range(len(knots) - 1):
        mask = (times >= knots[i]) & (times <= knots[i + 1])
        alpha = _smooth((times[mask] - knots[i]) / (knots[i + 1] - knots[i]))
        result[mask] = waypoints[i] + alpha[:, None] * (waypoints[i + 1] - waypoints[i])
    return result


def _arm_chain(robot, assembly):
    joints = {j.find("child").get("link"): j for j in robot.findall("joint")}
    chain = []
    child = assembly["tcp"]["body"]
    while child != assembly["mount"]["child"]:
        if child not in joints or any(j is joints[child] for j, _ in chain):
            raise ValueError("TCP does not form a chain below the configured arm mount")
        joint = joints[child]
        origin = joint.find("origin")
        matrix = transform(np.fromstring(origin.get("xyz", "0 0 0"), sep=" "),
                           np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
        if joint.get("type") not in {"revolute", "fixed"}:
            raise ValueError("Task construction expects a fixed/revolute arm chain")
        chain.append((joint, matrix))
        child = joint.find("parent").get("link")
    return chain[::-1]


def _fk_batch(q, base_xyz, chain, arm_names, mount, tcp):
    count = len(q)
    result = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    result[:, :3, 3] = base_xyz
    result = result @ mount
    for joint, origin in chain:
        result = result @ origin
        if joint.get("type") == "revolute":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            motion = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            angles = q[:, arm_names.index(joint.get("name")), None]
            motion[:, :3, :3] = Rotation.from_rotvec(angles * axis).as_matrix()
            result = result @ motion
    return result @ tcp


def _axis_angle(rotations):
    canonical = Rotation.from_matrix(rotations).as_rotvec()
    result = canonical.copy()
    for index in range(1, len(result)):
        length = np.linalg.norm(canonical[index])
        if length > 1e-10:
            axis = canonical[index] / length
            candidates = np.array([canonical[index] + winding * 2 * np.pi * axis
                                   for winding in [-1, 0, 1]])
            result[index] = candidates[np.argmin(np.linalg.norm(candidates - result[index - 1], axis=1))]
    return result


def _foot_height(robots, names, default, clearance):
    records = []
    for variant_name, robot in robots.items():
        poses = urdf_fk(robot, dict(zip(names, default)), [0, 0, 0], [1, 0, 0, 0])
        bottoms = []
        for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
            shape = robot.find(f"link[@name='{name}']/collision")
            sphere = shape.find("geometry/sphere")
            if sphere is None:
                raise ValueError("Expected existing AS2 foot collision spheres")
            local = np.fromstring(shape.find("origin").get("xyz", "0 0 0"), sep=" ")
            center = poses[name][:3, :3] @ local + poses[name][:3, 3]
            bottoms.append(float(center[2] - float(sphere.get("radius"))))
        records.append({"variant": variant_name, "minimum_foot_z_at_root_zero_m": min(bottoms)})
    height = max(-r["minimum_foot_z_at_root_zero_m"] for r in records) + clearance
    for record in records:
        record["root_height_for_zero_foot_gap_m"] = -record["minimum_foot_z_at_root_zero_m"]
        record["initial_foot_clearance_at_common_height_m"] = height + record["minimum_foot_z_at_root_zero_m"]
    return height, records


def _stats(values):
    values = np.asarray(values)
    return {"min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def _norm_stats(values):
    return _stats(np.linalg.norm(values, axis=-1))


def build_as2_ee_tasks(task_config, assembly_config, training_config, output):
    """Build the finite task, validating input and the real CPU loader before writes."""
    task_config, assembly_config, training_config = map(Path, [task_config, assembly_config, training_config])
    output = Path(output).resolve()
    filenames = ["as2_general_ee.pkl", "construction-witness.npz", "training-config-candidate.json", "manifest.json"]
    if any((output / name).exists() for name in filenames):
        raise ValueError("Output already contains task artifacts; choose a new output directory")
    task, assembly, recipe = map(_read_json, [task_config, assembly_config, training_config])
    family_ref = recipe["pawcerto_asset_family"]["manifest"]
    family_path = _relative_path(family_ref["path"], training_config).resolve()
    family = _read_json(family_path)
    expected = family_ref.get("sha256")
    if expected and _source(family_path)["sha256"] != expected:
        raise ValueError("Asset family manifest differs from the training configuration")
    if _source(assembly_config)["sha256"] != family["source"]["nominal_config_sha256"]:
        raise ValueError("Assembly configuration differs from the existing asset family")
    nominal = [v for v in family["variants"] if v["name"] == "nominal"]
    if len(nominal) != 1:
        raise ValueError("Expected one nominal source URDF in the existing asset family")
    _validate_recipe_binding(recipe, assembly, assembly_config, nominal[0], training_config)
    robots, urdf_sources = {}, []
    for entry in family["variants"]:
        path = _relative_path(entry["source_urdf_path"], family_path)
        identity = _source(path)
        if identity["sha256"] != entry["source_urdf_sha256"]:
            raise ValueError(f"Family source URDF hash differs for {entry['name']}")
        if entry["name"] in robots:
            raise ValueError("Duplicate variant name in asset family")
        robots[entry["name"]] = ET.parse(path).getroot()
        urdf_sources.append(identity)
    robot = robots["nominal"]
    times, knots, poses, arm_limits = validate_task_config(task, assembly, robot, recipe)
    names = assembly["controlled_joint_order"]
    default = np.asarray(assembly["default_joint_positions"])
    height, foot_records = _foot_height(robots, names, default, task["default_foot_clearance_m"])
    base_origin = np.array([*task["root_xy_m"], height])
    mount = transform(assembly["mount"]["xyz"], assembly["mount"]["rpy"])
    tcp = transform(assembly["tcp"]["xyz"], assembly["tcp"]["rpy"])
    chain = _arm_chain(robot, assembly)
    bound = np.linalg.norm(mount[:3, 3]) + sum(np.linalg.norm(t[:3, 3]) for _, t in chain) + np.linalg.norm(tcp[:3, 3])
    episodes, records, witness_q, witness_base, witness_tcp = [], [], [], [], []
    max_fk_error = 0.
    dt = task["dt"]
    for spec in task["trajectories"]:
        waypoints = np.asarray([poses[name] for name in spec["waypoints"]])
        q = _joint_path(times, knots, waypoints)
        if np.any(q < arm_limits[:, 0]) or np.any(q > arm_limits[:, 1]):
            raise ValueError(f"Generated trajectory exceeds joint limits: {spec['name']}")
        base = np.repeat(base_origin[None], len(times), axis=0)
        if "translation_direction" in spec:
            translation = task["translation"]
            alpha = _smooth((times - translation["start_s"]) / (translation["end_s"] - translation["start_s"]))
            base += alpha[:, None] * np.array(spec["translation_direction"])[None] * translation["distance_m"]
        fk = _fk_batch(q, base, chain, names[12:], mount, tcp)
        position, rotation = fk[:, :3, 3], fk[:, :3, :3]
        axis_angle = _axis_angle(rotation)
        full_q = np.c_[np.repeat(default[None, :12], len(times), axis=0), q]
        for index in np.unique(np.linspace(0, len(times) - 1, 13, dtype=int)):
            independent = urdf_fk(robot, dict(zip(names, full_q[index])), base[index], [1, 0, 0, 0])
            expected = independent[assembly["tcp"]["body"]] @ tcp
            max_fk_error = max(max_fk_error, float(np.max(abs(expected - fk[index]))))
            np.testing.assert_allclose(expected, fk[index], atol=1e-12, rtol=0)
        np.testing.assert_allclose(Rotation.from_rotvec(axis_angle).as_matrix(), rotation, atol=1e-12, rtol=0)
        if not np.isfinite(fk).all() or not np.isfinite(axis_angle).all():
            raise ValueError("Nonfinite generated pose")
        linear = np.diff(position, axis=0) / dt
        acceleration = np.diff(position, n=2, axis=0) / dt ** 2
        angular = Rotation.from_matrix(rotation[1:] @ rotation[:-1].transpose(0, 2, 1)).as_rotvec() / dt
        endpoint = float(np.linalg.norm(position[-1] - position[0]))
        minimum_translation = max(0, endpoint - 2 * bound)
        if ("translation_direction" in spec
                and minimum_translation <= task["translation"]["minimum_required_root_displacement_m"]):
            raise ValueError("Translation endpoints do not establish the requested root movement lower bound")
        records.append({
            "name": spec["name"], "description": spec["description"],
            "joint_waypoints_rad": waypoints.tolist(), "reference_base_displacement_m": (base[-1] - base[0]).tolist(),
            "linear_speed_m_s": _norm_stats(linear), "linear_acceleration_m_s2": _norm_stats(acceleration),
            "angular_speed_rad_s": _norm_stats(angular), "angular_acceleration_rad_s2": _norm_stats(np.diff(angular, axis=0) / dt),
            "endpoint_distance_m": endpoint,
            "conservative_min_root_translation_for_exact_endpoint_tracking_m": minimum_translation,
        })
        episodes.append({"t": times.copy(), "ee_pos": position, "ee_axis_angle": axis_angle})
        witness_q.append(full_q)
        witness_base.append(base)
        witness_tcp.append(fk)

    candidate = deepcopy(recipe)
    sampler = candidate["env"]["tasks"]["reaching"]["sequence_sampler"]
    sampler.pop("trajectory_selection", None)
    sampler.update(file_path=str(output / filenames[0]), episode_length_s=task["episode_length_s"],
                   dt=dt, planar_center=False, add_random_height_range=None)
    candidate["env"]["cfg"]["init_state"]["pos"] = base_origin.tolist()
    candidate["env"]["cfg"]["init_state"]["rot"] = [0., 0., 0., 1.]
    candidate["runner"]["init_at_random_ep_len"] = False
    data_bytes = pickle.dumps(episodes, protocol=4)
    # Exercise the actual loader against these exact bytes before publishing files.
    with tempfile.TemporaryDirectory(prefix="as2-ee-loader-") as temporary:
        path = Path(temporary) / filenames[0]
        path.write_bytes(data_bytes)
        loaded = PoseSequence(path, sampler, "cpu")
        positions, rotations = loaded.select_ids(list(range(len(episodes))), seed=0)
    if tuple(positions.shape) != (8, len(times), 3) or tuple(rotations.shape) != (8, len(times), 3, 3):
        raise ValueError("Actual loader changed the episode shape")
    np.testing.assert_allclose(positions.numpy(), np.stack([e["ee_pos"] for e in episodes]), atol=2e-7, rtol=0)
    np.testing.assert_allclose(rotations.numpy(), np.stack(witness_tcp)[:, :, :3, :3], atol=1e-6, rtol=0)
    manifest = {
        "schema_version": 1, "task": "Finite AS2/Piper-H general EE tracking candidate",
        "sources": [_source(p) for p in [task_config, assembly_config, training_config, family_path]] + urdf_sources,
        "sampler_contract": {
            "episode_length_s": task["episode_length_s"], "dt": dt, "frames": len(times),
            "last_sample_time_s": float(times[-1]), "planar_center": False, "add_random_height_range": None,
            "init_at_random_ep_len": False, "terminal_padding_frames": 0,
            "phase": "Begin at phase zero; time-only phase randomization does not reset a matching robot state.",
            "coordinates": "Fixed environment-local world targets: +X forward, +Y left, +Z up. No XY recenter or random height.",
            "cli": "Pass both --config training-config-candidate.json and --trajectory as2_general_ee.pkl to the existing consumer.",
        },
        "world_task_anchor": {
            "reference_base_xyz": base_origin.tolist(), "reference_base_quat_wxyz": [1, 0, 0, 0],
            "reset_noise": candidate["env"]["cfg"]["init_state"],
            "joint_reset_range_scale": candidate["env"]["dof_pos_reset_range_scale"],
            "unchanged_reward_root_height_target_m": candidate["env"]["constraints"]["root_height"]["target_height"],
            "default_foot_height_audit": foot_records,
        },
        "root_tcp_triangle_upper_bound_m": float(bound), "trajectories": records,
        "tcp": assembly["tcp"], "validation": {"max_batch_vs_independent_urdf_fk_error": max_fk_error,
            "actual_loader_positions_shape": list(positions.shape), "actual_loader_rotations_shape": list(rotations.shape),
            "joint_limits_passed": True, "simulator_steps": 0, "optimizer_steps": 0},
        "learning_budget": "Runner iteration counts are inherited configuration only; no learning run or budget is authorized by this generator.",
        "limits": [
            "FK witnesses are not actor inputs or supervision. Translated fixed legs do not constitute a gait.",
            "Foot clearance applies to the default zero-noise configuration; retained reset noise is unvalidated.",
            "Root-height reward target is retained. Reference height is geometric clearance, not equilibrium.",
            "Mass/COM/payload family remains in the supplied recipe; actual task-by-load visitation is unverified.",
            "No collision, swept volume, support, torque, full-load dynamics, hardware calibration or control-success proof is generated.",
            "Original tossing data, PoseSequence behavior and production training defaults are unchanged.",
        ],
    }
    # All input, kinematics and actual-loader checks above precede output writes.
    output.mkdir(parents=True, exist_ok=True)
    (output / filenames[0]).write_bytes(data_bytes)
    np.savez_compressed(output / filenames[1], time_s=times,
                        reference_base_xyz=np.stack(witness_base),
                        reference_base_quat_wxyz=np.broadcast_to([1, 0, 0, 0], (8, len(times), 4)),
                        reference_joint_q=np.stack(witness_q), reference_tcp_pose=np.stack(witness_tcp),
                        joint_names=np.array(names), trajectory_names=np.array([s["name"] for s in task["trajectories"]]))
    (output / filenames[2]).write_text(json.dumps(candidate, indent=2, allow_nan=False) + "\n")
    manifest["artifacts"] = {name: _source(output / name) for name in filenames[:3]}
    (output / filenames[3]).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
