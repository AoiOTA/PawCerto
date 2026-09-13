"""Real source-FK and loader contracts; no simulator, GPU or learning."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pawcerto.methods.umi_on_legs import PoseSequence
from pawcerto.methods.umi_on_legs.as2_tasks import build_as2_ee_tasks
from pawcerto.methods.umi_on_legs.robot_binding import as2_config
from pawcerto.mujoco.as2_piper_asset import CONFIG, assemble, transform, urdf_fk

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    assembly = json.loads(CONFIG.read_text())
    # A fixture mount below 0.1 m preserves the declared >0.4 m translation bound.
    assembly["mount"]["xyz"] = [0., 0., .09]
    assembly_path = tmp_path / "assembly.json"
    assembly_path.write_text(json.dumps(assembly))
    urdf = tmp_path / "nominal.urdf"
    ET.ElementTree(assemble(config=assembly)).write(urdf)
    # Exercise relative paths as well as the real source kinematics.
    family = {"source": {"nominal_config_sha256": hashlib.sha256(assembly_path.read_bytes()).hexdigest()},
              "variants": [{"name": "nominal", "source_urdf_path": urdf.name,
                            "source_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                            "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest()}]}
    family_path = tmp_path / "family.json"
    family_path.write_text(json.dumps(family))
    recipe = as2_config(json.loads((ROOT / "reference/checkpoints/tossing/ours/config.json").read_text()),
                        nominal_path=assembly_path, urdf_path=urdf)
    recipe["pawcerto_asset_family"] = {"manifest": {"path": family_path.name}}
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    task = json.loads((ROOT / "configs/as2_general_ee_tasks.json").read_text())
    return assembly_path, recipe_path, task, urdf


def generate(inputs, tmp_path, task):
    assembly, recipe, _, _ = inputs
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    return build_as2_ee_tasks(path, assembly, recipe, tmp_path / "result")


@pytest.mark.parametrize("error,match", [
    ("joint_limit", "joint limits"),
    ("episode", "Episode contract"),
    ("translation", "root movement lower bound"),
])
def test_invalid_contract_never_writes_task(inputs, tmp_path, error, match):
    task = deepcopy(inputs[2])
    if error == "joint_limit":
        task["poses_rad"]["high"][1] = 100.
    elif error == "episode":
        task["frames"] -= 1
    else:
        task["translation"]["distance_m"] = .1
    with pytest.raises(ValueError, match=match):
        generate(inputs, tmp_path, task)
    assert not (tmp_path / "result").exists()


def test_source_fk_actual_loader_and_recipe_preservation(inputs, tmp_path):
    manifest = generate(inputs, tmp_path, inputs[2])
    output = tmp_path / "result"
    recipe = json.loads((output / "training-config-candidate.json").read_text())
    sampler = recipe["env"]["tasks"]["reaching"]["sequence_sampler"]
    pos, rot = PoseSequence(output / "as2_general_ee.pkl", sampler, "cpu").select_ids(list(range(8)))
    witness = np.load(output / "construction-witness.npz")
    robot = ET.parse(inputs[3]).getroot()
    assembly = json.loads(inputs[0].read_text())
    tcp = transform(assembly["tcp"]["xyz"], assembly["tcp"]["rpy"])
    # An independent source FK sample on the final translation endpoint.
    task, frame = 7, 3399
    poses = urdf_fk(robot, dict(zip(witness["joint_names"], witness["reference_joint_q"][task, frame])),
                   witness["reference_base_xyz"][task, frame], [1, 0, 0, 0])
    expected = poses[assembly["tcp"]["body"]] @ tcp
    np.testing.assert_allclose(pos[task, frame], expected[:3, 3], atol=2e-7, rtol=0)
    np.testing.assert_allclose(rot[task, frame], expected[:3, :3], atol=1e-6, rtol=0)
    assert pos.shape == (8, 3400, 3) and rot.shape == (8, 3400, 3, 3)
    assert sampler["planar_center"] is False and sampler["add_random_height_range"] is None
    assert recipe["runner"]["init_at_random_ep_len"] is False
    original = json.loads(inputs[1].read_text())
    assert recipe["env"]["dof_pos_reset_range_scale"] == original["env"]["dof_pos_reset_range_scale"]
    assert recipe["env"]["constraints"] == original["env"]["constraints"]
    assert recipe["runner"]["max_iterations"] == original["runner"]["max_iterations"]
    assert all(t["conservative_min_root_translation_for_exact_endpoint_tracking_m"] > .4
               for t in manifest["trajectories"] if t["name"].startswith("translate"))


@pytest.mark.parametrize("field", ["assembly_identity", "tcp", "controller_offset"])
def test_inconsistent_execution_recipe_rejected_before_output(inputs, tmp_path, field):
    path = inputs[1]
    recipe = json.loads(path.read_text())
    if field == "assembly_identity":
        recipe["pawcerto_robot"]["nominal_input"]["sha256"] = "old-assembly"
    elif field == "tcp":
        recipe["pawcerto_robot"]["tcp_xyz"][2] += .01
    else:
        recipe["env"]["controller"]["offset"]["data"][12] += .1
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="differs|differ"):
        generate(inputs, tmp_path, inputs[2])
    assert not (tmp_path / "result").exists()


def test_changed_non_nominal_source_rejected_before_height_use(inputs, tmp_path):
    path = tmp_path / "family.json"
    family = json.loads(path.read_text())
    variant_path = tmp_path / "legs105.urdf"
    variant_path.write_bytes(inputs[3].read_bytes())
    variant = deepcopy(family["variants"][0])
    variant.update(name="legs105", source_urdf_path=variant_path.name)
    family["variants"].append(variant)
    path.write_text(json.dumps(family))
    # Change a consumed non-nominal foot radius without altering the declared hash.
    robot = ET.parse(variant_path)
    sphere = robot.find("link[@name='FL_foot']/collision/geometry/sphere")
    sphere.set("radius", str(float(sphere.get("radius")) + .01))
    robot.write(variant_path)
    with pytest.raises(ValueError, match="Family source URDF hash differs for legs105"):
        generate(inputs, tmp_path, inputs[2])
    assert not (tmp_path / "result").exists()
