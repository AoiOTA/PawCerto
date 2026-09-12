"""Opt-in actual installed-Lab/external-source CPU interoperability checks."""

import os
from pathlib import Path
import sys

import pytest


@pytest.fixture
def external():
    path = os.environ.get("PAWCERTO_RELIC_EXTERNAL_ROOT")
    if not path:
        pytest.skip("requires explicitly supplied external research-licensed source")
    sys.path.insert(0, str(Path(path).resolve() / "source/relic"))
    pytest.importorskip("isaaclab")
    from relic.actuators import SpotKneeActuator, SpotKneeActuatorCfg
    from relic.assets.spot.constants import JOINT_PARAMETER_LOOKUP_TABLE
    cfg = SpotKneeActuatorCfg(
        joint_names_expr=[".*"], stiffness=100.0, damping=2.0,
        joint_parameter_lookup=JOINT_PARAMETER_LOOKUP_TABLE,
        enable_torque_speed_limit=True, min_delay=0, max_delay=0,
    )
    return SpotKneeActuator, cfg


def test_actual_constructor_and_original_speed_clipping(external):
    import torch
    from isaaclab.actuators.actuator_pd import RemotizedPDActuator
    from isaaclab.utils.types import ArticulationActions
    from pawcerto.methods.relic.isaac_eval import compatible_spot_knee

    original, cfg = external
    kwargs = dict(cfg=cfg, joint_names=["fl_kn", "fr_kn"], joint_ids=[0, 1],
                  num_envs=1, device="cpu", actuator_effort_limit=100.0,
                  actuator_velocity_limit=20.0)
    with pytest.raises(TypeError, match="actuator_effort_limit"):
        original(**kwargs)
    compatible = compatible_spot_knee(original)
    assert compatible.compute is original.compute
    actuator = compatible(**kwargs)
    baseline = RemotizedPDActuator(**kwargs)
    assert torch.equal(actuator._pos_torque_speed_data, torch.tensor(cfg.pos_torque_speed_limit))
    assert torch.equal(actuator._neg_torque_speed_data, torch.tensor(cfg.neg_torque_speed_limit))
    assert torch.all(actuator.stiffness == 100.0)
    assert torch.all(actuator.damping == 2.0)
    for speed in [-15.0, 0.0, 15.0]:
        position = torch.tensor([[1.0, 1.0]])
        velocity = torch.tensor([[speed, speed]])
        def desired():
            return ArticulationActions(joint_positions=torch.tensor([[10.0, -10.0]]),
                joint_velocities=torch.zeros(1, 2), joint_efforts=torch.zeros(1, 2))
        native = baseline.compute(desired(), position, velocity).joint_efforts
        expected = torch.clamp(native, min=actuator._neg_torque_speed_limit.compute(velocity),
                               max=actuator._pos_torque_speed_limit.compute(velocity))
        actual = actuator.compute(desired(), position, velocity).joint_efforts
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actuator.applied_effort, expected)


def test_external_play_config_and_public_recorder(external):
    import relic.tasks
    from relic.tasks.loco_manipulation.config.spot.spot_env_cfg import SpotInterlimbEnvCfg_PLAY
    from pawcerto.methods.relic.isaac_eval import recorder_config
    from isaaclab.managers.recorder_manager import DatasetExportMode

    cfg = SpotInterlimbEnvCfg_PLAY()
    assert cfg.scene.num_envs == 128
    assert not cfg.observations.policy.enable_corruption
    recorder = recorder_config()
    assert recorder.dataset_export_mode == DatasetExportMode.EXPORT_NONE


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("vector_force", [True, False])
def test_actual_external_ground_contact_singleton_batch(external, num_envs, vector_force):
    from types import SimpleNamespace
    import torch
    from relic.tasks.loco_manipulation.mdp.terminations import illegal_ground_contact

    # Current observed PhysX normal forces are scalar [N,B,F]; historical force
    # vectors use [N,B,F,3]. The source selects four bodies and one ground filter.
    forces = torch.zeros((num_envs, 4, 1, 3) if vector_force else (num_envs, 4, 1))
    if vector_force:
        forces[0, 2, 0, 1] = 2.0
    else:
        forces[0, 2, 0] = 2.0
    env = SimpleNamespace(scene=SimpleNamespace(sensors={
        "ground": SimpleNamespace(data=SimpleNamespace(force_matrix_w=forces))}))
    actual = illegal_ground_contact(env, 1.0, SimpleNamespace(name="ground", body_ids=[0, 1, 2, 3]))
    assert actual.shape == (num_envs,)
    assert actual.tolist() == [True] + [False] * (num_envs - 1)
