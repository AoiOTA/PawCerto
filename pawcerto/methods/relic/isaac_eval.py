"""Authored interoperability glue for an explicit external ReLIC checkout."""

import hashlib
from pathlib import Path


UPSTREAM_REVISION = "27f8033c5064d32f049a17accb71cd1091422878"


def file_identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def compatible_spot_knee(original):
    """Adapt only the old constructor call; inherit original init and compute.

    The installed Lab collection owns physical armature/friction. The upstream
    Spot constructor supplies their zero defaults to its old positional super
    call; remove those slots, and route current actuator limits by name.
    No installed class or external module is modified.
    """
    base = original.__bases__[0]

    class ConstructorBridge(base):
        def __init__(self, cfg, joint_names, joint_ids, num_envs, device,
                     stiffness, damping, armature, friction, effort_limit, velocity_limit):
            if armature != 0 or friction != 0:
                raise ValueError("physical armature/friction must be configured through current Lab")
            super().__init__(
                cfg=cfg, joint_names=joint_names, joint_ids=joint_ids,
                num_envs=num_envs, device=device, stiffness=stiffness, damping=damping,
                actuator_effort_limit=self._external_ctor_limits[0],
                actuator_velocity_limit=self._external_ctor_limits[1],
            )

    class CompatibleSpotKnee(original, ConstructorBridge):
        def __init__(self, *args, actuator_effort_limit=None, actuator_velocity_limit=None, **kwargs):
            self._external_ctor_limits = (actuator_effort_limit, actuator_velocity_limit)
            super().__init__(*args, **kwargs)
            del self._external_ctor_limits

    return CompatibleSpotKnee


def recorder_config():
    """Use Lab's public recorder lifecycle to capture before automatic reset."""
    from isaaclab.managers import RecorderTerm, RecorderTermCfg
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, DatasetExportMode
    from isaaclab.utils import configclass

    def cpu(value):
        return value.detach().cpu().numpy().copy()

    class Capture(RecorderTerm):
        def record_pre_step(self):
            env = self._env
            action = env.action_manager.get_term("joint_pos")
            env._external_relic_before = {
                "raw_actor": cpu(action.raw_actions),
                "processed_leg_targets": cpu(action.processed_actions),
                "processed_arm_targets": cpu(action.arm_processed_actions),
                "selected_leg_targets": cpu(action.leg_processed_actions),
                "selected_leg": cpu(action.command.command_leg_idxs),
                "manipulation_active": cpu(action.command.command_leg),
                "base_command": cpu(env.command_manager.get_command("base_velocity")),
                "interlimb_command": cpu(action.command.command),
            }
            return None, None

        def record_post_step(self):
            env = self._env
            robot = env.scene["robot"]
            row = env._external_relic_before
            row.update({
                "root_state_w": cpu(robot.data.root_state_w),
                "body_state_w": cpu(robot.data.body_state_w),
                "joint_pos": cpu(robot.data.joint_pos),
                "joint_vel": cpu(robot.data.joint_vel),
                "joint_pos_target": cpu(robot.data.joint_pos_target),
                "applied_torque": cpu(robot.data.applied_torque),
                "reward": cpu(env.reward_buf),
                "terminated": cpu(env.reset_terminated),
                "truncated": cpu(env.reset_time_outs),
                "episode_length": cpu(env.episode_length_buf),
            })
            for name in env.termination_manager.active_terms:
                row["termination/" + name] = cpu(env.termination_manager.get_term(name))
            env._external_relic_records.append(row)
            return None, None

    @configclass
    class CaptureCfg(RecorderManagerBaseCfg):
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        capture = RecorderTermCfg(class_type=Capture)

    return CaptureCfg()
