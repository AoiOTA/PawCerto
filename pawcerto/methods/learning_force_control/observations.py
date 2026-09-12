# Sensor tensor semantics derived from official b1_gym/sensors at fixed revision.
# Copyright (c) 2022 MIT Improbable AI Lab. See LICENSE.
"""Official actor inputs. Force measurements are privileged supervision only.

commands[:,12:15] are requested force components, [:,22] the hybrid indicator;
measured/applied force is not concatenated into the deployed actor observation.
Callers provide supported-API sensor tensors and the resolved command scales.
"""
import torch


def actor_observation(projected_gravity, commands, command_scales, joint_pos,
                      default_joint_pos, joint_vel, actions, clock_inputs,
                      force_or_position_control, *, dof_pos_scale=1.0,
                      dof_vel_scale=0.05, control_only_z1=False):
    scaled_commands = commands * command_scales
    scaled_commands[force_or_position_control == 1, 15:18] = 0
    position = (joint_pos - default_joint_pos) * dof_pos_scale
    velocity = joint_vel * dof_vel_scale
    if control_only_z1:
        position = position.clone()
        velocity = velocity.clone()
        position[:, :12] = 0
        velocity[:, :12] = 0
    return torch.cat((projected_gravity, scaled_commands, position, velocity,
                      actions, clock_inputs), dim=-1)


def force_in_base_yaw(forces_world, base_yaw):
    """EeGripperForceSensor rotates applied gripper force by inverse base yaw."""
    c, s = torch.cos(base_yaw), torch.sin(base_yaw)
    x, y, z = forces_world.unbind(-1)
    return torch.stack((c*x + s*y, -s*x + c*y, z), -1)
