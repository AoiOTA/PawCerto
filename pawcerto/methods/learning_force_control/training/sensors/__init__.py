from .orientation_sensor import OrientationSensor
from .rc_sensor import RCSensor
from .joint_position_sensor import JointPositionSensor
from .joint_velocity_sensor import JointVelocitySensor
from .action_sensor import ActionSensor
from .clock_sensor import ClockSensor
from .body_velocity_sensor import BodyVelocitySensor
from .joint_dynamics_sensor import JointDynamicsSensor
from .ee_gripper_force_sensor import EeGripperForceSensor
from .friction_sensor import FrictionSensor
from .ee_gripper_position_sensor import EeGripperPositionSensor
from .ee_gripper_target_position_sensor import EeGripperTargetPositionSensor
ALL_SENSORS = {"OrientationSensor": OrientationSensor, "RCSensor": RCSensor, "JointPositionSensor": JointPositionSensor, "JointVelocitySensor": JointVelocitySensor, "ActionSensor": ActionSensor, "ClockSensor": ClockSensor, "BodyVelocitySensor": BodyVelocitySensor, "JointDynamicsSensor": JointDynamicsSensor, "EeGripperForceSensor": EeGripperForceSensor, "FrictionSensor": FrictionSensor, "EeGripperPositionSensor": EeGripperPositionSensor, "EeGripperTargetPositionSensor": EeGripperTargetPositionSensor}
