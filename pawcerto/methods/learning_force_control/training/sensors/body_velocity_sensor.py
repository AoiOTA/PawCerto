# Copyright (c) 2022 MIT Improbable AI Lab. See ../../LICENSE.
# Fixed original sensor; simulator math imports replaced by tensor algebra.
from .sensor import Sensor

from ..torch_math import *
import torch
from ..math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift

class BodyVelocitySensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        return self.env.base_lin_vel
    
    # Privileged sensor input does not contain noise
    # def get_noise_vec(self):
    #     import torch
    #     return torch.zeros(1, device=self.env.device)

    def get_dim(self):
        return 3
